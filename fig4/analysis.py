import numpy as np
import torch
from tqdm import trange, tqdm 
from sklearn.decomposition import TruncatedSVD

def circ_dist(xpos):
    dd = torch.abs(xpos.unsqueeze(0) - xpos.unsqueeze(1))
    dxmin = torch.minimum(dd, 1-dd)
    return dxmin 

def strong_pairs(ypos, xpos, spks=None, cov=None, circular=False, pstrong=0.01, bin_size=100, 
                 nbins=30, dist_min=30, device=torch.device("cuda")):
    """ spks is neurons by time; assumes spks is z-scored in time 
    
    Args:
        ypos: numpy array or torch tensor of y positions
        xpos: numpy array or torch tensor of x positions
        spks: numpy array of neurons by time
        cov: covariance matrix (computed from spks if not provided)
        circular: compute distances using circular coordinates
        pstrong: fraction of connections defined as strong pairs
        bin_size: bin size for distance
        nbins: number of bins
        dist_min: minimum distance
        device: torch device to use

    Returns:
        dbin_strong: strong pairs in each bin
        dbin_other: other pairs in each bin
        cov_bin: covariance in each bin
        drand: random subset of distances
        crand: random subset of covariances    
    """

    if cov is None:
        if spks is not None:
            if device.type == "cuda":
                spks_gpu = torch.from_numpy(spks).to(device)
                cov = (spks_gpu @ spks_gpu.T) / spks_gpu.shape[1]
                cov -= torch.diag(torch.diag(cov))
                cov = cov.cpu().numpy()
            else:
                cov = (spks @ spks.T) / spks.shape[1]
                cov -= np.diag(np.diag(cov))
        else:
            raise ValueError("either spks or cov must be provided")

    if isinstance(xpos, np.ndarray):
        xpos = torch.from_numpy(xpos).to(device)
        ypos = torch.from_numpy(ypos).to(device)
    
    # remove neuron pairs within dist_min of each other
    if circular:
        dxmin = circ_dist(xpos)
        dymin = circ_dist(ypos)
        dist = dymin**2 + dxmin**2
        dist **=0.5
    else:
        dist = (xpos.unsqueeze(0) - xpos.unsqueeze(1))**2 
        dist += (ypos.unsqueeze(0) - ypos.to(device).unsqueeze(1))**2
        dist **= 0.5
    dist_filter = (dist > dist_min).cpu().numpy()
        
    dist = dist.cpu().numpy()
    dist[~dist_filter] = np.nan
    cov_filter = cov.copy() 
    cov_filter[~dist_filter] = -np.inf

    # sort covariance to get top pairs
    isort = cov_filter.argsort(axis=1)[:, ::-1]
    nstrong = int(np.round(pstrong * cov.shape[0]))

    # distance bin for each neuron pair
    dist_bin = np.floor(dist / bin_size)
    dist_bin[~np.isnan(dist_bin)].max()
    
    # distance bins for strong pairs vs other pairs
    dist_strong = np.array([dist_bin[i, isort[i, :nstrong]] for i in range(cov.shape[0])])
    dist_other = np.array([dist_bin[i, isort[i, nstrong:]] for i in range(cov.shape[0])])
    
    # number of strong pairs and other pairs in each distance bin
    dbin_strong = np.zeros((cov.shape[0], nbins), "uint16")
    dbin_other = np.zeros((cov.shape[0], nbins), "uint16")
    for d in range(nbins):
        dbin_strong[:,d] = (dist_strong == d).sum(axis=1)
        dbin_other[:,d] = (dist_other == d).sum(axis=1)

    # mean covariance in each distance bin
    cov_bin = np.zeros((cov.shape[0], nbins), "float32")
    for d in range(nbins):
        d0 = dist_bin == d
        cov_bin[:, d] = np.array([np.nanmean(cov_filter[i, d0[i]], axis=-1) 
                                    for i in range(cov.shape[0])])
    
    # random subset of pairwise distances and covariances
    irand = np.random.randint(0, cov.shape[0]**2, size=(20000,))
    drand = dist.flatten()[irand]
    crand = cov_filter.flatten()[irand]

    return dbin_strong, dbin_other, cov_bin, drand, crand

def neural_strong_pairs(files, device=torch.device("cpu")):
    """ compute strong pairs and PCs for neural data 

        saves data to "strongpairs_data_all.npy"

    Args:
        files: list of file names with "spks" and "xpos" and "ypos" fields
        device: torch device to use

    Returns:
        A tuple of (pcs_all, ypos_all, xpos_all, var_all,
                    dbin_strong_all, dbin_other_all, cov_bin_all, drand_all, crand_all)
    
    """
    pcs_all = []
    ypos_all = []
    xpos_all = []
    dbin_strong_all = []
    dbin_other_all = []
    cov_bin_all = []
    drand_all = []
    crand_all = []
    var_all = np.zeros(len(files))
    bin_size = 200
    nbins = 15
    pstrong = 0.01
    dist_min = 20
    for i, f in tqdm(enumerate(files)):
        dat = np.load(f)
        spks = dat["spks"] if "spks" in dat else dat["sp"] 
        spks = spks.astype("float32")
        ypos = dat["ypos"].astype("float32")
        xpos = dat["xpos"].astype("float32")
        print(f">>> {f.stem}, n_neurons = {spks.shape[0]}, nt = {spks.shape[1]} ({spks.shape[1]/3/60:.1f} minutes)")

        # zscore activity
        spks -= spks.mean(axis=1, keepdims=True)
        spks /= spks.std(axis=1, keepdims=True)

        # compute PCs
        V = TruncatedSVD(n_components=100).fit_transform(spks.T)
        V /= (V**2).sum(axis=0)**0.5
        U = spks @ V
        pcs_all.append(U)
        ypos_all.append(ypos)
        xpos_all.append(xpos)

        # compute ratio of variance of PC weights near center-of-mass vs all weights
        ypos = torch.from_numpy(ypos)
        xpos = torch.from_numpy(xpos)
        U = torch.from_numpy(U)
        com_y = ((U**2) * ypos.unsqueeze(1)).mean(axis=0) / (U**2).mean(axis=0)
        com_x = ((U**2) * xpos.unsqueeze(1)).mean(axis=0) / (U**2).mean(axis=0)
        ydist = torch.minimum(torch.abs(com_y - ypos.unsqueeze(1)), 
                            torch.abs(1 - (com_y - ypos.unsqueeze(1))))
        xdist = torch.minimum(torch.abs(com_x - xpos.unsqueeze(1)), 
                            torch.abs(1 - (com_x - xpos.unsqueeze(1))))
        udist = (xdist**2 + ydist**2)**0.5
        varin = torch.tensor([(U[udist[:,i] <= 250*1.5, i]**2).mean() for i in range(U.shape[1])])
        varall = (U**2).mean(axis=0).cpu()
        varrat = (varin / varall).mean().item()
        var_all[i] = varrat

        # compute distance distribution of strong pairs
        dbin_strong, dbin_other, cov_bin, drand, crand = strong_pairs(ypos, xpos, 
                                                                            spks=spks,
                                                                            pstrong=pstrong,
                                                                            bin_size=bin_size, 
                                                                            nbins=nbins, 
                                                                            dist_min=dist_min,
                                                                            device=device)
        dbin_strong_all.append(dbin_strong)
        dbin_other_all.append(dbin_other)
        cov_bin_all.append(cov_bin)
        drand_all.append(drand)
        crand_all.append(crand)

        nstrong = int(np.round(pstrong * spks.shape[0]))
        

    return (pcs_all, ypos_all, xpos_all, var_all, 
            dbin_strong_all, dbin_other_all, cov_bin_all, drand_all, crand_all,
            bin_size, pstrong, dist_min)