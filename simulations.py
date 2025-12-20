import torch
from tqdm import trange
from scipy.stats import zscore
from powerlaw import fit_powerlaw_exp, zscore_and_compute_evals, SVCA, SVCA2, compute_evals
from torchaudio.functional import fftconvolve
import numpy as np
from rastermap import Rastermap
from torch.nn import functional as F

def set_seed(seed, device):
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed(seed)


def random_connectivity(nn=10000, nonsym=0, distribution="uniform", 
                        device=torch.device("cuda")):
    """ generate random connectivity matrix 
    
    Args:
        nn: number of neurons
        nonsym: 0 for symmetric matrix, 1 for nonsymmetric matrix, and 0-1 for partially symmetric matrix
        distribution: probability distribution of random matrix
        device: device to generate matrix on

    Returns:
        A: connectivity matrix
    
    """
    if distribution == "uniform":
        A = 2 * torch.rand((nn, nn), device=device) - 1
    elif distribution == "binary":
        A = torch.randn((nn, nn), device=device)
        A[A > 0] = 1
        A[A <= 0] = -1
    elif distribution == "gaussian":
        # gaussian
        A = torch.randn((nn, nn), device=device)
    elif distribution == "trunc_gaussian":
        A = torch.abs(torch.randn((nn, nn), device=device))
        A -= A.mean()
    elif distribution == "exponential":
        A = - torch.log(1 - torch.rand((nn, nn), device=device))
        A -= 1
       
    symmetric = True if nonsym==0 else False
    if symmetric:
        A -= torch.triu(A)
        A = A + A.T
    else:
        if nonsym!=1:
            Aupper = torch.triu(A)
            Alower = torch.tril(A)
            A = Aupper + (1-nonsym) * Aupper.T + nonsym * Alower
            del Aupper, Alower

    A -= torch.diag(torch.diag(A))
    A /= nn**0.5 * A.std()
    A /= 2. if symmetric else 1.

    return A     

def simulate_random(nn=10000, nonsym=0, nd=80, distribution="uniform", T=60000, 
                    tpad=4000, dt=2, tau=20, tbin=23, emax=0.998, device=torch.device("cuda")):
    """ Simulate dynamics of neurons with a random connectivity matrix.
    
    Args:
        nn: number of neurons
        nonsym: 0 for symmetric matrix, 1 for nonsymmetric matrix, and 0-1 for partially symmetric matrix
        nd: number of initial conditions
        distribution: probability distribution of random matrix
        T: number of timesteps
        tpad: number of initial timepoints to exclude (to exclude transient dynamics)
        dt: time step
        tau: neuron time constant
        tbin: time binning
        device: device to run simulation on
    
    Returns:
        A: connectivity matrix
        Xi: simulated dynamics
        evals0: eigenvalues of connectivity matrix

    """
    
    # random connectivity matrix with given distribution
    A = random_connectivity(nn=nn, nonsym=nonsym, distribution=distribution, 
                            device=device)
    if device.type == "cuda":
        torch.cuda.empty_cache()
    
    # compute eigenvalues
    if nonsym == 0:
        evals0, evecs0 = torch.linalg.eigh(A)
    else:
        evals0, evecs0 = torch.linalg.eig(A)
    
    # normalize matrix so eigenvalues are less than 1
    enorm = torch.real(evals0).max() / emax
    evals0 /= enorm
    A /= enorm       

    # simulate dynamics
    X = torch.randn((nn, nd), device=device) 
    Xi = torch.zeros((nn, nd, (T-tpad)//tbin), device=device)
    for t in trange(T):
        eps = torch.randn((nn, nd), device=device)
        X += dt / tau * (-X + A @ X + eps)
        if t >= tpad and (t-tpad)//tbin < Xi.shape[-1]:
            Xi[:, :, (t-tpad)//tbin] += X
    Xi /= tbin
    Xi = Xi.reshape(nn, -1)

    return A, Xi, evals0

def compute_evals_tbin(Xi, tbins):
    """ compute eigenvalues for different time binning 
    
    Args:
        Xi: torch tensor of neurons

    Returns:
        evals_all: eigenvalues for different time binning
    
    """
    nn, nt = Xi.shape
    evals_all = np.nan * np.zeros((len(tbins), nn))
    for ti, tbin0 in enumerate(tbins):
        if tbin0 > 1:
            Xbin = Xi[:,:(nt//tbin0)*tbin0].reshape(nn, -1, tbin0).mean(axis=-1).clone()
        else:
            Xbin = Xi
        evals, evecs = zscore_and_compute_evals(Xbin)
        ntmax = Xbin.shape[1]
        evals_all[ti, :ntmax] = evals[:ntmax]
    return evals_all

def tbin_analysis(n_sim = 10, nn = 10000, device = torch.device("cuda")):
    """ simulate neurons with random uniform connectivity matrix and compute eigenvalues
    
        many sims (200 per connectivity matrix) 

    Args:
        n_sim: number of simulations
        nn: number of neurons
        device: torch device

    Returns:
        A tuple containing evals_all, nonsyms, Xsym_ex, Xnonsym_ex, evals_bin_all, tbins
    
    """
    set_seed(0, device)

    # nonsymmetry levels of random connectivity matrix
    nonsyms = [0., 1]
    tbins = np.unique(np.round(np.exp(np.linspace(np.log(1), np.log(2000), 20)))).astype(int)
    evals_bin_all = np.zeros((n_sim, len(nonsyms), len(tbins), nn))
    for i in range(n_sim):
        for ni, nonsym in enumerate(nonsyms):
            A, Xi, evals0 = simulate_random(nn=nn, nonsym=nonsym, nd=40, tbin=3)

            evals_bin_all[i, ni] = compute_evals_tbin(Xi, tbins)
            
            del Xi
    
    return nonsyms, evals_bin_all, tbins * 0.006


def random_probs_sim(n_sim = 10, nn = 10000, device = torch.device("cuda")):
    """ simulate neurons with symmetric random connectivity matrix with different probability distributions 
    
        saves output in evals_probs.npy

    Args:
        n_sim: number of simulations
        nn: number of neurons
        device: torch device

    Returns:
        A tuple containing evals_all, Aexs, distributions
    
    """
    set_seed(0, device)
    nonsym = 0 # symmetric
    evals_all = np.zeros((n_sim, 4, nn), "float32") * np.nan
    Aexs = np.zeros((4, nn, nn), "float32")
    distributions = ["binary", "gaussian", "trunc_gaussian", "exponential"]
    for i in range(n_sim):
        for si in range(4):
            A, Xi, evals0 = simulate_random(nn=nn, nonsym=nonsym, nd=80,
                                            distribution=distributions[si], 
                                            device=device)
            
            evals, evecs = zscore_and_compute_evals(Xi)
            evals_all[i, si] = evals

            alpha, yp = fit_powerlaw_exp(evals, np.arange(10, 500))
            print(f"{distributions[si]}, {alpha:.3f}")
            
            if i==0:
                Aexs[si] = A.cpu().numpy()
    
    return evals_all, Aexs, distributions

def ca_imaging_noise_dcnv(Xn, shot_noise=0.2, downsample=1):
    """ updates Xn """
    device = Xn.device
    lam = 22/4
    expfilt = torch.exp(- torch.arange(0, 50) / lam)
    expfilt /= expfilt.mean()

    Xn = Xn.cpu()
    Xn = fftconvolve(Xn, expfilt.unsqueeze(0), mode="same")
    Xn = Xn.to(device)
    Xn *= 8 
    Xn += 400 

    if downsample > 1:
        Xn = Xn[:, ::downsample]

    exprand = torch.empty(Xn.shape[0], device=device)
    exprand.exponential_(lambd=1/shot_noise)
    exprand += 0.001
    Xn = torch.poisson(Xn * exprand.unsqueeze(1))
    
    # deconvolve
    try:
        import udcnv
        from suite2p.extraction import dcnv 
        dF = dcnv.baseline_maximin(Xn.cpu().numpy(), win_baseline=60, sig_baseline=10, fs=22/downsample, batch_size=500)
        spks0 = udcnv.apply(dF, 22/downsample, '/media/carsen/disk1/grive/strongpairs/notebooks/sim_right_flex.th', batch_size=64)
        Xn = torch.from_numpy(spks0).to(device)
    except Exception as e:
        print(e)
        print('!!!!!!!!!!!!!!!! ERROR !!!!!!!!!!!!!!!!!!!!!')
        print('ERROR: new deconvolution not available - simulations will not be deconvolved')
    
    return Xn


def random_uniform_sim_sizes(n_sim=10, nn=10000, nonsyms=[0, 1./3, 2./3, 1],
                             device=torch.device("cuda")):

    set_seed(0, device)

    noise_levels = [('poisson', 0.7), ('poisson', 0.5),  ('poisson', 0.3), 
                     ('poisson', 0.5), ('poisson', 0.5), ('poisson', 0.5)]
    shot_noise_levels = [None, None, None, 0.5, 0.2, 0.08]
    # SNR ~= 0.4, 0.56, 0.71
    
    nneurons = np.unique(np.round(np.exp(np.linspace(np.log(250), np.log(10000), 10)))).astype(int)
    ntimes = np.unique(np.round(np.exp(np.linspace(np.log(23 * 60 * 15), np.log(194720), 10)))).astype(int)

    # nonsymmetry levels of random connectivity matrix
    evals_gt_all = np.zeros((n_sim, len(nonsyms), nn)) * np.nan
    evals_direct_all = np.zeros((n_sim, len(nonsyms), len(noise_levels), len(nneurons), len(ntimes), nn)) * np.nan
    evals_svca_all = np.zeros((n_sim, len(nonsyms), len(noise_levels), len(nneurons), len(ntimes), nn)) * np.nan
    evals_svca2_all = np.zeros((n_sim, len(nonsyms), len(noise_levels), len(nneurons), len(ntimes), nn)) * np.nan
    alphas = np.zeros((3, n_sim, len(nonsyms), len(noise_levels), len(nneurons), len(ntimes))) * np.nan
    for i in range(n_sim):
        for ni, nonsym in enumerate(nonsyms):
            A, Xi, evals0 = simulate_random(nn=nn, nonsym=nonsym, nd=80)
            Xi_mean = Xi.mean(axis=1, keepdim=True)

            # zscore for evals
            Xi -= Xi_mean
            Xi /= Xi.std(axis=1, keepdim=True)
            
            # compute eigenvalues of covariance matrix
            evals_gt, evecs = compute_evals(Xi)
            evals_gt_all[i, ni] = evals_gt

            Xi += Xi_mean

            # add noise and smooth in time
            for nl, (noise_level, shot_noise) in enumerate(zip(noise_levels, shot_noise_levels)):
                Xn = Xi.clone()

                Xn = torch.poisson(F.relu(Xn) * noise_level[1]) # / noise_level[1]
                
                if shot_noise is not None:
                    Xn = ca_imaging_noise_dcnv(Xn, shot_noise=shot_noise)
                
                Xn -= Xn.mean(axis=1, keepdim=True)
                Xn /= Xn.std(axis=1, keepdim=True)

                for jj, ntime in enumerate(ntimes[::-1]):
                    for ii, nneur in enumerate(nneurons):
                        if Xn.shape[0] < nneur:
                            continue
                        X0 = Xn[:nneur][:, :ntime]
                        
                        evals, evecs = compute_evals(X0)
                        evals_direct_all[i, ni, nl, ii, jj, :len(evals)] = evals
                        
                        # compute eigenvalues with SVCA and SVCA2
                        evals_svca2 = SVCA2(X0)
                        evals_svca2_all[i, ni, nl, ii, jj, :len(evals_svca2)] = evals_svca2
                        evals_svca = SVCA(X0)[0]
                        evals_svca_all[i, ni, nl, ii, jj, :len(evals_svca)] = evals_svca
                        
                        # power-law decay exponent
                        yrange = np.arange(10, min(250, min(len(evals_svca), len(evals_svca2), len(evals)) - 50))
                        alpha_gt = fit_powerlaw_exp(evals_gt, yrange)[0]
                        alpha = fit_powerlaw_exp(evals, yrange)[0]
                        alpha_svca = fit_powerlaw_exp(evals_svca, yrange)[0]
                        alpha_svca2 = fit_powerlaw_exp(evals_svca2, yrange)[0]
                        
                        alphas[0, i, ni, nl, ii, jj] = alpha
                        alphas[1, i, ni, nl, ii, jj] = alpha_svca 
                        alphas[2, i, ni, nl, ii, jj] = alpha_svca2

                        if ntime==ntimes[-1] or nneur==nneurons[-1]:
                            print(f"{i} {nl} {X0.shape[0]} {X0.shape[1]} {nonsym:.2f}; alpha_gt: {alpha_gt:.2f}, alpha: {alpha:.2f}, alpha_svca: {alpha_svca:.2f}, alpha_svca2: {alpha_svca2:.2f}")    
                    torch.cuda.empty_cache()
            del Xi, Xn
            torch.cuda.empty_cache()

    return alphas, evals_gt_all, evals_direct_all, evals_svca_all, evals_svca2_all, nonsyms, nneurons, ntimes[::-1], noise_levels


def sims_fig2(n_sim=10):
    
    # one example sim with enorm 0.975, multiple sims for top PC
    evals_all = np.zeros((2, n_sim, 10000)) * np.nan
    Xemb_ex = []
    for j, enorm in enumerate([0.998, 0.975]):
        for i in range(n_sim):
            
            A, Xi, evals0 = simulate_random(nn=10000, nonsym=0, nd=40, T=100000, emax=enorm, tbin=23, tau=20, dt=2)
            Xi /= Xi.std(axis=1, keepdim=True)
            Xi = torch.poisson(F.relu(Xi) * 0.5)
            print(Xi.shape)
            Xi = ca_imaging_noise_dcnv(Xi, shot_noise=0.2)
            Xi -= Xi.mean(axis=1, keepdim=True)
            Xi /= Xi.std(axis=1, keepdim=True)
          
            evals = SVCA2(Xi)
            evals_all[j, i, :len(evals)] = evals

            if i==0:
                spks = Xi.cpu().numpy()
                model = Rastermap(n_clusters=100, n_PCs=128, mean_time=False,
                            normalize=False, bin_size=20, time_bin=7).fit(spks)
                nn, nt = spks.shape
                # bin_size = 40
                # Xemb = spks[model.isort[:(nn//bin_size)*bin_size]].reshape(nn//bin_size, bin_size, -1).mean(axis=1)
                # time_bin = 7
                # Xemb = Xemb[:,:(nt//time_bin)*time_bin].reshape(-1, nt//time_bin, time_bin).mean(axis=-1)
                # Xemb = zscore(Xemb, axis=1)
                Xemb = model.X_embedding

                Xemb_ex.append(Xemb)

            del Xi
            torch.cuda.empty_cache()

    return evals_all, Xemb_ex

def enorm_sweep(n_sim=10):
    enorms = np.arange(0.975, 0.9985, 0.001)
    print(enorms)
    nnorm = len(enorms)
    evals_all = np.zeros((nnorm, n_sim, 10000))
    alphas_all = np.zeros((nnorm, n_sim))
    for i in trange(n_sim):
        A = random_connectivity(nn=10000, nonsym=0)
        evals0, evecs0 = torch.linalg.eigh(A)
        for j, emax in enumerate(enorms):
            enorm = torch.real(evals0).max() / emax
            evals0 /= enorm
            # A /= enorm
            evals = evals0.cpu().numpy()[::-1]
            # eigenvalues of covariance matrix are 0.5 / (1 - eigenvalues of A)
            evals = 0.5 / (1 - evals)
            alpha, yp = fit_powerlaw_exp(evals, np.arange(10, 500))
            if i==0:
                print(f'max eval: {emax:.3f}, alpha: {alpha:.3f}')
            evals_all[j,i] = evals
            alphas_all[j,i] = alpha
    return evals_all, alphas_all, enorms
        
