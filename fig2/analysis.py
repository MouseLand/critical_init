import numpy as np 
import torch
from natsort import natsorted
from scipy.sparse import csr_matrix
from powerlaw import fit_powerlaw_exp, SVCA, SVCA2
from lyapun import dmd
from rastermap import Rastermap


def neural_powerlaws(root, dbs_2P, dbs_ephys, device=torch.device("cuda")):
    """ compute powerlaws for neural data 

        saves data to "evals_neural_all.npy"

    Args:
        dbs_2P: list of dictionaries with "mouse_name" and "area" fields
        dbs_ephys: list of dictionaries with "mouse_name" field
        device: torch device to use
    
    Returns:
        A tuple of (evals_all, evals_svca2_all, evals_svca_all, evals_shuff_all, areas_all, Xexs, isorts, snr_rat)
    """
    evals_all = []
    evals_svca2_all = []
    evals_all_neurons = []
    evals_svca_all_neurons = []
    evals_svca2_all_neurons = []
    evals_all_times = []
    evals_svca_all_times = []
    evals_svca2_all_times = []
    evals_svca_all = []
    evals_shuff_all = []
    areas_all = []
    isorts = []
    Xexs = []
    dbex = [8, 20]
    snr_rat = []
    shapes = []

    bsize = 4000
    ikeeps = np.arange(50, bsize+1, 200)
    ikeeps[-1] = 4000
    ineurs = np.arange(0.01, 1.05, 0.05)
    ineurs[-1] = 1.0
    # 2P data
    for i, db in enumerate(dbs_2P):
        mouse_name = db["mouse_name"]
        area = db["area"]
        date = db["date"]
        dat = np.load(root / f"F_{mouse_name}_{date}.npz")
        spks = dat["sp"]
        if "tstart" in dat:
            spks = spks[:, dat["tstart"]:]
        print(f">>> {mouse_name}, n_neurons(SNR>0.3) = {spks.shape[0]}, nt = {spks.shape[1]} ({spks.shape[1]/22/60:.1f} minutes)")
        ypos = dat["ypos"]
        xpos = dat["xpos"]

        # normalize spks 
        spks -= spks.mean(axis=1, keepdims=True)
        spks /= spks.std(axis=1, keepdims=True)

        shapes.append(spks.shape)

        np.random.seed(0)
        iperm = np.random.permutation(spks.shape[0])
        spks = spks[iperm]
        ypos = ypos[iperm]
        xpos = xpos[iperm]
        
        # powerlaw
        spks_gpu = torch.from_numpy(spks.copy()).to(device)
        evals, evecs = torch.linalg.eigh((spks_gpu @ spks_gpu.T) / spks_gpu.shape[1])
        evals = evals.cpu().numpy()[::-1]
        alpha = fit_powerlaw_exp(evals, np.arange(10, 500))[0]
        evals_all.append(evals)

        nn = spks.shape[0]
        ss = SVCA2(spks_gpu, xpos=xpos, ypos=ypos)
        alpha_svca2 = fit_powerlaw_exp(ss, np.arange(10, 500))[0]
        evals_svca2_all.append(ss)
        areas_all.append(area)

        ss = SVCA(spks_gpu, xpos=xpos, ypos=ypos)[0]
        alpha_svca = fit_powerlaw_exp(ss, np.arange(10, 500))[0]
        evals_svca_all.append(ss)

        nneurons = (spks.shape[0] * ineurs).astype('int')
        evals_svca2_all_neurons.append([])
        evals_svca_all_neurons.append([])
        evals_all_neurons.append([])
        alpha_svca2_ni = []
        
        for ni in nneurons:
            ss = SVCA2(spks_gpu[:ni], xpos=xpos[:ni], ypos=ypos[:ni])
            if len(ss)//2 > 10:
                alp = fit_powerlaw_exp(ss, np.arange(10, min(500, len(ss)//2)))[0]
            else:
                alp = 0
            alpha_svca2_ni.append(alp)
            evals_svca2_all_neurons[-1].append(ss)
            ss = SVCA(spks_gpu[:ni], xpos=xpos[:ni], ypos=ypos[:ni])[0]
            evals_svca_all_neurons[-1].append(ss)
            evals, evecs = torch.linalg.eigh((spks_gpu[:ni] @ spks_gpu[:ni].T) / spks_gpu.shape[1])
            evals = evals.cpu().numpy()[::-1]
            evals_all_neurons[-1].append(evals)
        print(np.array(alpha_svca2_ni))
        
        # shuffle in time
        torch.random.manual_seed(1)
        #spks_gpu = torch.from_numpy(spks.copy()).to(device)
        for j in range(500):
            irand = torch.randperm(nn)[:nn//2]
            trand = np.random.randint(0, spks.shape[1])
            spks_gpu[irand] = torch.roll(spks_gpu[irand], trand, dims=1)
        ss = SVCA2(spks_gpu, xpos=xpos, ypos=ypos)
        
        alpha_shuff = fit_powerlaw_exp(ss, np.arange(10, 500))[0]
        evals_shuff_all.append(ss)

        alpha_svca2_ni = []
        evals_svca2_all_times.append([])
        evals_svca_all_times.append([])
        evals_all_times.append([])
        for ki, ikeep in enumerate(ikeeps):
            inds = (np.arange(0, spks.shape[1]) % bsize) < ikeep
            spks0 = spks[:, inds].copy()
            spks0 -= spks0.mean(axis=1, keepdims=True)
            spks0 /= spks0.std(axis=1, keepdims=True) + 1e-3
            spks_gpu = torch.from_numpy(spks0).to(device)
            ss = SVCA2(spks_gpu, xpos=xpos, ypos=ypos)
            alpha_svca2_ni.append(fit_powerlaw_exp(ss, np.arange(10, min(500, len(ss)//2)))[0])
            evals_svca2_all_times[-1].append(ss)
            ss = SVCA(spks_gpu, xpos=xpos, ypos=ypos)[0]
            evals_svca_all_times[-1].append(ss)
            evals, evecs = torch.linalg.eigh((spks_gpu @ spks_gpu.T) / spks_gpu.shape[1])
            evals = evals.cpu().numpy()[::-1]
            evals_all_times[-1].append(evals)
        print(np.array(alpha_svca2_ni))
        
        print(f"\talpha: direct={alpha:.2f}, SVCA2={alpha_svca2:.2f}, SVCA={alpha_svca:.2f}, SVCA2_shuff={alpha_shuff:.2f}")
        
        del spks_gpu
        torch.cuda.empty_cache()

        # rastermap
        if i in dbex:
            print("running rastermap")
            model = Rastermap(n_clusters=100, n_PCs=128, normalize=False, bin_size=20, time_bin=7, mean_time=False).fit(spks)
            Xexs.append(model.X_embedding)
            isorts.append(model.isort)       

    
    # ephys data
    tbin = 1. / 22 # 22 Hz binning
    for i, db in enumerate(dbs_ephys):
        mouse_name = db["mouse_name"]
        dat = np.load(root / f"{mouse_name}_spks_face.npz", allow_pickle=True)
        areas_all.append("ephys")
        area = dat["areas"]
        clu = dat["clu"]
        st = dat["st"]
        st -= st.min()
        spks = csr_matrix((np.ones(clu.size, "uint8"), (clu, np.round(st / tbin).astype("uint32"))), 
                                    shape=(clu.max()+1, int(np.round(st.max() / tbin)) + 1))
        spks = spks.toarray().astype("float32")
        
        # keep neurons with firing rate > 0.01 Hz
        igood = spks.mean(axis=1) > 0.01 * tbin
        spks = spks[igood]
        area = area[igood]
        ypos = dat["ypos"][igood]
        xpos = dat["iprobe"][igood].astype("float32") * 0
        print(f">>> {mouse_name}, n_neurons(fr>0.01Hz) = {spks.shape[0]}, nt = {spks.shape[1]} ({spks.shape[1]/22/60:.1f} minutes)")

        spks -= spks.mean(axis=1, keepdims=True)
        spks /= spks.std(axis=1, keepdims=True)
        print(spks.shape)

        shapes.append(spks.shape)

        np.random.seed(0)
        iperm = np.random.permutation(spks.shape[0])
        spks = spks[iperm]
        ypos = ypos[iperm]
        xpos = xpos[iperm]
            
        # powerlaw
        spks_gpu = torch.from_numpy(spks.copy()).to(device)
        evals, evecs = torch.linalg.eigh((spks_gpu @ spks_gpu.T) / spks_gpu.shape[1])
        evals = evals.cpu().numpy()[::-1]
        alpha = fit_powerlaw_exp(evals, np.arange(10, 500))[0]
        evals_all.append(evals)

        ni = spks.shape[0]
        ss = SVCA2(spks_gpu, xpos=xpos, ypos=ypos, spacing=40)
        
        alpha_svca2 = fit_powerlaw_exp(ss, np.arange(10, 500))[0]
        evals_svca2_all.append(ss)

        ss = SVCA(spks_gpu, xpos=xpos, ypos=ypos, spacing=40)[0]
        alpha_svca = fit_powerlaw_exp(ss, np.arange(10, 500))[0]
        evals_svca_all.append(ss)

        nneurons = (spks.shape[0] * ineurs).astype('int')
        evals_svca2_all_neurons.append([])
        evals_svca_all_neurons.append([])
        evals_all_neurons.append([])
        alpha_svca2_ni = []
        for ni in nneurons:
            ss = SVCA2(spks_gpu[:ni], xpos=xpos[:ni], ypos=ypos[:ni], spacing=40)
            if len(ss) > 20:
                alp = fit_powerlaw_exp(ss, np.arange(10, min(500, len(ss)//2)))[0]
            else:
                alp = 0
            alpha_svca2_ni.append(alp)
            evals_svca2_all_neurons[-1].append(ss)
            ss = SVCA(spks_gpu[:ni], xpos=xpos[:ni], ypos=ypos[:ni], spacing=40)[0]
            evals_svca_all_neurons[-1].append(ss)
            evals, evecs = torch.linalg.eigh((spks_gpu[:ni] @ spks_gpu[:ni].T) / spks_gpu.shape[1])
            evals = evals.cpu().numpy()[::-1]
            evals_all_neurons[-1].append(evals)
        print(np.array(alpha_svca2_ni))

        # shuffle in time
        torch.random.manual_seed(0)
        for j in range(500):
            irand = torch.randperm(spks.shape[0])[:spks.shape[0]//2]
            trand = np.random.randint(0, spks.shape[1])
            spks_gpu[irand] = torch.roll(spks_gpu[irand], trand, dims=1)
        ss = SVCA2(spks_gpu, xpos=xpos, ypos=ypos, spacing=40)

        alpha_shuff = fit_powerlaw_exp(ss, np.arange(10, 500))[0]
        evals_shuff_all.append(ss)

        alpha_svca2_ni = []
        evals_svca2_all_times.append([])
        evals_svca_all_times.append([])
        evals_all_times.append([])
        for ki, ikeep in enumerate(ikeeps):
            inds = (np.arange(0, spks.shape[1]) % bsize) < ikeep
            spks0 = spks[:, inds].copy()
            spks0 -= spks0.mean(axis=1, keepdims=True)
            spks0 /= spks0.std(axis=1, keepdims=True) + 1e-3
            spks_gpu = torch.from_numpy(spks0).to(device)
            ss = SVCA2(spks_gpu, xpos=xpos, ypos=ypos, spacing=40)
            alpha_svca2_ni.append(fit_powerlaw_exp(ss, np.arange(10, min(500, len(ss)-50)))[0])
            evals_svca2_all_times[-1].append(ss)
            ss = SVCA(spks_gpu, xpos=xpos, ypos=ypos, spacing=40)[0]
            evals_svca_all_times[-1].append(ss)
            evals, evecs = torch.linalg.eigh((spks_gpu @ spks_gpu.T) / spks_gpu.shape[1])
            evals = evals.cpu().numpy()[::-1]
            evals_all_times[-1].append(evals)
        print(np.array(alpha_svca2_ni))

        del spks_gpu
        torch.cuda.empty_cache()

        print(f"\talpha: direct={alpha:.2f}, SVCA2={alpha_svca2:.2f}, SVCA={alpha_svca:.2f}, SVCA2_shuff={alpha_shuff:.2f}")

        # rastermap
        if i==1:
            print("running rastermap")
            model = Rastermap(n_clusters=100, n_PCs=128, bin_size=10, time_bin=7, mean_time=False).fit(spks)
            Xexs.append(model.X_embedding)
            isorts.append(model.isort)

    return evals_all, evals_all_neurons, evals_all_times, evals_svca2_all, evals_svca2_all_neurons, evals_svca2_all_times, evals_svca_all, evals_svca_all_neurons, evals_svca_all_times, evals_shuff_all, areas_all, Xexs, isorts, snr_rat, shapes

def ephys_areas(root, dbs_ephys, device=torch.device('cuda')):
    area_groups = {'striatum': ["CP", "LS"], 'hippocampal formation': ["HPF"], 
                'subcortical areas': ["MB", "SC","TH"], 
                'visual cortex': ["V1", "V2"], 
                'sensorimotor cortices': ["FrMoCtx", "SSCtx", "SomMoCtx"]}
    evals_areas = {aname: [[], [], []] for aname in area_groups.keys()}
    nneurons = {aname: np.zeros(3) for aname in area_groups.keys()}
    
    for iexp, db in enumerate(dbs_ephys):
        mouse_name = db["mouse_name"]
        tbin = 1./22
        dat = np.load(root / f"{mouse_name}_spks_face.npz", allow_pickle=True)
        area = dat["areas"]

        clu = dat["clu"]
        st = dat["st"]
        st -= st.min()
        spks = csr_matrix((np.ones(clu.size, "uint8"), (clu, np.round(st / tbin).astype("uint32"))), 
                                    shape=(clu.max()+1, int(np.round(st.max() / tbin)) + 1))
        spks = spks.toarray().astype("float32")
        
        # keep neurons with firing rate > 0.01 Hz
        igood = spks.mean(axis=1) > 0.01 * tbin
        spks = spks[igood]
        area = area[igood]
        ypos = dat["ypos"][igood]
        xpos = dat["iprobe"][igood] * 0
        print(f">>> {mouse_name}, n_neurons(fr>0.01Hz) = {spks.shape[0]}, nt = {spks.shape[1]} ({spks.shape[1]/22/60:.1f} minutes)")

        spks -= spks.mean(axis=1, keepdims=True)
        spks /= spks.std(axis=1, keepdims=True)
        print(spks.shape)

        for area_group in area_groups.keys():
            iarea = np.isin(area, area_groups[area_group])
            print(area_group, iarea.sum())
            nneurons[area_group][iexp] = iarea.sum()
            if iarea.sum() < 50:
                print("\t skip, n_neurons < 50")
                continue
            spks_area = spks[iarea]
            ypos_area = ypos[iarea]
            xpos_area = xpos[iarea]
            # powerlaw
            
            spks_gpu = torch.from_numpy(spks_area.copy()).to(device)
            ss = SVCA2(spks_gpu, xpos=xpos_area, ypos=ypos_area, spacing=40)
            ymax = min(len(ss)//2, 500)
            alpha_svca2, ypred = fit_powerlaw_exp(ss, np.arange(10, ymax))
            evals_areas[area_group][iexp] = ss

            del spks_gpu
            torch.cuda.empty_cache()

            print(f"\talpha: SVCA2={alpha_svca2:.2f}")

    return evals_areas, area_groups, nneurons

def ephys_tbins(root, dbs_ephys, device=torch.device('cuda')):
    evals_all = []
    #tbins = 0.004 * np.array([1., 4., 9., 12., 24., 48., 96., 192.])#np.unique(np.round(np.exp(np.linspace(np.log(1), np.log(1000), 10)))).astype(int)
    tbins = 0.005 * np.unique(np.round(np.exp(np.linspace(np.log(1), np.log(20), 10)))).astype(int)
        
    for db in dbs_ephys:
        evals_all.append([])
        # ephys data
        mouse_name = db["mouse_name"]
        dat = np.load(root / f"{mouse_name}_spks_face2.npz", allow_pickle=True)
        area = dat["areas"]
        
        # keep neurons with firing rate > 0.01 Hz
        tbin = 1./22
        clu = dat["clu"]
        st = dat["st"]
        st -= st.min()
        spks = csr_matrix((np.ones(clu.size, "uint8"), (clu, np.round(st / tbin).astype("uint32"))), 
                                    shape=(clu.max()+1, int(np.round(st.max() / tbin)) + 1))
        spks = spks.toarray().astype("float32")
        igood = spks.mean(axis=1) > 0.01 * (1. / 22) # 22hz binnning

        spks = spks[igood]
        area = area[igood]
        ypos = dat["ypos"][igood]
        xpos = dat["iprobe"][igood] * 0
        print(f">>> {mouse_name}, n_neurons(fr>0.01Hz) = {spks.shape[0]}, nt = {spks.shape[1]} ({spks.shape[1]/22/60:.1f} minutes)")

        
        for tbin0 in tbins:
            spks_b = csr_matrix((np.ones(clu.size, "uint8"), (clu, np.round(st / tbin0).astype("uint32"))), 
                                    shape=(clu.max()+1, int(np.round(st.max() / tbin0)) + 1))
            spks_b = spks_b.toarray().astype("float32")
            spks_b = spks_b[igood]
            spks_b -= spks_b.mean(axis=1, keepdims=True)
            spks_b /= spks_b.std(axis=1, keepdims=True)
            
            # powerlaw
            spks_gpu = torch.from_numpy(spks_b).to(device)
            ss = SVCA2(spks_gpu, xpos=xpos, ypos=ypos, spacing=40)
            alpha_svca2, ypred = fit_powerlaw_exp(ss, np.arange(10, 500))
            del spks_gpu
            torch.cuda.empty_cache()
            evals_all[-1].append(ss)
            
            print(f"tbin={tbin0:.3f}, \talpha_SVCA2={alpha_svca2:.2f}")#, SVCA={alpha_svca:.2f}, SVCA2_shuff={alpha_shuff:.2f}")
        
    return evals_all, tbins

def running(root, dbs_2P, device=torch.device('cuda')):
    alphas = []
    ms = []
    meds = []
    irun = np.array([db.get('hasrunning', False) for db in dbs_2P])
    nrun = irun.sum()
    evals_run_all = np.zeros((2, nrun, 1000)) * np.nan
    dmd_evals_all = np.zeros((2, nrun, 1000), 'complex64') * np.nan
    iexp = 0
    for ii in np.nonzero(irun)[0]:
        db = dbs_2P[ii]
        mouse_name = db["mouse_name"]
        #print(mouse_name)
        area = db["area"]
        date = db["date"]
        dat = np.load(root / f"F_{mouse_name}_{date}.npz")
        print(mouse_name, dat['isrunning'].mean(), len(dat['isrunning']))
        
        spks = dat["sp"]
        ypos = dat["ypos"]
        xpos = dat["xpos"]

        spks -= spks.mean(axis=1, keepdims=True)
        spks /= spks.std(axis=1, keepdims=True)

        inds = [np.nonzero(dat['isrunning'])[0], np.nonzero(~dat['isrunning'])[0]]
        l0 = min(len(inds[0]), len(inds[1]))
        inds = [ind[:l0] for ind in inds]
        
        from powerlaw import SVCA2, fit_powerlaw_exp
        spks_gpu = torch.from_numpy(spks.copy()).to(device)

        alphas = []
        for i, ind in enumerate(inds):
            #print(ind.sum())
            ss = SVCA2(spks_gpu[:,ind], xpos=xpos, ypos=ypos)
            ymax = min(len(ss), 500)
            alpha_svca2, ypred = fit_powerlaw_exp(ss, np.arange(10, ymax))
            evals_run_all[i, iexp, :len(ss)] = ss[:1000]
            alphas.append(alpha_svca2)
        print(f'alpha_run = {alphas[0]:.3f}, alpha_notrun = {alphas[1]:.3f}')
        

        # less than 30 min for dmd
        ibad = l0 < (30*60*22)
        if ibad:
            iexp += 1
            continue
        
        for i, ind in enumerate(inds):
            At, e, v = dmd(spks, delta=5, lam=0.1, nt=5000, inds=ind)
            dmd_evals_all[i, iexp, :len(e)] = e[:1000]
            ix = np.abs(e)>.25
            iang = np.angle(e[ix]) / (2*np.pi)
            iabs = -np.log10(np.abs(e[ix]))    
            irot = iang/iabs
            ixx = e[ix].imag>=0
            mu = irot[ixx].mean()
            sd = irot[ixx].std()
            m = np.percentile(irot[ixx], [5, 25, 75, 95])
            med = np.median(irot[ixx])
            print(['dmd_run', 'dmd_notrun'][i], m, med)
        iexp += 1
            

    areas = [dbs_2P[ii]['area'] for ii in np.nonzero(irun)[0]]

    return evals_run_all, dmd_evals_all, areas

def gcamp_subsample(root, dbs_2P, device=torch.device('cuda')):

    # > using gcamp6 data used in Figure 4
    files = natsorted(list(root.glob('spont_*_spks.npz')))
    evals_all = np.nan * np.zeros((len(files), 1000))
    alphas_all = np.zeros(len(files)) * np.nan
    for iexp, f in enumerate(files):
        dat = np.load(f)
        spks = dat['spks'].astype('float32')
        igood = spks.std(axis=1) > 1e-2 
        ypos = dat['ypos'][igood]
        xpos = dat['xpos'][igood]
        spks = spks[igood]

        spks -= spks.mean(axis=1, keepdims=True)
        spks /= spks.std(axis=1, keepdims=True)

        if spks.shape[0] > 50000:
            irand = np.random.permutation(spks.shape[0])[:50000]
            spks = spks[irand]
            xpos = xpos[irand]
            ypos = ypos[irand]
        spks_gpu = torch.from_numpy(spks).to(device)
        
        ymax = 500
        ss = SVCA(spks_gpu, xpos=xpos, ypos=ypos)[0]
        alpha_svca, ypred = fit_powerlaw_exp(ss, np.arange(10, ymax))
        print(alpha_svca)
        evals_all[iexp, :len(ss)] = ss[:1000]
        alphas_all[iexp] = alpha_svca

    np.save('../results/gcamp6_svca_janelia.npy', {'evals_all': evals_all, 'alphas_all': alphas_all})

    # download spont_M* files from 
    # https://figshare.com/articles/dataset/Recordings_of_ten_thousand_neurons_in_visual_cortex_during_spontaneous_behaviors/6163622
    files = list(root.glob('spont_M*.mat'))
    from scipy.io import loadmat 

    evals_all = np.nan * np.zeros((len(files), 1000))
    alphas_all = np.zeros(len(files)) * np.nan
    for iexp, f in enumerate(files):
        dat = loadmat(f, squeeze_me=True)
        spks = dat['Fsp']
        igood = spks.std(axis=1) > 1e-3
        spks = spks[igood]

        ypos = dat['med'][igood,0]
        xpos = dat['med'][igood,1]

        npl = len(np.unique(dat['med'][:,-1])) + 1
        spks_gpu = torch.from_numpy(spks).to(device)
        fs = 30 / npl
        ymax = 500
        print(fs, spks.shape)
        ss = SVCA(spks_gpu, xpos=xpos, ypos=ypos)[0]
        alpha_svca, ypred = fit_powerlaw_exp(ss, np.arange(10, ymax))
        print(alpha_svca)
        evals_all[iexp, :len(ss)] = ss[:1000]
        alphas_all[iexp] = alpha_svca

    np.save('../results/gcamp6_svca_london.npy', {'evals_all': evals_all, 'alphas_all': alphas_all})


    # > subsampled then deconvolved traces from gcamp8 @ 22hz
    evals_all = np.nan * np.zeros((len(dbs_2P), 1000))
    alphas_all = np.zeros(len(dbs_2P)) * np.nan
    for i, db in enumerate(dbs_2P):
        mouse_name = db["mouse_name"]
        area = db["area"]
        date = db["date"]
        dat = np.load(root / f"Fsub2/Fsub_{mouse_name}_{date}.npz")
        spks = dat["sp"]
        fs = 3
        print(f">>> {mouse_name}, n_neurons = {spks.shape[0]}, nt = {spks.shape[1]} ({spks.shape[1]/fs/60:.1f} minutes)")
        ypos = dat["ypos"]
        xpos = dat["xpos"]

        spks -= spks.mean(axis=1, keepdims=True)
        spks /= spks.std(axis=1, keepdims=True)

        spks_gpu = torch.from_numpy(spks).to(device)
        
        ymax = 500
        ss = SVCA(spks_gpu, xpos=xpos, ypos=ypos, fs=fs)[0]
        alpha_svca, ypred = fit_powerlaw_exp(ss, np.arange(10, ymax))
        print(alpha_svca)
        evals_all[i, :len(ss)] = ss[:1000]
        alphas_all[i] = alpha_svca

    np.save('../results/gcamp8_svca_subsample.npy', {'evals_all': evals_all, 'alphas_all': alphas_all})
