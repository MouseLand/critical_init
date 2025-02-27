import numpy as np 
import torch
from powerlaw import fit_powerlaw_exp, SVCA, SVCA2
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
    evals_svca_all = []
    evals_shuff_all = []
    areas_all = []
    isorts = []
    Xexs = []
    dbex = [1, 7]
    snr_rat = []

    # 2P data
    for i, db in enumerate(dbs_2P):
        mouse_name = db["mouse_name"]
        area = db["area"]
        dat = np.load(root / f"F_{mouse_name}.npz")
        snr = dat["snr"]
        snr_rat.append((snr>0.3).sum()/snr.shape[0])
        spks = dat["sp"]
        print(f">>> {mouse_name}, n_neurons(SNR>0.3) = {(snr>0.3).sum()}, nt = {spks.shape[1]} ({spks.shape[1]/22/60:.1f} minutes)")
        ypos = dat["ypos"]
        xpos = dat["xpos"]

        # normalize spks and remove low snr neurons
        spks = spks[dat["snr"] > 0.3]
        ypos = ypos[dat["snr"] > 0.3]
        xpos = xpos[dat["snr"] > 0.3]
        spks -= spks.mean(axis=1, keepdims=True)
        spks /= spks.std(axis=1, keepdims=True)
        
        # powerlaw
        spks_gpu = torch.from_numpy(spks.copy()).to(device)
        evals, evecs = torch.linalg.eigh((spks_gpu @ spks_gpu.T) / spks_gpu.shape[1])
        evals = evals.cpu().numpy()[::-1]
        alpha = fit_powerlaw_exp(evals, np.arange(10, 500))[0]
        evals_all.append(evals)

        ss = SVCA2(spks_gpu, xpos=xpos, ypos=ypos)
        alpha_svca2 = fit_powerlaw_exp(ss, np.arange(10, 500))[0]
        evals_svca2_all.append(ss)
        areas_all.append(area)

        ss = SVCA(spks_gpu, xpos=xpos, ypos=ypos)[0]
        alpha_svca = fit_powerlaw_exp(ss, np.arange(10, 500))[0]
        evals_svca_all.append(ss)

        # shuffle in time
        torch.random.manual_seed(0)
        for j in range(500):
            irand = torch.randperm(spks.shape[0])[:spks.shape[0]//2]
            trand = np.random.randint(0, spks.shape[1])
            spks_gpu[irand] = torch.roll(spks_gpu[irand], trand, dims=1)
        ss = SVCA2(spks_gpu, xpos=xpos, ypos=ypos)
        alpha_shuff = fit_powerlaw_exp(ss, np.arange(10, 500))[0]
        evals_shuff_all.append(ss)
        del spks_gpu
        torch.cuda.empty_cache()

        print(f"\talpha: direct={alpha:.2f}, SVCA2={alpha_svca2:.2f}, SVCA={alpha_svca:.2f}, SVCA2_shuff={alpha_shuff:.2f}")

        # rastermap
        if i in dbex:
            print("running rastermap")
            model = Rastermap(n_clusters=100, n_PCs=128, normalize=False, bin_size=20).fit(spks)
            Xexs.append(model.X_embedding)
            isorts.append(model.isort)

    
    # ephys data
    tbin = 1. / 22 # 22 Hz binning
    for i, db in enumerate(dbs_ephys):
        mouse_name = db["mouse_name"]
        dat = np.load(root / f"{mouse_name}_spks_face.npz", allow_pickle=True)
        areas_all.append("ephys")
        area = dat["areas"]
        spks = dat["spks"].astype("float32").copy()
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
            
        # powerlaw
        spks_gpu = torch.from_numpy(spks.copy()).to(device)
        evals, evecs = torch.linalg.eigh((spks_gpu @ spks_gpu.T) / spks_gpu.shape[1])
        evals = evals.cpu().numpy()[::-1]
        alpha = fit_powerlaw_exp(evals, np.arange(10, 500))[0]
        evals_all.append(evals)

        ss = SVCA2(spks_gpu, xpos=xpos, ypos=ypos, spacing=40)
        alpha_svca2 = fit_powerlaw_exp(ss, np.arange(10, 500))[0]
        evals_svca2_all.append(ss)

        ss = SVCA(spks_gpu, xpos=xpos, ypos=ypos, spacing=40)[0]
        alpha_svca = fit_powerlaw_exp(ss, np.arange(10, 500))[0]
        evals_svca_all.append(ss)

        # shuffle in time
        torch.random.manual_seed(0)
        for j in range(500):
            irand = torch.randperm(spks.shape[0])[:spks.shape[0]//2]
            trand = np.random.randint(0, spks.shape[1])
            spks_gpu[irand] = torch.roll(spks_gpu[irand], trand, dims=1)
        ss = SVCA2(spks_gpu, xpos=xpos, ypos=ypos)
        alpha_shuff = fit_powerlaw_exp(ss, np.arange(10, 500))[0]
        evals_shuff_all.append(ss)
        del spks_gpu
        torch.cuda.empty_cache()

        print(f"\talpha: direct={alpha:.2f}, SVCA2={alpha_svca2:.2f}, SVCA={alpha_svca:.2f}, SVCA2_shuff={alpha_shuff:.2f}")

        # rastermap
        if i==1:
            print("running rastermap")
            model = Rastermap(n_clusters=100, n_PCs=128, bin_size=10).fit(spks)
            Xexs.append(model.X_embedding)
            isorts.append(model.isort)

    return evals_all, evals_svca2_all, evals_svca_all, evals_shuff_all, areas_all, Xexs, isorts, snr_rat
