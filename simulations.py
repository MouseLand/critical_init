import torch
from tqdm import trange
from powerlaw import fit_powerlaw_exp, zscore_and_compute_evals, SVCA, SVCA2
from torchaudio.functional import fftconvolve
import numpy as np
from rastermap import Rastermap

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

def simulate_random(nn=10000, nonsym=0, nd=200, distribution="uniform", T=60000, 
                    tpad=4000, dt=2, tau=50, tbin=23, device=torch.device("cuda")):
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
    T = 60000 # number of timesteps
    tpad = 4000 # number of initial timepoints to exclude (to exclude transient dynamics)
    dt = 2 # time step
    tau = 50 # neuron time constant
    tbin = 23 # time binning

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
    enorm = evals0.real.max() / 0.999
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
            Xbin = Xi[:,:(nt//tbin0)*tbin0].reshape(nn, -1, tbin0).mean(axis=-1)
        else:
            Xbin = Xi
        evals, evecs = zscore_and_compute_evals(Xbin)
        ntmax = Xbin.shape[1]
        evals_all[ti, :ntmax] = evals[:ntmax]
    return evals_all

def random_uniform_sim(n_sim = 10, nn = 10000, device = torch.device("cuda")):
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
    nonsyms = [0, 1./3, 2./3, 1]
    tbins = np.unique(np.round(np.exp(np.linspace(np.log(1), np.log(1000), 20)))).astype(int)
    evals_all = np.zeros((n_sim, len(nonsyms), nn))
    evals_bin_all = np.zeros((n_sim, 2, len(tbins), nn))
    for i in range(n_sim):
        for ni, nonsym in enumerate(nonsyms):
            A, Xi, evals0 = simulate_random(nn=nn, nonsym=nonsym, nd=200)

            # keep example connectivity matrix and sims
            if i==0 and (nonsym==0 or nonsym==1):
                if nonsym==0:
                    Xsym_ex = Xi[:, :50000].cpu().numpy()
                else:
                    Anonsym_ex = A.cpu().numpy()
                    Xnonsym_ex = Xi[:, :50000].cpu().numpy()
                   
            # compute eigenvalues of covariance matrix
            evals, evecs = zscore_and_compute_evals(Xi)
            alpha, yp = fit_powerlaw_exp(evals, np.arange(10, 500))
            print(fr"{i} {nonsym:.2f}; alpha = {alpha:.3f}")
            
            evals_all[i, ni] = evals.copy()

            # compute eigenvalues for different time binning
            if nonsym==0 or nonsym==1:
                evals_bin_all[i, nonsym] = compute_evals_tbin(Xi, tbins)
            
            del Xi

    # sort neurons with rastermap
    model = Rastermap(mean_time=False, bin_size=25, 
                  time_lag_window=0, locality=0.).fit(Xsym_ex)
    Xsym_ex = model.X_embedding
    model = Rastermap(mean_time=False, bin_size=25, 
                  time_lag_window=0, locality=0.).fit(Xnonsym_ex)
    Xnonsym_ex = model.X_embedding
                
    
    return (evals_all, nonsyms, Xsym_ex, Xnonsym_ex, evals_bin_all, tbins)


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
            A, Xi, evals0 = simulate_random(nn=nn, nonsym=nonsym, nd=200,
                                            distribution=distributions[si], 
                                            device=device)
            
            evals, evecs = zscore_and_compute_evals(Xi)
            evals_all[i, si] = evals

            alpha, yp = fit_powerlaw_exp(evals, np.arange(10, 500))
            print(f"{distributions[si]}, {alpha:.3f}")
            
            if i==0:
                Aexs[si] = A.cpu().numpy()
    
    return evals_all, Aexs, distributions


def random_uniform_sim_noisy(n_sim = 10, nn = 10000, device = torch.device("cuda")):
    """ simulate neurons with random uniform connectivity matrix and compute SVCA 

        simulates fewer example sims to represent real neural data, and adds 
        gaussian noise and smooths in time;
        saves output in evals_svca_all.npy

    Args:
        n_sim: number of simulations
        nn: number of neurons
        device: torch device

    Returns:
        A tuple containing evals_gt_all, evals_all, evals_svca_all, evals_svca2_all, nonsyms
    
    """

    set_seed(0, device)

    # nonsymmetry levels of random connectivity matrix
    nonsyms = [0, 1./3, 2./3, 1]
    evals_all = np.zeros((n_sim, len(nonsyms), nn)) * np.nan
    evals_gt_all = np.zeros((n_sim, len(nonsyms), nn)) * np.nan
    evals_svca_all = np.zeros((n_sim, len(nonsyms), nn)) * np.nan
    evals_svca2_all = np.zeros((n_sim, len(nonsyms), nn)) * np.nan
    for i in range(n_sim):
        for ni, nonsym in enumerate(nonsyms):
            A, Xi, evals0 = simulate_random(nn=nn, nonsym=nonsym, nd=10)
                   
            # compute eigenvalues of covariance matrix
            evals_gt, evecs = zscore_and_compute_evals(Xi)
            evals_gt_all[i, ni] = evals_gt

            # add noise and smooth in time
            Xi += 1.0 * torch.randn(Xi.shape, device=device)
            sig = 1
            gaussian = torch.exp(- torch.arange(-4, 5, dtype=torch.float)**2 / 2*sig**2)
            gaussian /= gaussian.sum()
            Xi = Xi.cpu()
            Xi = fftconvolve(Xi, gaussian.unsqueeze(0), mode="same")
            Xi -= Xi.mean(axis=1, keepdim=True)
            Xi /= Xi.std(axis=1, keepdim=True)
            Xi = Xi.to(device)
            
            # compute eigenvalues directly
            evals, evecs = zscore_and_compute_evals(Xi)
            evals_all[i, ni] = evals
            
            # compute eigenvalues with SVCA and SVCA2
            evals_svca = SVCA(Xi)[0]
            evals_svca_all[i, ni, :len(evals_svca)] = evals_svca
            evals_svca2 = SVCA2(Xi)
            evals_svca2_all[i, ni, :len(evals_svca2)] = evals_svca2

            # power-law decay exponent
            alpha_gt = fit_powerlaw_exp(evals_gt[:1000], np.arange(10, 500))[0]
            alpha = fit_powerlaw_exp(evals[:1000], np.arange(10, 500))[0]
            alpha_svca = fit_powerlaw_exp(evals_svca[:1000], np.arange(10, 500))[0]
            alpha_svca2 = fit_powerlaw_exp(evals_svca2[:1000], np.arange(10, 500))[0]
            
            print(f"{i} {nonsym:.2f}; alpha_gt: {alpha_gt:.2f}, alpha: {alpha:.2f}, alpha_svca: {alpha_svca:.2f}, alpha_svca2: {alpha_svca2:.2f}")    

            del Xi
            
    return evals_gt_all, evals_all, evals_svca_all, evals_svca2_all, nonsyms

