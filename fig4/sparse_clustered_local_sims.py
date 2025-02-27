import torch 
import numpy as np
from tqdm import trange
from powerlaw import fit_powerlaw_exp
import analysis

def sparse_evals(n_sim = 10, nn = 10000, device = torch.device("cuda")):
    """ compute eigenvalues for a random matrix with different sparsities

        saves output in sim_sparse.npy

    Args:
        n_sim: number of simulations
        nn: number of neurons
        device: torch device

    Returns:
        A tuple containing evals_sparsities_all, sparsities

    """
    sparsities = 2.**np.arange(-12, -1, 1)
    
    n_sparsities = len(sparsities)
    evals_sparsities_all = np.zeros((n_sim, n_sparsities, nn), "float32")
    alphas = np.zeros((n_sim, n_sparsities), "float32")
    for ns in trange(n_sim):
        for n, sparsity in enumerate(sparsities):
            # connectivity matrix with bernoulli(p=sparsity)
            A = (torch.rand((nn, nn), device=device) < sparsity).float()
            # subtract by sparsity rate to make matrix zero mean
            A -= sparsity 
            # symmetrize
            A -= torch.triu(A) 
            A = 0.5 * (A + A.T)
            A /= (nn * sparsity)**0.5

            # compute eigenvalues
            evals0, U = torch.linalg.eigh(A)
            evals = evals0.cpu().numpy()[::-1]
            # normalize eigenvalues so they are less than 1
            enorm = evals.max() / 0.999
            evals /= enorm

            # eigenvalues of covariance matrix are 0.5 / (1 - eigenvalues of A)
            evals = 0.5 / (1 - evals)
            evals_sparsities_all[ns, n] = evals
            alpha, yp = fit_powerlaw_exp(evals, np.arange(10, 500))
            alphas[ns, n] = alpha
            if ns == n_sim - 1:
                alpha0 = alphas[:,n].mean(axis=0)
                print(f"p(conn) (%) = {sparsity*100:.3f}, alpha = {alpha0:.2f}")
    
    return evals_sparsities_all, sparsities

def clustered_evals(n_sim = 10, nn = 10000, ncl = 500, device = torch.device("cuda")):
    """ compute eigenvalues for a matrix with clustered connectivity
    
        saves output in sim_cluster.npy

    Args:
        n_sim: number of simulations
        nn: number of neurons
        ncl: number of neurons per cluster
        device: torch device

    Returns:
        A tuple containing evals_cluster_all, Aex, pglobals

    """
    pglobals = 2.**np.arange(-12, -0, 1)
    n_globals = len(pglobals)
    
    evals_cluster_all = np.zeros((n_sim, n_globals, nn), "float32")
    var_all = np.zeros((n_sim, n_globals), "float32")
    alphas = np.zeros((n_sim, n_globals), "float32")
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    for ns in trange(n_sim):
        for n, pglobal in enumerate(pglobals):
            # connectivity matrix with bernoulli(p=pglobal) for global connections
            # and bernoulli(p=0.5) for connections within clusters
            sparsity_mat = torch.ones((nn, nn), device=device) * pglobal
            for i in range(nn//ncl):
                sparsity_mat[i * ncl : (i+1) * ncl, i * ncl : (i+1) * ncl] = 0.5
            A = (torch.rand((nn, nn), device=device) < sparsity_mat).float()
            A -= sparsity_mat
            # symmetrize
            A -= torch.triu(A)
            A = 0.5 * (A + A.T)
            A /= (nn * sparsity_mat.mean())**0.5

            # compute eigenvalues
            evals0, U = torch.linalg.eigh(A)
            evals = evals0.cpu().numpy()[::-1]
            # normalize eigenvalues so they are less than 1
            enorm = evals.max() / 0.999
            evals /= enorm
            
            # eigenvalues of covariance matrix are 0.5 / (1 - eigenvalues of A)
            evals = 0.5 / (1 - evals)
            alpha, yp = fit_powerlaw_exp(evals, np.arange(10, 500))
            evals_cluster_all[ns, n] = evals

            if ns==0 and n==6:
                Aex = A.cpu().numpy() / enorm

            alphas[ns, n] = alpha
            if ns == n_sim - 1:
                alpha0 = alphas[:,n].mean(axis=0)
                print(f"p(local) / p(global) = {int(0.5/pglobal)}, alpha = {alpha0:.2f}")
    
    return evals_cluster_all, Aex, pglobals
    
def local_connectivity(nn=10000, sig=250, isig=1, pglobal=0.04, 
                plocal=125, device=torch.device("cuda")):
    ypos = 8000 * torch.rand((nn,), device=device) #8000
    xpos = 8000 * torch.rand((nn,), device=device)

    dxmin = analysis.circ_dist(xpos)
    dymin = analysis.circ_dist(ypos)

    dist = (dymin**2 + dxmin**2)**0.5
    sparsity_mat = plocal * torch.exp(-dist / sig) / sig
    
    sparsity_mat_I = plocal * torch.exp(-dist / (isig * sig)) / (isig * sig)    
    sparsity_mat_I /= sparsity_mat_I.mean() / sparsity_mat.mean() 
    
    sparsity_mat[sparsity_mat < pglobal] = pglobal
    sparsity_mat_I[sparsity_mat_I < pglobal] = pglobal

    sparsity_mat -= torch.diag(torch.diag(sparsity_mat))
    sparsity_mat_I -= torch.diag(torch.diag(sparsity_mat_I))  

    return sparsity_mat, sparsity_mat_I, ypos, xpos  

def local_evals(n_sim = 10, nn = 10000, device = torch.device("cuda")):
    """ compute eigenvalues for a matrix with local connectivity

        saves output in sim_local.npy

    Args:
        n_sim: number of simulations
        nn: number of neurons
        device: torch device

    Returns:
        A tuple containing evals_local_all, perc, perc_local_all, var_all, pglobals, pcs_local_all, ypos_local_all, xpos_local_all, 
        Aex, cov_ex, ypos_ex, xpos_ex, dbin_strong_all, dbin_other_all, cov_bin_all, drand_all, crand_all, pconn, bin_size, pstrong, dist_min

    """
    sig = 250 # local connectivity
    pglobals = 2.**np.arange(-12, -0, 1)
    n_globals = len(pglobals)
    
    # strongpair parameters
    bin_size = 200
    nbins = 15
    pstrong = 0.01
    dist_min = 20

    perc = 10**np.arange(-3, 0, 0.25)
    
    pconn = np.zeros(n_globals, "float32")
    evals_local_all = np.zeros((n_sim, n_globals, nn), "float32")
    perc_local_all = np.zeros((n_sim, n_globals, len(perc)), "float32")
    pcs_local_all = np.zeros((n_globals, nn, 100), "float32")
    var_all = np.zeros((n_sim, n_globals), "float32")
    ypos_local_all = np.zeros((n_globals, nn), "float32")
    xpos_local_all = np.zeros((n_globals, nn), "float32")
    dbin_strong_all = np.zeros((n_sim, n_globals, nn, nbins), "uint16")
    dbin_other_all = np.zeros((n_sim, n_globals, nn, nbins), "uint16")
    cov_bin_all = np.zeros((n_sim, n_globals, nn, nbins), "float32")
    drand_all = np.zeros((n_sim, n_globals, 20000), "float32")
    crand_all = np.zeros((n_sim, n_globals, 20000), "float32")
    device = torch.device("cuda")

    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    alphas = np.zeros((n_sim, n_globals), "float32")
    for ns in trange(n_sim):
        for n, pglobal in enumerate(pglobals):
            # connectivity matrix with local connectivity with exponential decay
            # and global connectivity with bernoulli(p=pglobal)
            sparsity_mat, sparsity_mat_I, ypos, xpos = local_connectivity(nn, sig=sig,
                                                                          plocal=sig*0.5,
                                                                          pglobal=pglobal)
            if ns == 0:
                pconn[n] = sparsity_mat.mean().item()
            A = (torch.rand((nn, nn), device=device) < sparsity_mat).float()
            A -= sparsity_mat_I
            # symmetrize
            A -= torch.triu(A)
            A = 0.5 * (A + A.T)

            # compute eigenvalues
            evals0, U = torch.linalg.eigh(A)
            evals = evals0.cpu().numpy()[::-1]
            # normalize eigenvalues so they are less than 1
            enorm = evals.max() / 0.999
            evals /= enorm
            
            # eigenvalues of covariance matrix are 0.5 / (1 - eigenvalues of A)
            evals = 0.5 / (1 - evals)
            alpha, yp = fit_powerlaw_exp(evals, np.arange(10, 500))
            evals_local_all[ns, n] = evals
            alphas[ns, n] = alpha
            
            # compute covariance matrix
            cov = (U * torch.from_numpy(evals[::-1].copy()).to(device)) @ U.T
            cov /= torch.outer(cov.diag(), cov.diag())**0.5
            cov -= torch.diag(torch.nan * cov.diag())

            # sort covariance matrix
            cov_sort, isort = torch.sort(cov, axis=1)
            Asort = torch.stack([A[i, isort[i]] for i in range(nn)], axis=0)
            # check connectivity from A for different top % covariance pairs
            ineurs = np.round(perc * nn).astype(int)
            itrue = torch.stack([(Asort[:, -ineur:] > 0).float().mean(axis=1) for ineur in ineurs], axis=1)
            perc_local_all[ns, n] = itrue.mean(axis=0).cpu().numpy()

            cov[torch.isnan(cov)] = 0
            cov = cov.cpu().numpy()

            # compute strong pairs and distance dependence
            dbin_strong, dbin_other, cov_bin, drand, crand = analysis.strong_pairs(ypos, xpos, 
                                                                                cov=cov,
                                                                                pstrong=pstrong,
                                                                                bin_size=bin_size, 
                                                                                nbins=nbins, 
                                                                                dist_min=dist_min,
                                                                                device=device)
            dbin_strong_all[ns, n] = dbin_strong
            dbin_other_all[ns, n] = dbin_other
            cov_bin_all[ns, n] = cov_bin
            drand_all[ns, n] = drand
            crand_all[ns, n] = crand

            # compute spatial distribution of variance of top 100 PCs
            U = U[:,-100:]
            # compute center of mass in circular coordinates
            theta_y = 2 * np.pi * ypos / 8000 - np.pi
            theta_x = 2 * np.pi * xpos / 8000 - np.pi
            com_y = (U**2 * torch.exp(1j * theta_y.unsqueeze(1))).sum(axis=0)
            com_y /= (U**2).sum(axis=0)
            com_y = torch.angle(com_y)
            com_x = (U**2 * torch.exp(1j * theta_x.unsqueeze(1))).sum(axis=0)
            com_x /= (U**2).sum(axis=0)
            com_x = torch.angle(com_x)
            # convert to euclidean coordinates
            com_y += np.pi
            com_x += np.pi
            com_y *= 8000 / (2 * np.pi)
            com_x *= 8000 / (2 * np.pi)
            # compute variance of PC weights within 250*1.5 um of center of mass
            ydist = torch.minimum(torch.abs(com_y - ypos.unsqueeze(1)), 
                                        torch.abs(1 - (com_y - ypos.unsqueeze(1))))
            xdist = torch.minimum(torch.abs(com_x - xpos.unsqueeze(1)), 
                                torch.abs(1 - (com_x - xpos.unsqueeze(1))))
            udist = (xdist**2 + ydist**2)**0.5
            varin = torch.tensor([(U[udist[:,i] <= sig*1.5, i]**2).mean() for i in range(U.shape[1])])
            # normalize by total variance
            varall = (U**2).mean(axis=0).cpu()
            varrat = (varin / varall).mean().item()

            var_all[ns, n] = varrat
            
            nstrong = int(np.round(pstrong * nn))
            nother = nn - nstrong
            dnorm = (np.nanmean(dbin_strong, axis=0) / nstrong / 
                      (np.nanmean(dbin_other, axis=0) / nother))
            
            if ns==0:
                # save U, ypos, xpos from one set of sims
                U = U.cpu().numpy()[:, ::-1]
                ypos = ypos.cpu().numpy()
                xpos = xpos.cpu().numpy()
                U /= (U**2).sum(axis=1, keepdims=True)**0.5
                pcs_local_all[n] = U
                ypos_local_all[n] = ypos
                xpos_local_all[n] = xpos

                if n==6:
                    # save example connectivity matrix
                    Aex = A.cpu().numpy() / enorm
                    cov_ex = cov
                    ypos_ex = ypos
                    xpos_ex = xpos
            
            if ns == n_sim - 1:
                alpha0 = alphas[:,n].mean(axis=0)
                print(f"p(local) / p(global) = {int(0.5/pglobal)}, alpha = {alpha0:.2f}")
    

       
    return (evals_local_all, perc, perc_local_all, var_all, pglobals, pcs_local_all, ypos_local_all, xpos_local_all, 
            Aex, cov_ex, ypos_ex, xpos_ex, dbin_strong_all, dbin_other_all, cov_bin_all, drand_all, crand_all, 
            pconn, bin_size, pstrong, dist_min)
        