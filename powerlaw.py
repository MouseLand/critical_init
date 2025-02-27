import numpy as np 
from scipy.stats import zscore
import torch
from tqdm import trange, tqdm 
from sklearn.decomposition import TruncatedSVD
from rastermap import Rastermap

def SVCA2(X, xpos=None, ypos=None, dt = 0, spacing=25):
    """ compute SVCA2 for neural data
    
    Args:
        X: torch tensor of neurons by time
        xpos: numpy array of x positions
        ypos: numpy array of y positions
        dt: time lag
        spacing: spacing of neurons in pixels for splitting 

    Returns:
        ss: singular values of covariance matrix
    
    """
    NN,NT = X.shape

    if xpos is None or ypos is None: 
        ix = np.random.rand(NN)>.5
    else:
        dx = (xpos % (spacing*2) < spacing).astype('int32')
        dy = (ypos % (spacing*2) < spacing).astype('int32')
        ix = (dx + dy)%2==0

    ntrain = ix.nonzero()[0]
    ntest  = (~ix).nonzero()[0]

    cov = (X[ntrain, dt:] @ X[ntest, :NT-dt].T)

    e, u = torch.linalg.eigh(cov @ cov.T)

    v = u.T @ cov 
    ss = (v**2).sum(-1)**.5
    
    return ss.cpu().numpy()[::-1]

def split_traintest(n_t, frac=0.2, pad=3, fold=0, split_time=False):
    """ Returns a deterministic split of train and test in time chunks.
    
    Args:
        n_t (int): Number of timepoints to split.
        frac (float, optional): Fraction of points to put in test set. Defaults to 0.2.
        pad (int, optional): Number of timepoints to exclude from test set before and after training segment. Defaults to 3.
        fold (int, optional): Fold index for cross-validation. Defaults to 0.
        split_time (bool, optional): Split train and test into beginning and end of experiment. Defaults to False.
    
    Returns:
        itrain (numpy.ndarray): 2D array of times in train set, arranged in chunks.
        itest (numpy.ndarray): 2D array of times in test set, arranged in chunks.
                    split_time=False):
    """
    n_segs = int(min(40, n_t/4)) 
    n_len = int(np.floor(n_t/n_segs)) - pad
    inds = np.linspace(0, n_t - n_len - 5, n_segs).astype("int")
    n_segs_test = int(np.floor(n_segs*frac))
    split = n_segs // n_segs_test
    np.random.seed(0)
    iseg = np.random.randint(0, split, n_segs)
    itest0 = iseg==fold
    itrain0 = ~itest0
    inds_test = inds[itest0]
    inds_train = inds[itrain0]
    itrain = (inds_train[:,np.newaxis] + np.arange(0, n_len, 1, int))
    itest = (inds_test[:,np.newaxis] + np.arange(0, n_len, 1, int))
    
    return itrain, itest


def SVCA(X, xpos=None, ypos=None, spacing=25, frac=0.25, rand_split=True):
    """ compute SVCA for neural data

    splits neurons and timepoints 

    Args:
        X: torch tensor of neurons by time
        xpos: numpy array of x positions
        ypos: numpy array of y positions
        spacing: spacing of neurons in pixels for splitting
        frac: fraction of timepoints to put in test set
        rand_split: random split of timepoints
    
    Returns:
        scov: estimated singular values of covariance matrix
        varcov: variance of projections

    """
    NN,NT = X.shape

    if xpos is None or ypos is None: 
        ix = np.random.rand(NN)>.5
    else:
        dx = (xpos % (spacing*2) < spacing).astype('int32')
        dy = (ypos % (spacing*2) < spacing).astype('int32')
        ix = (dx + dy)%2==0

    ntrain = ix.nonzero()[0]
    ntest  = (~ix).nonzero()[0]

    if rand_split:
        itrain, itest = split_traintest(NT, frac=frac)
        itrain = itrain.flatten()
        itest = itest.flatten()
    else:
        itrain = np.zeros(NT, dtype=bool)
        itrain[:int(NT*(1-frac))] = True
        itest = ~itrain

    cov = X[ntrain][:, itrain] @ X[ntest][:, itrain].T

    e, u = torch.linalg.eigh(cov @ cov.T)

    v = cov.T @ u
    
    u /= (u**2).sum(axis=0)**0.5
    v /= (v**2).sum(axis=0)**0.5

    strain = u.T @ X[ntrain][:, itest]
    stest = v.T @ X[ntest][:, itest]

    # covariance k is uk.T * F * G.T * vk / npts
    scov = (strain * stest).mean(axis=1)
    varcov = (strain**2 + stest**2).mean(axis=1) / 2

    return scov.cpu().numpy()[::-1], varcov.cpu().numpy()[::-1]

def zscore_and_compute_evals(Xi):
    """ compute eigenvalues of covariance matrix after zscoring Xi in place
    
    Args:
        Xi: torch tensor of neurons by time
    """
    Xi -= Xi.mean(axis=1, keepdim=True)
    Xi /= Xi.std(axis=1, keepdim=True)        
    cov = Xi @ Xi.T / Xi.shape[1]
    evals, evecs = torch.linalg.eigh(cov)
    evals = evals.cpu().numpy()[::-1]
    evecs = evecs.cpu().numpy()[:,::-1]
    return evals, evecs


def fit_powerlaw_exp(ss, trange):
    """ fit powerlaw to eigenvalues 

    Args:
        ss: eigenvalues
        trange: PC indices to fit powerlaw to

    Returns:
        alpha: fitted exponent
        ypred: predicted eigenvalues from powerlaw fit

    """
    logss = np.log(np.abs(ss))
    y = logss[trange][:,np.newaxis]
    trange += 1
    nt = trange.size
    x = np.concatenate((-np.log(trange)[:,np.newaxis], np.ones((nt,1))), axis=1)
    w = 1.0 / trange.astype(np.float32)[:,np.newaxis]
    b = np.linalg.solve(x.T @ (x * w), (w * x).T @ y).flatten()

    allrange = np.arange(0, ss.size).astype(int) + 1
    x = np.concatenate((-np.log(allrange)[:,np.newaxis], np.ones((ss.size,1))), axis=1)
    ypred = np.exp((x * b).sum(axis=1))
    alpha = b[0]

    ir = np.exp(np.linspace(np.log(trange[0]), np.log(trange[-1]-1), 100)).astype(int)
    r = (zscore(np.log(ir)) * zscore(logss[ir])).mean()

    return alpha, ypred