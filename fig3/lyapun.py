import numpy as np
import torch
from torch.fft import fft, ifft 
device = torch.device('cuda')
from tqdm import trange
from torchaudio.functional import fftconvolve
from sklearn.decomposition import TruncatedSVD 

# generate a random matrix 
def getA(N, symm = True, device = torch.device('cuda')):

    A = torch.randn((N, N), device = device, dtype = torch.float32)
    
    # \sqrt{m} is the normalizer, eps=1e-3 for conditioning
    m = N
    eps = 1e-3
    if symm:
        A          = symmetrize(A)        
        A = .5 * A / (eps +  m)**.5
    else:
        A = 1 * A / (eps +  m)**.5
    return A

# symmetrize and remove the diagonal
def symmetrize(A):
    A = A - torch.triu(A)
    A = A + A.T
    return A 

def Aregularize(A, symm=True, emax = 0.998):
    if symm:        
        evals0,u = torch.linalg.eigh(A)                
    else: 
        evals0,u = torch.linalg.eig(A)        
    
    enorm = torch.real(evals0).max() / emax    

    A = A /enorm

    return A

# this zscore function works for torch data as well as numpy data
def zscore(X, axis = -1, eps = 1e-3, itrain = None):
    if itrain is None:        
        X = X - X.mean(axis=axis,keepdims=True)
        X = X/ (eps + (X**2).mean(axis=axis,keepdims=True)**.5)
    else:
        X = X - X[:,itrain].mean(axis=axis,keepdims=True)
        X = X/ (eps + (X[:,itrain]**2).mean(axis=axis,keepdims=True)**.5)
    return X

def simulateA(A, dt = 2, tau = 20, device = torch.device('cuda')):
    nn = A.shape[0]
    
    T = 120 * (1000//dt) 
    nd = 50
    X = torch.randn((nn, nd), device=device) 
    
    nsamp = int(1000/dt/22)    

    Xt = torch.zeros((T//nsamp, nn, nd), device=device)

    for t in trange(T):
        eps = torch.randn((nn, nd), device=device)
        X += dt / tau * (-X + A @ X + eps)
        if nd <= 250 and t//nsamp < T//nsamp:
            Xt[t//nsamp] += X
        
    Xt /= nsamp    
    
    Xt = Xt.permute(1, 2, 0)[:,:,20:].reshape(nn, -1)
    
    Xt -= Xt.mean(axis=1, keepdim=True)
    Xt /= Xt.std(axis=1, keepdim=True)
    
    return Xt

def dmd(sp, lam = .01, delta = 1, device = torch.device('cuda')):
    
    X = torch.from_numpy(sp).to(device)
    
    NT = X.shape[1]
    cov = (X[:,:NT] @ X[:,:NT].T)/X.shape[0]
    e, u = torch.linalg.eigh(cov @ cov.T)
    
    X = u.T @ X
    
    At = dynamics_lag(X[-1000:], delta = delta, lam = lam, device = device)
    e = torch.linalg.eigvals(At)
    e = e.cpu().numpy()

    return e

def dynamics_lag(X, Y = None, delta = 10, lam = 0.1, device = torch.device('cuda')):
    NN, NT = X.shape
    if Y is None:
        Y = X
    nt = 10000 

    #dt = 10 
    nblocks = (NT-delta)//nt

    xxt = torch.zeros((NN, NN), device = device)
    xyt = torch.zeros((NN, NN), device = device)

    for j in range(nblocks):
        x = X[:,j*nt:j*nt + nt]
        y = Y[:,j*nt+delta:j*nt + nt+delta]

        xxt += (x @ x.T)/nt
        xyt += (x @ y.T)/nt 

    xxt /= nblocks
    xyt /= nblocks

    teye = torch.eye(NN, device = device)

    A = torch.linalg.solve(xxt + lam * teye, xyt).T
    return A


def ephys_load(dat):
    sp = dat['spks']
    ypos = dat['ypos']
    xpos = dat['iprobe']*25
    sp = sp.astype('float32')
    ypos = ypos.astype('float32')
    xpos = xpos.astype('float32')
    return sp, xpos, ypos

def pc_timescales(Xdev, xpos, ypos, sig = 0, device = torch.device('cuda')):
    NN, NT = Xdev.shape 

    tblock = NT//20
    #tblock = 2000
    iblock = np.arange(NT)//tblock

    Xdev = Xdev[:,:tblock*(NT//tblock)].reshape((NN, -1, tblock))
    Xdev = zscore(Xdev, axis = -1)

    iblock = np.arange(Xdev.shape[1])

    Xs = Xdev[:,iblock%2==0, :].reshape((NN, -1))

    dx = (xpos%50<25).astype('int32')
    dy = (ypos%50<25).astype('int32')
    ix = (dx + dy)%2==0

    Xs = torch.from_numpy(Xs).to(device)
    if sig>0:
        kern = torch.exp(-torch.arange(-20,21, device = device)**2 / (2*sig**2))
        Xsm = torch.nn.functional.conv1d(Xs.unsqueeze(1), kern.unsqueeze(0).unsqueeze(0)).squeeze(1)
    
        cov = (Xsm[ix] @ Xsm[~ix].T)/Xsm.shape[1]
    else:
        cov = (Xs[ix] @ Xs[~ix].T)/Xs.shape[1]
    ss,u = torch.linalg.eigh(cov @ cov.T)

    v = cov.T @ u
    v = v/ (v**2).sum(0)**.5

    if sig>0:
        cov2 = (Xs[ix] @ Xs[~ix].T)/Xs.shape[1]
        v2 = ((u * (cov2 @ v))**2).sum(0) 
        isort = torch.argsort(v2)
        u = u[:,isort]
        v = v[:,isort]

    Ys = torch.from_numpy(Xdev[:,iblock%2==1]).to(device)

    Xpca1 = u[:,-1000:].T @ Ys[ix].reshape((ix.sum(), -1))
    Xpca2 = v[:,-1000:].T @ Ys[~ix].reshape(((~ix).sum(), -1))

    Xpca1 = zscore(Xpca1, axis=-1)/tblock**.5    
    Xpca2 = zscore(Xpca2, axis=-1)/tblock**.5    

    Xpca1 = Xpca1.reshape((Xpca1.shape[0], -1, tblock))
    Xpca2 = Xpca2.reshape((Xpca2.shape[0], -1, tblock))

    fX1 = fft(Xpca1, dim = -1)
    fX2 = fft(Xpca2, dim = -1)
    ac = ifft(fX1 * torch.conj(fX2),dim = -1).real

    
    ac = ac.mean(1).cpu().numpy()
    ac_all = ac[::-1]

    acg = ac_all[:, :100]

    return acg

    
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
    elif distribution == "sparse":        
        A = (torch.rand((nn, nn), device=device)<.10).float()
        A = A - A.mean()
       
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

def zscore(X, axis = -1, eps = 1e-3, itrain = None):
    if itrain is None:        
        X = X - X.mean(axis=axis,keepdims=True)
        X = X/ (eps + (X**2).mean(axis=axis,keepdims=True)**.5)
    else:
        X = X - X[:,itrain].mean(axis=axis,keepdims=True)
        X = X/ (eps + (X[:,itrain]**2).mean(axis=axis,keepdims=True)**.5)
    return X
