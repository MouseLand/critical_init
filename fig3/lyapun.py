import numpy as np
import torch
from torch.fft import fft, ifft 

def getdb():
    db = [] 
    
    db.append({'mname': 'FX26'})
    db.append({'mname': 'FX14'}) 
    db.append({'mname': 'FX17'}) 
    db.append({'mname': 'FX42'}) 
    db.append({'mname': 'FX43'}) 
    db.append({'mname': 'QZ2'}) 
    
    db.append({'mname': 'GP9'})  
    db.append({'mname': 'GP7'})      
    db.append({'mname': 'GP10'})  
    db.append({'mname': 'GP8'})  

    return db

def SVCA2(X, xpos=None, ypos=None, dt = 0, device = torch.device('cuda')):
    if not torch.is_tensor(X):
        X = torch.from_numpy(X).to(device)
    NN,NT = X.shape

    if xpos is None or ypos is None: 
        ix = np.random.rand(NN)>.5
    else:
        dx = (xpos%50<25).astype('int32')
        dy = (ypos%50<25).astype('int32')
        ix = (dx + dy)%2==0

    ntrain = ix.nonzero()[0]
    ntest  = (~ix).nonzero()[0]


    cov = (X[ntrain, dt:] @ X[ntest, :NT-dt].T)

    e, u = torch.linalg.eigh(cov @ cov.T)

    v = u.T @ cov 
    ss = (v**2).sum(-1)**.5
    
    return ss.cpu().numpy()[::-1]

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

# this zscore function works for torch data as well as numpy data
def zscore(X, axis = -1, eps = 1e-3):
    X = X - X.mean(axis=axis,keepdims=True)
    X = X/ (eps + (X**2).mean(axis=axis,keepdims=True)**.5)
    return X

from tqdm import trange
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



def Aregularize(A, symm=True):

    if symm:        
        e,u = torch.linalg.eigh(A)
        emax = 1.01 * e.max()
        e = e/emax
        A = (u * e) @ u.T
    else: 
        e,u = torch.linalg.eig(A)
        r = torch.abs(e)
        r[r>.99]  = .99
        enorm = e * r / torch.abs(e) 
        A = (u * enorm) @ torch.linalg.inv(u)
        A = A.real

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

    tblock = NT//10
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

    Xpca1 = Xpca1.reshape((Xpca1.shape[0], -1, tblock))
    Xpca2 = Xpca2.reshape((Xpca2.shape[0], -1, tblock))

    fX1 = fft(Xpca1, dim = -1)
    fX2 = fft(Xpca2, dim = -1)
    ac = ifft(fX1 * torch.conj(fX2),dim = -1).real

    ac = ac.mean(1).cpu().numpy()
    ac = ac/ac[:,:1]
    ac_all = ac[::-1]

    acg = ac_all[:, :100]

    return acg
