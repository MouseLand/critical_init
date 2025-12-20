import numpy as np
import torch
from torch.fft import fft, ifft 
from tqdm import trange


# this zscore function works for torch data as well as numpy data
def zscore(X, axis = -1, eps = 1e-3):
    X = X - X.mean(axis=axis,keepdims=True)
    X = X/ (eps + (X**2).mean(axis=axis,keepdims=True)**.5)
    return X


def dynamics_lag(X, Y = None, delta = 10, lam = 0.1, nt=10000, valid_inds=None,
                 device = torch.device('cuda')):
    NN, NT = X.shape
    if Y is None:
        Y = X
    
    #dt = 10 
    nblocks = max(1, (NT-delta)//nt)
    nt = min(nt, NT-delta)

    xxt = torch.zeros((NN, NN), device = device)
    xyt = torch.zeros((NN, NN), device = device)

    NT0 = 0
    for j in range(nblocks):
        x = X[:,j*nt:j*nt + nt-delta]
        y = Y[:,j*nt+delta:j*nt + nt]
        if valid_inds is not None:
            vinds = np.isin(np.arange(j*nt, j*nt + nt-delta), valid_inds)
            x = x[:, vinds]
            y = y[:, vinds]
            nt0 = x.shape[1]
        else:
            nt0 = nt
        
        if nt0 > 50:
            xxt += (x @ x.T)/NT * nt0
            xyt += (x @ y.T)/NT * nt0
            NT0 += nt0

    xxt *= NT / NT0
    xyt *= NT / NT0

    teye = torch.eye(NN, device = device)
    A = torch.linalg.solve(xxt + lam * teye, xyt).T
    return A


def dmd(sp, lam = .01, delta = 1, nt=10000, n_comps=1000, inds=None,
        device = torch.device('cuda')):
    
    X = torch.from_numpy(sp).to(device)
    
    if X.shape[0] < X.shape[1]:
        cov = (X @ X.T)/X.shape[1]
        e, u = torch.linalg.eigh(cov @ cov.T)
    else:
        cov = (X.T @ X)/X.shape[1]
        e, u = torch.linalg.eigh(cov.T @ cov)

    if n_comps < X.shape[0]:
        if X.shape[0] < X.shape[1]:
            X = u.T @ X
        else:
            X = (((u.T @ X.T)**2).sum(axis=1)**0.5).unsqueeze(1) * u.T
        X = X[-n_comps:]  

    valid_inds = inds[np.isin(inds + delta, inds)] if inds is not None else None

    At = dynamics_lag(X, delta = delta, lam = lam, nt=nt, valid_inds=valid_inds, device = device)
    e, v = torch.linalg.eig(At)
    At = At.cpu().numpy()
    e = e.cpu().numpy()
    v = v.cpu().numpy()
    isort = e.real.argsort()[::-1]
    e = e[isort]
    v = v[:, isort]

    return At, e, v


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
