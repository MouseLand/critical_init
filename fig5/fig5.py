import torch 
import numpy as np 
from tqdm import trange
device = torch.device('cuda')

import matplotlib.pyplot as plt
# plotting function


def plot_with_time(cmap, config, dat, grid, dt = 0.002, k = 0, label = False):
    ax = plt.subplot(grid[3:, k])
    
    time_pt = dat['wm_isi'] * dt
    mu = dat['acc'].mean(0)
    sd = dat['acc'].std(0)

    for j in range(3):
        plt.plot(time_pt, mu[:,j], '-', lw= 2, color = cmap(j))
        plt.fill_between(time_pt, mu[:,j] - sd[:,j], mu[:,j] + sd[:,j], alpha=0.2, color = cmap(j))

    plt.title('fixed time-lag')
    plt.xlabel('time (sec)')
    plt.ylabel('accuracy')

    if label:
        for j in range(3):
            plt.text(.95, .4-j*.1, config['names'][j], transform = ax.transAxes, ha = 'right', color = cmap(j))

    ax = plt.subplot(grid[3:, k+1])
    time_pt = dat['wm_isi'] * dt
    mu = dat['acc_shuff'].mean(0)
    sd = dat['acc_shuff'].std(0)

    for j in range(3):
        plt.plot(time_pt, mu[:,j], '-', lw= 2, color = cmap(j))
        plt.fill_between(time_pt, mu[:,j] - sd[:,j], mu[:,j] + sd[:,j], alpha=0.2, color = cmap(j))

    plt.title('time-independent')
    plt.xlabel('time (sec, up to)')
    plt.ylabel('accuracy')

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



def simulate_random(inputs, tinputs, toutputs, A = None, relu = False, enorm = None,
                    nonsym=0, distribution="uniform", T=60000, gains = 1,
                    tpad=4000, dt=2, tau=50, tbin=23, device=torch.device("cuda"), emax = 0.998):
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
    if device.type == "cuda":
        torch.cuda.empty_cache()
    nstim, nn, nd = inputs.shape

    if A is None:        
        A = random_connectivity(nn=nn, nonsym=nonsym, distribution=distribution, 
                                device=device)    
        if enorm is None:                    
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
    out = torch.zeros((len(toutputs), nstim, nn, nd), device=device, requires_grad = False) 
    Xi = torch.zeros((nn, nd, T//tbin), device=device)
    kk = 0

    #print(dt)
    #loss = 0
    for t in trange(T+tpad):
        eps = .1 * (tau/50 * 5/dt)**.5 * torch.randn((nn, nd), device=device)

        
        if kk < len(tinputs) and t>tpad + tinputs[kk]-25 and t<=tpad + tinputs[kk]:
            eps = eps + inputs[kk] * (5/dt) /25

        X = X + dt/tau * (-X + gains * (A @ X) + eps)
        if relu:
            X = torch.relu(X)
            
        if t >= tpad and (t-tpad)//tbin < Xi.shape[-1]:
            Xi[:, :, (t-tpad)//tbin] += X

        if kk < len(tinputs):
            for j in range(len(toutputs)):
                tmax = toutputs[j,kk]+tpad

                if t <= tmax and t > tmax-50:# np.isin(t, tpad+toutputs):
                    #kk = (toutputs+tpad==t).nonzero()[0]
                    out[j, kk] = out[j, kk] + X # Xi[:, :, (t-tpad)//tbin-1]
                    #if t==tmax:
                        #print((W@X).std().item())

                        #loss = loss + ((W @ X - inputs[kk])**2).mean()
            
            if t==tmax:
                kk+=1
        
    Xi /= tbin
    #Xi = Xi.reshape(nn, -1)

    return A, Xi, out

def zscore(X, axis = -1, eps = 1e-3, itrain = None):
    if itrain is None:        
        X = X - X.mean(axis=axis,keepdims=True)
        X = X/ (eps + (X**2).mean(axis=axis,keepdims=True)**.5)
    else:
        X = X - X[:,itrain].mean(axis=axis,keepdims=True)
        X = X/ (eps + (X[:,itrain]**2).mean(axis=axis,keepdims=True)**.5)
    return X

def decode_nearest(out, labels):
    nstim, NN, nd = out.shape 

    xout = torch.permute(out, (0,2,1)).reshape((-1, NN))

    nmax = int((labels.max()+1).item())
    x0 = torch.zeros((nmax, NN), device = device)
    x1 = torch.zeros((nmax, NN), device = device)
    for j in range(nmax):
        ix = (labels.flatten()==j).nonzero()
        x0[j,:] = xout[ix[0]]
        x1[j,:] = xout[ix[1]]

    cc = 2 * x0 @ x1.T - (x0**2).sum(-1).unsqueeze(-1) - (x1**2).sum(-1)
    imax = torch.max(cc, axis=-1)[1].cpu().numpy()
    correct = (imax==np.arange(len(imax)))

    return correct, imax

def decode_kmeans(out, labels, nfolds = 2):
    nstim, NN, nd = out.shape 
    xout = torch.permute(out, (0,2,1)).reshape((-1, NN))

    xout_z = zscore(xout, axis=-1)
    nmax = int((labels.max()+1).item())

    nstim_all = xout_z.shape[0]
    correct = np.zeros(nstim_all,)
    iperm = np.random.permutation(nstim_all)
    for k in range(nfolds):
        itest = iperm%nfolds==k
        itrain = ~itest 

        lbl_mat = (labels[itrain].unsqueeze(-1)==torch.arange(nmax, device = device)).float()
        xyt = xout_z[itrain].T @ lbl_mat
        xyt /= (1e-4 + lbl_mat.sum(0))
        xyt = xyt.T

        cc = 2 * xyt @ xout_z.T - (xyt**2).sum(-1).unsqueeze(-1) - (xout_z**2).sum(-1)
        label_pred = torch.max(cc, axis=0)[1]
        #label_pred = torch.min(((xout_z.unsqueeze(-1) - xyt)**2).mean(-2), axis=-1)[1]

        correct[itest] =  (label_pred[itest] == labels[itest]).cpu().numpy()

    return correct

def decode(out, inputs, nfolds = 2, lam = 10, nearest = True):
    nstim, NN, nd = out.shape 
    xout = torch.permute(out, (0,2,1)).reshape((-1, NN))
    xin = torch.permute(inputs, (0,2,1)).reshape((-1, NN))

    xin_z  = zscore(xin, axis=-1)
    xout_z = zscore(xout, axis=-1)

    #cc = xin_z @ xout_z.T
    #imax = torch.max(cc, axis=-1)[1].cpu().numpy()
    #(imax==np.arange(nstim*nd)).mean()

    nstim_all = xin_z.shape[0]
    correct = np.zeros(nstim_all,)
    iperm = np.random.permutation(nstim_all)
    for k in range(nfolds):
        itest = iperm%nfolds==k
        itrain = ~itest 

        xxt = xout_z[itrain].T @ xout_z[itrain]/ itrain.sum() + lam * torch.eye(NN,device=device)
        #xxt = torch.eye(NN,device=device)
        xyt = xout_z[itrain].T @ xin_z[itrain] / itrain.sum()

        B = torch.linalg.solve(xxt, xyt)

        ypred = xout_z @ B

        if nearest:
            x0 = ypred[itest]
            x1 = xin_z[itest]
            cc = 2 * x0 @ x1.T - (x0**2).sum(-1).unsqueeze(-1) - (x1**2).sum(-1)            
        else:
            cc = ypred[itest] @ xout_z[itest].T       

        imax = torch.max(cc, axis=-1)[1].cpu().numpy()
        correct[itest] = (imax==np.arange(len(imax)))

    #print(correct.mean())
    return correct

def random_inputs_2x(nstim, NN, nd):
    inputs = torch.randn((nstim//2, NN, nd), device = device)
    labels = torch.arange(nd*nstim//2, device = device).reshape((nstim//2,nd))

    inputs = torch.tile(inputs, (2,1,1))
    labels = torch.tile(labels, (2,1))

    return inputs, labels


def run_zeroshot(config, nstim, tisi, wm_isi, mode = 'independent',
                 nd = 40, NN = 10000, niter = 10, magi = 2.5, tau = 20, dt = 2):

    gains = config['gains']
    nonsym = config['nonsym']
    relu = config['relu']
    distribution = config['distribution']


    nreps = len(wm_isi)

    acc       = np.nan * np.ones((niter, nreps,len(gains)))
    acc_shuff = np.nan * np.ones((niter, nreps,len(gains)))

    tinputs  = np.arange(0, nstim*tisi, tisi)
    Tmax = tinputs[-1] + 1000
    toutputs = tinputs + wm_isi[:,np.newaxis] 


    for j in range(niter):
        for k in range(3):         
            #inputs = torch.randn((nstim, NN, nd), device = device)    
            i_inputs = torch.randn((nstim, 100, nd), device = device)
            wrot = torch.randn((NN, 100), device = device)/100**.5    
            inputs = torch.einsum('slr, nl->snr', i_inputs, wrot)    
            
            try:
                A = None
                if mode == 'aligned':
                    A, Xi, out = simulate_random(0. * inputs, tinputs, toutputs, A = None, gains = gains[k], 
                                             relu = relu[k],dt = dt, tau = tau,  T = Tmax, nonsym = nonsym[k], 
                                             distribution = distribution[k])                
                    Xi = Xi.reshape((NN, -1))
                    evals0, evecs0  = torch.linalg.eigh((Xi @ Xi.T)/Xi.shape[1])        
                    wrot = evecs0[:,-100:] * 10
                    inputs = torch.einsum('slr, nl->snr', i_inputs, wrot)
                    
                A, Xi, out = simulate_random(magi * inputs, tinputs, toutputs, A = A, gains = gains[k], 
                                             relu = relu[k],dt = dt, tau = tau,  T = Tmax, nonsym = nonsym[k], 
                                             distribution = distribution[k])
                acc[j,:,k]       = get_acc(out, inputs)           
                acc_shuff[j,:,k] = get_acc_shuffle(out, inputs) 
            except Exception as e:        
                print(f"An unexpected error occurred: {e}")

        print(acc[j].mean(0), acc_shuff[j].mean(0))

    return acc, acc_shuff

def run_multishot(config, nstim, tisi, wm_isi, mode = 'independent',
                 nd = 40, NN = 10000, niter = 10, magi = 2.5, tau = 20, dt = 2):

    gains = config['gains']
    nonsym = config['nonsym']
    relu = config['relu']
    distribution = config['distribution']


    nreps = len(wm_isi)

    acc       = np.nan * np.ones((niter, nreps,len(gains)))
    acc_shuff = np.nan * np.ones((niter, nreps,len(gains)))

    tinputs  = np.arange(0, nstim*tisi, tisi)
    Tmax = tinputs[-1] + 1000
    toutputs = tinputs + wm_isi[:,np.newaxis] 


    for j in range(niter):
        for k in range(3):
            inputs, labels = random_inputs_reps(nstim, NN, nd, nobj=2)         
            try:
                A = None
                if mode == 'aligned':
                    A, Xi, out = simulate_random(0. * inputs, tinputs, toutputs, A = None, gains = gains[k], 
                                             relu = relu[k],dt = dt, tau = tau,  T = Tmax, nonsym = nonsym[k], 
                                             distribution = distribution[k])                
                    Xi = Xi.reshape((NN, -1))
                    evals0, evecs0  = torch.linalg.eigh((Xi @ Xi.T)/Xi.shape[1])        
                    wrot = evecs0[:,-100:] * 10
                    inputs, labels = random_inputs_reps(nstim, NN, nd, nobj=2, wrot = wrot)
                    
                A, Xi, out = simulate_random(magi * inputs, tinputs, toutputs, A = A, gains = gains[k], 
                                             relu = relu[k],dt = dt, tau = tau,  T = Tmax, nonsym = nonsym[k], 
                                             distribution = distribution[k])
                
                
                acc[j,:,k]       = get_acc_kmeans(out, labels)
                acc_shuff[j,:,k] = get_acc_kmeans_shuffle(out, labels)
            except Exception as e:        
                print(f"An unexpected error occurred: {e}")

        print(acc[j].mean(0), acc_shuff[j].mean(0))

    return acc, acc_shuff

def random_inputs_reps(nstim, NN, nd, nobj=100, wrot = None):    
    if wrot is None:
        input_objects = torch.randn((nobj, NN), device = device)
    else:
        i_inputs = torch.randn((nobj, wrot.shape[-1]), device = device)
        input_objects = torch.einsum('sl, nl->sn', i_inputs, wrot)

    irand = np.random.randint(nobj, size=(nstim*nd, ))
    #labels = torch.from_numpy(2 * (irand==1) - 1).to(device).float()
    labels = torch.from_numpy(irand).to(device)
    inputs = input_objects[irand].reshape((nstim, nd, NN)).permute((0,2,1))
    return inputs, labels

def get_acc_kmeans_shuffle(out, labels):
    acc = np.zeros((len(out),))
    for j in range(len(out)):
        if j>0:
            out[:j] = time_shuffle(out[:j])
        correct = decode_kmeans(out[0], labels, nfolds = 2)        
        acc[j] = correct.mean()
    return acc 


def get_acc_kmeans(out, labels):
    acc = np.zeros((len(out),))
    for j in range(len(out)):
        correct = decode_kmeans(out[j], labels, nfolds = 2)        
        acc[j] = correct.mean()
    return acc 

def get_acc_shuffle(out, inputs, lam = 10):
    acc = np.zeros((len(out),))
    for j in range(len(out)):
        if j>0:
            out[:j] = time_shuffle(out[:j])
        correct = decode(out[0], inputs, nfolds = 2, lam = lam, nearest = True)    
        acc[j] = correct.mean()
    return acc 


def get_acc(out, inputs, lam = 10):
    acc = np.zeros((len(out),))
    for j in range(len(out)):
        correct = decode(out[j], inputs, nfolds = 2, lam = lam, nearest = True)    
        acc[j] = correct.mean()
    return acc 

def time_shuffle(out):
    for j in range(out.shape[1]):
        for k in range(out.shape[-1]):
            irand = torch.randperm(out.shape[0], device = device)
            out[:,j,:,k] = out[irand,j,:,k]
    return out

