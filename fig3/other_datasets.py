from rastermap import Rastermap
from pynwb import NWBHDF5IO
from pathlib import Path
import numpy as np
import lyapun 
from natsort import natsorted
import matplotlib.pyplot as plt 
from nlb_tools.nwb_interface import NWBDataset
from mat73 import loadmat
from scipy.sparse import csr_array
from scipy.interpolate import interp1d
import requests 
import json
from fig_utils import *

def plitt_hippo(root):
    root = Path(root)
    f = root / 'sub-R1_ses-20190209T210000_obj-bxnk6r_behavior+ophys.nwb'
    with NWBHDF5IO(f, 'r') as io:
        nwbfile = io.read()
        # if 'ophys' not in nwbfile.processing:
        #     continue
        # if 'morph' not in nwbfile.stimulus:
        #     continue
        morph = np.array(nwbfile.stimulus['morph'].data[:])
        nb, bb = np.histogram(morph, np.arange(-0.5, 1.5, 0.25))
        # if len(np.unique(morph)) > 3:
        #     continue
        # else:
        print(np.unique(morph))
        spks = np.array(nwbfile.processing['ophys'].data_interfaces['Fluorescence'].roi_response_series['Deconvolved'].data).T
        print(f.stem)
        spks_z = spks.copy()
        spks_z_std = spks_z.std(axis=1, keepdims=True)
        valid = spks_z_std[:,0] > 1e-3
        spks_z = spks_z[valid]
        print(spks_z.shape)

        spks_z -= spks_z.mean(axis=1, keepdims=True)
        spks_z /= spks_z.std(axis=1, keepdims=True)
        
        delta = 30
        At, evals, evecs = lyapun.dmd(spks_z, lam=0.1, delta=delta, nt=4000)
        
        e = evals.copy() #/ evals.real[0] * 0.999
        ix = np.abs(e)>.25
        iang = np.angle(e[ix]) / (2*np.pi)
        iabs = -np.log10(np.abs(e[ix]))    
        irot = iang/iabs
        ixx = e[ix].imag>=0
        mu = irot[ixx].mean()
        sd = irot[ixx].std()
        m = np.percentile(irot[ixx], [5, 25, 75, 95])
        print(ix.sum(), mu, sd, m)
        
        morph_ex = morph.copy()
        corr_starts_ex = np.nonzero(nwbfile.processing['behavior'].data_interfaces['BehavioralTimeSeries']['tstart'].data[:])[0]
        speed_ex = nwbfile.processing['behavior'].data_interfaces['BehavioralTimeSeries']['speed'].data[:]
        pos_ex = nwbfile.processing['behavior'].data_interfaces['BehavioralTimeSeries']['pos'].data[:]
        rzone_ex = nwbfile.processing['behavior'].data_interfaces['BehavioralTimeSeries']['rzone'].data[:]
        lick_ex = nwbfile.processing['behavior'].data_interfaces['BehavioralTimeSeries']['lick'].data[:]
        time_bin = 10
        model = Rastermap(n_clusters=100, n_PCs=128, time_lag_window=10, time_bin=10,
                                    locality=0.9, mean_time=False, normalize=False).fit(spks_z, 
                                                                                        compute_X_embedding=True)
        Xemb = spks_z[model.isort[:(spks_z.shape[0]//10) * 10]].reshape(-1, 10, spks_z.shape[1]).mean(axis=1)
        Xemb -= Xemb.mean(axis=1, keepdims=True)
        Xemb /= Xemb.std(axis=1, keepdims=True)
        evals_ex = evals.copy()

    np.save('../results/plitt_results.npy', {'fname': str(f), 'evals': evals_ex, 'shape': spks_z.shape, 
                                'Xemb': Xemb, 'morph_ex': morph_ex, 'corr_starts_ex': corr_starts_ex,
                                'speed_ex': speed_ex, 'pos_ex': pos_ex, 'rzone_ex': rzone_ex, 'lick_ex': lick_ex})
   
def grosmark_hippo(root):
    # download from crcns
    # https://crcns.org/data-sets/hc/hc-11/about-hc-11
    f = loadmat(Path(root) / 'Achilles_10252013_sessInfo.mat')

    
    st = np.array(f["sessInfo"]["Spikes"]["SpikeTimes"])
    clu = np.array(f["sessInfo"]["Spikes"]["SpikeIDs"])
    cells, clu = np.unique(clu, return_inverse=True)
    bin_sec = 1./15
    spks = csr_array((np.ones(len(st), "uint8"), 
                    (clu, np.floor(st / bin_sec).astype("int"))))

    ## load location info ## all in maze
    locations_2d = np.array(f["sessInfo"]["Position"]["TwoDLocation"])
    locations = np.array(f["sessInfo"]["Position"]["OneDLocation"])
    nnan = np.nonzero(~np.isnan(locations))[0]
    inan = np.nonzero(np.isnan(locations))[0]
    fi = interp1d(nnan, locations[nnan], 
                    bounds_error=False)#, kind="nearest")
    locations_raw = locations.copy()
    locations[inan] = fi(inan)
    for j in range(2):
        nnan = np.nonzero(~np.isnan(locations_2d[:,j]))[0]
        inan = np.nonzero(np.isnan(locations_2d[:,j]))[0]
        fi = interp1d(nnan, locations_2d[nnan,j], bounds_error=False)#, kind="nearest")
        locations_2d[inan,j] = fi(inan)
    locations_2d[locations_2d[:,1] > 0.1, 1] = 0 # some weirdness at the beginning of the session
    locations_times = np.array(f["sessInfo"]["Position"]["TimeStamps"]).flatten()

    keys = list(f["sessInfo"]["Epochs"].keys())
    epochs = []
    for key in keys:
        epochs.append(np.array(f["sessInfo"]["Epochs"][key]))

    maze_epoch = epochs[keys.index("MazeEpoch")].flatten()
    maze_bins = np.floor(maze_epoch / bin_sec).astype("int")
    spks_maze = spks[:, maze_bins[0] : maze_bins[1]].todense().astype("float32")

    # locations in spks time frame
    locations_bins = np.floor(locations_times/bin_sec).astype("int") - maze_bins[0]
    locations_vec = np.nan * np.zeros(spks_maze.shape[1])
    locations_raw_vec = np.nan * np.zeros(spks_maze.shape[1])
    locations_2d_vec = np.nan * np.zeros((spks_maze.shape[1],2))
    locations_vec[locations_bins] = locations
    locations_raw_vec[locations_bins] = locations_raw
    locations_2d_vec[locations_bins] = locations_2d

    spks_z = spks_maze.copy()
    spks_z -= spks_z.mean(axis=1, keepdims=True)
    spks_z /= spks_z.std(axis=1, keepdims=True)

    delta = 30
    At, evals, evecs = lyapun.dmd(spks_z, lam=0.1, delta=delta, nt=5000)
    print(evals.real[0])

    model = Rastermap(n_clusters=None, n_PCs=64, time_lag_window=10, time_bin=5,
                locality=0.1, mean_time=False, normalize=True).fit(spks_z, 
                                                                    compute_X_embedding=False)


    np.save('../results/achilles_results.npy', {'evals': evals, 'Xemb': spks_z[model.isort], 
                                    'locations_2d_vec': locations_2d_vec})
    


def zhong_visual(root='/media/carsen/disk2/zhong-et-al-2025/'):
    root = Path(root)
    ### downloading specific data (after learning sessions)
    Item_ID = 28811129
    #Set the base URL
    BASE_URL = 'https://api.figshare.com/v2'
    r = requests.get(BASE_URL + '/articles/' + str(Item_ID))
    file_metadata = json.loads(r.text)
    file_info = []
    for j in file_metadata['files']: #add the item id to each file record- this is used later to name a folder to save the file to
        j['item_id'] = Item_ID
        file_info.append(j) #Add the file metadata to the list

    if not (root / 'Imaging_Exp_info.npy').exists():
        # download exp info
        response = requests.get(BASE_URL + '/file/download/54183854')
        with open(root / 'Imaging_Exp_info.npy', 'wb') as f:
            f.write(response.content)

        response = requests.get(BASE_URL + '/file/download/54183860')
        with open(root / 'Beh_sup_train1_after_learning.npy', 'wb') as f:
            f.write(response.content)

    beh = np.load(root / 'Beh_sup_train1_after_learning.npy', allow_pickle=True).item()
    exp_info = np.load(root / 'Imaging_Exp_info.npy', allow_pickle=True).item()
    dbs = list(exp_info['sup_train1_after_learning'])
    db = dbs[0]
    fstr = f"{db['mname']}_{db['datexp']}_{db['blk']}"
    fname = f'{fstr}_neural_data.npy'
    for k in file_info:
        if k['name'] == fname:
            if not (root / k['name']).exists():
                response = requests.get(BASE_URL + f"/file/download/{k['id']}")
                with open(root / fname, 'wb') as f:
                    f.write(response.content)
    f = root / fname
    beh = beh[fstr]
    
    dat = np.load(f, allow_pickle=True).item()
    spks_z = np.vstack([sp for sp in dat['spks']])
    if np.abs(spks_z[:,0]).sum() == 0:
        spks_z = spks_z[:,1:]
    spks_z_std = spks_z.std(axis=1, keepdims=True)
    valid = spks_z_std[:,0] > 1e-3
    spks_z = spks_z[valid]
    spks_z -= spks_z.mean(axis=1, keepdims=True)
    spks_z /= spks_z_std[valid]
    print(spks_z.shape)

    delta = 6
    At, evals, evecs = lyapun.dmd(spks_z, lam=0.1, delta=delta, nt=4000) # some issues with large matrices w/ eigh (cut at 25000)
    
    e = evals.copy() #/ evals.real[0] * 0.999
    ix = np.abs(e)>.25
    iang = np.angle(e[ix]) / (2*np.pi)
    iabs = -np.log10(np.abs(e[ix]))    
    irot = iang/iabs
    ixx = e[ix].imag>=0
    mu = irot[ixx].mean()
    sd = irot[ixx].std()
    m = np.percentile(irot[ixx], [5, 25, 75, 95])
    print(ix.sum(), mu, sd, m)

    model = Rastermap(n_clusters=100, n_PCs=128, normalize=False, mean_time=False,
                    time_lag_window=10, locality=0.9, bin_size=100).fit(spks_z)
    bin_size = 100
    nn = spks_z.shape[0]
    Xemb = spks_z[model.isort[:(nn//bin_size)*bin_size]].reshape(nn//bin_size, bin_size, -1).mean(axis=1)
    Xemb -= Xemb.mean(axis=1, keepdims=True)
    Xemb /= Xemb.std(axis=1, keepdims=True)
        
    np.save('../results/zhong_results.npy', {'fname': str(f), 'evals': evals, 'shape': spks_z.shape,
                                'Xemb': Xemb, 'beh': beh})

 
def area2_bump():
    # area2_bump
    dataset = NWBDataset("000127/sub-Han/", "*train", split_heldout=False)

    go_cue = np.round(dataset.trial_info['move_onset_time'].to_numpy().astype('float') / 1e6).astype('int')
    igood = go_cue > 0
    go_cue = go_cue[igood]
    spks_1ms = dataset.data['spikes'].to_numpy().T

    inds = (go_cue[:,np.newaxis] + np.arange(-400, 800)).astype('int')

    spks = spks_1ms[:, inds]

    #plt.plot(spks.mean(axis=(0, 1)))

    bin_size = 20
    spks = spks[:, :, :(spks.shape[2]//bin_size)*bin_size].reshape(spks.shape[0], 
                                                                spks.shape[1], spks.shape[2]//bin_size, bin_size).sum(axis=-1)
    spks = spks.astype('float32') / bin_size

    # All 16 conditions, in the format (ctr_hold_bump, cond_dir)
    conds = [(False, 0.0), (False, 45.0), (False, 90.0), (False, 135.0),
                        (False, 180.0), (False, 225.0), (False, 270.0), (False, 315.0),
                        (True, 0.0), (True, 45.0), (True, 90.0), (True, 135.0),
                        (True, 180.0), (True, 225.0), (True, 270.0), (True, 315.0)]

    nconds = len(conds)
    psth = np.zeros((spks.shape[0], nconds, spks.shape[2]))

    for i in range(nconds):
        icond = (np.all(dataset.trial_info[['ctr_hold_bump', 'cond_dir']] == conds[i], axis=1))
        psth[:,i] = spks[:, icond[igood]].mean(axis=1)

    #plt.plot(psth.mean(axis=(0, 1)))

    #spks = spks[:, ~np.isnan(spks[0])]
    spks = psth.reshape(psth.shape[0], -1).copy()
    #spks = psth[:,0].copy()
    spks = spks[spks.sum(axis=1) > 1]
    print(spks.shape)
    spks_z = (spks.copy() - spks.mean(axis=1, keepdims=True)) / spks.std(axis=1, keepdims=True)

    delta = 10
    At, evals, evecs = lyapun.dmd(spks_z, lam=0.1, delta=delta, nt=psth.shape[-1])
    # At, evals, evecs = dmd(spks_z, lam=0.1, delta=delta, nt=psth.shape[-1])
    print(evals.real[0])

    model = Rastermap(n_clusters=None, n_PCs=48, time_lag_window=5, bin_size=1,
                locality=0.5, mean_time=False, normalize=True).fit(spks_z, 
                                                                    compute_X_embedding=False)


    np.save('../results/area2_bump_results.npy', {'evals': evals, 'Xemb': spks_z[model.isort], 'conds': conds,
                                    'nt': psth.shape[-1], 'tstart': 400//bin_size})



def mc_rtt():
    dataset = NWBDataset("000129/sub-Indy", "*train", split_heldout=False)
    spks_1ms = dataset.data['spikes'].to_numpy().T
    NT = spks_1ms.shape[1]

    target_pos = dataset.data['target_pos'].to_numpy()
    finger_vel = dataset.data['finger_vel'].to_numpy()
    finger_speed = (finger_vel**2).sum(axis=1)**0.5

    go_cue = np.nonzero((np.diff(target_pos, axis=0) != 0).sum(axis=1) )[0] + 1
    go_cue = go_cue[:-1][np.diff(go_cue) > 1]
    go_cue = go_cue[~np.isnan(target_pos[go_cue-1, 0])]
    move_onset = np.zeros(0, dtype='int')
    igood = np.ones(len(go_cue), dtype='bool')
    for i, (s, e) in enumerate(zip(go_cue, np.hstack((go_cue[1:], NT)))):
        #onset = np.nonzero((speed_onset >= s) * (speed_onset < e))[0]
        onset = np.nonzero(finger_speed[s:e] > 20)[0]
        if len(onset) > 0:
            move_onset = np.hstack((move_onset, s + onset[0]))
        else:
            igood[i] = False
    go_cue = go_cue[igood]

    init = target_pos[go_cue - 1]
    targs = target_pos[go_cue]
    angle = np.arctan2(targs[:,1] - init[:,1], targs[:,0] - init[:,0]) / np.pi * 180
    angle[angle >= 180] = 180 - 1e-3
    angle_bins = np.floor((angle + 180) / 15).astype('int')
    nconds = angle_bins.max() + 1

    inds = (move_onset[:,np.newaxis] + np.arange(-400, 800)).astype('int')

    spks = spks_1ms[:, inds]
    igood = np.isnan(spks).sum(axis=(0, -1)) == 0
    spks = spks[:, igood]
    angle_bins = angle_bins[igood]

    bin_size = 20
    spks = spks[:, :, :(spks.shape[2]//bin_size)*bin_size].reshape(spks.shape[0], 
                                                                spks.shape[1], spks.shape[2]//bin_size, bin_size).sum(axis=-1)
    spks = spks.astype('float32') / bin_size

    psth = np.zeros((spks.shape[0], nconds, spks.shape[2]))

    for i in range(nconds):
        psth[:,i] = spks[:, angle_bins == i].mean(axis=1)

    spks = psth.reshape(psth.shape[0], -1).copy()
    spks = spks[spks.sum(axis=1) > 1]
    print(spks.shape)
    spks_z = (spks.copy() - spks.mean(axis=1, keepdims=True)) / spks.std(axis=1, keepdims=True)

    delta = 10
    At, evals, evecs = lyapun.dmd(spks_z, lam=0.1, delta=delta, nt=psth.shape[-1])
    print(evals.real[0])

    model = Rastermap(n_clusters=None, n_PCs=48, time_lag_window=5, bin_size=1,
                locality=0.5, mean_time=False, normalize=True).fit(spks_z, 
                                                                    compute_X_embedding=False)


    np.save('../results/mc_rtt_results.npy', {'evals': evals, 'Xemb': spks_z[model.isort], 'conds': np.arange(-180, 180, 15),
                                    'nt': psth.shape[-1], 'tstart': 400//bin_size})

def mc_maze():
    dataset = NWBDataset(f"000128/sub-Jenkins/", "*train", split_heldout=False)

    bin_size = 20
    cursor_pos = dataset.data['cursor_pos'].to_numpy()

    target_pos = dataset.trial_info['target_pos'].to_numpy()
    move_onset = dataset.trial_info['move_onset_time'].to_numpy().astype('int') // 1e6
    go_cue = dataset.trial_info['go_cue_time'].to_numpy().astype('int') // 1e6
    start_time = dataset.trial_info['start_time'].to_numpy().astype('int') // 1e6
    go_cue = move_onset[:-2]
    end_time = dataset.trial_info['end_time'].to_numpy().astype('int') // 1e6

    go_cue = []
    igood = np.zeros(len(start_time), 'bool')
    for i, (start, end) in enumerate(zip(start_time[:-1], end_time[:-1])):
        start = int(start)
        end = int(end)
        moves = np.nonzero((np.diff(cursor_pos[start:end], axis=0)**2).sum(axis=1) > 0.05)[0]
        if len(moves) == 0:
            continue
        igood[i] = True
        go_cue.append(start + moves[0] + 0)
    go_cue = np.array(go_cue)

    spks_1ms = dataset.data['spikes'].to_numpy().T

    inds = (go_cue[:,np.newaxis] + np.arange(-400, 1000)).astype('int')
    print(inds[(inds <= start_time[igood,np.newaxis]) | (inds > end_time[igood,np.newaxis])].shape)

    spks = spks_1ms[:, inds].astype('float32')
    spks = spks[:, :, :(inds.shape[1]//bin_size)*bin_size].reshape(spks_1ms.shape[0], inds.shape[0], inds.shape[1]//bin_size, bin_size).mean(axis=-1)
    print(spks.shape)

    igood2 = np.isnan(spks[0]).sum(axis=-1) == 0
    spks = spks[:,igood2]
    igood[np.nonzero(igood)[0][~igood2]] = False

    # Find unique conditions
    conds = dataset.trial_info.set_index(['trial_type', 'trial_version']).index.unique().tolist()
    trial_conds = dataset.trial_info.set_index(['trial_type', 'trial_version']).index
    trial_conds = trial_conds[igood]
    target_pos = target_pos[igood]

    psth = np.zeros((spks.shape[0], len(conds), spks.shape[2]), 'float32')
    targets = []
    for i, cond in enumerate(conds):
        #print(i, (trial_conds == cond).sum())
        ic = trial_conds == cond 
        targets.append(target_pos[ic][0])
        psth[:,i] = spks[:, trial_conds == cond, :].mean(axis=1)

    spks = psth.reshape(psth.shape[0], -1).copy()
    spks_z = (spks.copy() - spks.mean(axis=1, keepdims=True)) / spks.std(axis=1, keepdims=True)

    delta = 10
    At, evals, evecs = lyapun.dmd(spks_z, lam=0.1, delta=delta, nt=psth.shape[-1])
    print(evals.real[0])
    
    model = Rastermap(n_clusters=30 if spks_z.shape[0] > 150 else None, n_PCs=48, time_lag_window=5, bin_size=1,
            locality=0.5, mean_time=False, normalize=True).fit(spks_z, 
                                                                compute_X_embedding=False)
    Xemb = spks_z[model.isort].copy()
    
    np.save('../results/mc_maze_results.npy', {'evals': evals, 'Xemb': Xemb, 'conds': conds,
                                    'nt': psth.shape[-1], 'shape': spks_z.shape, 
                                    'targets': targets, 'tstart': 400//bin_size})

def suppfig_rotate(dsets, fnames):
    fig = plt.figure(figsize=(14, 8.5), dpi=150)
    yratio = 14/8.5
    il = 0
    grid = plt.GridSpec(3, 5, wspace=0.4, hspace=0.35, figure=fig, 
                            bottom=0.05, top=0.95, left=0.02, right=0.98)
    cmap = plt.get_cmap('RdPu')(np.linspace(0.4, 1, 5))
    
    #    dcolors = ['b', cmap[-1], cmap[2], 'r', [1, 0.5, 0], [0.5, 0.5, 0]]
    dcolors = np.zeros((len(fnames), 3))

    xlim = {'achilles': [8000, 12000], 
            'plitt': [2000, 2600],
            'zhong': [8550, 8900],
            }

    titles = {'achilles': f'Rat CA1, linear track ({dsets["achilles"]["Xemb"].shape[0]:,} neurons)',
              'plitt': f'Mouse CA1, virtual reality ({dsets["plitt"]["shape"][0]:,} neurons)',
              'zhong': f'Mouse visual cortex, virtual reality ({dsets["zhong"]["shape"][0]:,} neurons)',
              'mc_maze': f'Macaque PMd + M1, center-out reach - PSTHs ({dsets["mc_maze"]["Xemb"].shape[0]:,} neurons)',
              'mc_rtt': f'Macaque M1, sequential reach - PSTHs ({dsets["mc_rtt"]["Xemb"].shape[0]:,} neurons)',
              'area2_bump': f'Macaque Area 2, center-out reach - PSTHs ({dsets["area2_bump"]["Xemb"].shape[0]:,} neurons)',
              }
    
    axsum = plt.subplot(grid[0, -1])
    pos = axsum.get_position().bounds
    axsum.set_position([pos[0]+0.06, pos[1], pos[2]-0.06, pos[3]])
    
    for d, fname in enumerate(fnames):
        dset = dsets[fname]
        tbin = 5 if fname=='plitt' else 1
        grid1 = matplotlib.gridspec.GridSpecFromSubplotSpec(1, 4, subplot_spec=grid[d//2, 2*(d%2) : 2*(d%2)+2],
                                                                wspace=0.7, hspace=0.)

        
        evals = dset['evals']
        Xemb = dset['Xemb']
        if tbin > 1:
            nt = Xemb.shape[-1]
            Xemb = Xemb[:, :(nt//tbin)*tbin].reshape(-1, nt//tbin, tbin).mean(axis=-1)


        grid2 = matplotlib.gridspec.GridSpecFromSubplotSpec(4, 1, subplot_spec=grid1[:, :-1],
                                                                wspace=0.7, hspace=0.1)
        
        ax = plt.subplot(grid2[1:])

        if fname=='zhong' or fname=='achilles' or fname=='plitt':
            fs = 3 if fname=='zhong' or fname=='plitt' else 15
            xmin = xlim[fname][0]
            xmax = xlim[fname][1]
            ax.imshow(Xemb[:, xmin:xmax], aspect="auto", vmin=0, vmax=1, cmap="gray_r")
            if fname=='zhong':
                beh = dset['beh']
                colors = {'leaf1': cmap[2], 'circle1': 'b'}
                run_speed = beh['RunFr']
                lick_times = np.round(beh['LickFr']).astype(int)
                reward_starts = np.round(beh['RewardFr']).astype(int)
                for i, (tstart, tend, sid) in enumerate(zip(beh['StartFr'], beh['GrayFr'], beh['TrialStim'])):
                    if tstart < xmax and (tstart >= xmin or tend > xmin):
                        tstart = int(max(xmin, tstart))
                        tend = int(min(xmax, tend))
                        ax.axvspan(tend - xmin, tstart - xmin, color=colors[sid], alpha=0.1, lw=0)
            elif fname=='plitt':
                run_speed = dset['speed_ex'][:(nt//tbin)*tbin].reshape(nt//tbin, tbin).mean(axis=-1)
                licks = dset['lick_ex'][:(nt//tbin)*tbin].reshape(nt//tbin, tbin).sum(axis=-1)
                lick_times = np.nonzero(licks)[0]
                pos_ex, morph_ex, rzone_ex = dset['pos_ex'], dset['morph_ex'], dset['rzone_ex'] 
                dark = ((pos_ex < 0) * (pos_ex > -10)).astype('int')
                dark_starts = np.nonzero(np.diff(dark) > 0)[0] + 1

                for i in range(len(dark_starts)-1):
                    dark_end = np.nonzero(dark[dark_starts[i]:]==0)[0][0] + dark_starts[i]
                    dark_start = dark_starts[i+1]
                    dark_start //= tbin
                    dark_end //= tbin
                    if dark_end < len(morph_ex):
                        if dark_start < xmax and (dark_start >= xmin or dark_end > xmin):
                            dend = max(xmin, dark_end)
                            dstart = min(xmax, dark_start)
                            ax.axvspan(dend - xmin, dstart-1 - xmin, 
                                color=cmap[int(4*morph_ex[dend*tbin+1])], alpha=0.1, lw=0)
                rdiff = np.nonzero(np.diff(rzone_ex) > 0)[0]
                reward_starts = rdiff[rzone_ex[rdiff] == 0] // tbin
            
            if fname=='plitt' or fname=='zhong':
                reward_starts = reward_starts[(reward_starts > xmin) * (reward_starts < xmax)]
                for rs in reward_starts:
                    ax.axvline(rs - xmin, color='g', ls='--', alpha=1, lw=2)
                ax.text(0.2, -0.1, 'reward', fontstyle='italic', color='g',
                        transform=ax.transAxes)
            T = xmax-xmin
            tbar = 10 * fs 
            tstr = '10 sec.'
        else:
            fs = 50
            nt, tstart, conds = dset['nt'], dset['tstart'], dset['conds']
            nn = Xemb.shape[0]
            Xt = Xemb.reshape(nn, -1, nt)
            ntrials = Xt.shape[1]
            nplt = 6
            if fname=='mc_maze':
                targets = dset['targets']
                nobarriers = np.array([c[1]==0 for c in conds])
                Xt = Xt[:, nobarriers]
                targets = np.array([t[0] for j,t in enumerate(targets) if nobarriers[j]])
                conds = np.array([c[0] for c in conds if c[1]==0])
                angles = np.arctan2(targets[:,1], targets[:,0]) #/ np.pi * 180
                isort = angles.argsort()[::-1]
                conds = conds[isort]
                angles = angles[isort]
                targets = targets[isort]
                Xt = Xt[:, isort]
                ntrials = Xt.shape[1]
                ip = np.arange(-np.pi, np.pi, np.pi/3)[::-1]
                jp = np.abs(angles[:, np.newaxis] - ip).argmin(axis=0)
                tpad = 10
            elif fname=='mc_rtt':
                angles = dset['conds'] / 180 * np.pi
                angles = angles[::-1]
                Xt = Xt[:, ::-1]
                jp = np.arange(0+ntrials//(nplt*2), ntrials, ntrials//nplt)
                tpad = 10
            else:
                angles = np.array([c[1] for c in dset['conds']]) / 180 * np.pi
                jp = np.arange(0, ntrials//2, ntrials//nplt)[:nplt//2]
                jp = np.hstack((jp, np.arange(ntrials//2, ntrials, ntrials//nplt)[:nplt//2]))
                tpad = 20
            Xplt = np.zeros((nn, nplt*(nt + tpad)), 'float32')
            for j in range(nplt):
                Xplt[:, j*(nt+tpad):j*(nt+tpad)+nt] = Xt[:, jp[j]]
            ax.imshow(Xplt, aspect="auto", vmin=0, vmax=1, cmap="gray_r")
            for j in range(nplt):
                ax.axvline(tstart + j*(nt+tpad), color=[1,0,0], ls='--', alpha=1, lw=2)

            ax.text(0.2, -0.1, 'move onset', fontstyle='italic', color=[1,0,0],
                        transform=ax.transAxes)
            T = Xplt.shape[1]
            tbar = 1*fs
            tstr = '1 sec.'
        ax.set_xlim([0, T])
        ax.axis('off')

        axin = ax.inset_axes([0, -0.05, 1, 0.05])
        axin.plot([0, tbar], np.zeros(2), color='k')
        axin.set_xlim([0, T])
        axin.text(0, -0.1, tstr, ha='left', va='top')
        axin.axis('off')


        
        ax = plt.subplot(grid2[:1])
        if fname=='plitt' or fname=='zhong' or fname=='achilles':
            if fname=='plitt' or fname=='zhong':
                rs = run_speed[xmin:xmax]
                rs /= rs.max()
                rs /= 1.2 if fname=='plitt' else 1
                ax.fill_between(np.arange(0, xmax-xmin), y1=0, y2=run_speed[xmin:xmax], color=0.8*np.ones(3))
                lt = lick_times[(lick_times > xmin) * (lick_times < xmax)] - xmin
                ax.scatter(lt, -0.2 * np.ones(len(lt)), s=8, marker='|', lw=1, 
                           alpha=0.1 if fname=='plitt' else 0.5, color=[0,1,0])
                ax.text(0.8, 0.9, 'run speed', fontstyle='italic', color=0.8*np.ones(3),
                        transform=ax.transAxes)
                ax.text(1, -0.1, 'licks', fontstyle='italic', color=[0,1,0],
                        transform=ax.transAxes)                
                
            else:
                pos = dset['locations_2d_vec'].copy()
                pos -= pos[~np.isnan(pos[:,0]),0].min()
                pos /= pos[~np.isnan(pos[:,0]),0].max()
                pos -= 0.1
                ax.plot(pos[xmin:xmax,0], color=[0.3,0.6,1], lw=1)
                ax.plot(pos[xmin:xmax,1], color=[0.5,0.5,1], lw=1)
                ax.text(0.6, 0.8, 'x-pos', fontstyle='italic', color=[0.3,0.6,1],
                        transform=ax.transAxes)
                ax.text(0.6, 0.5, 'y-pos', fontstyle='italic', color=[0.5,0.5,1],
                        transform=ax.transAxes)

            ax.set_xlim([0, xmax-xmin])
            ax.set_ylim([-0.25, 1])
            
        else:#if fname=='mc_maze' or fname=='mc_rtt':
            wid = (nt+tpad)//2 - 15
            for j in range(nplt):
                color = 'k'
                if j > nplt//2-1 and fname=='area2_bump':
                    color = 0.5*np.ones(3)
                x0 = j*(nt+tpad) + wid + (nt-wid)/4
                y0 = wid
                dx = wid*np.cos(angles[jp[j]])
                dy = wid*np.sin(angles[jp[j]])
                if fname=='mc_maze' or fname=='area2_bump':
                    ax.scatter(x0, y0, marker='o', edgecolor=[1,0,0], 
                           lw=1, s=30, facecolor='none')
                else:
                    x0 -= dx/2
                    y0 -= dy/2
                ax.arrow(x0, y0, dx, dy, head_width=8, head_length=8, 
                            fc=color, ec=color, lw=2)
                #ax.plot([x0, x0+dx], [y0, y0+dy], color=color)
                #circle = plt.Circle((x0, y0), wid, edgecolor=color, facecolor='none', lw=1)
                #ax.add_patch(circle)
                ax.set_xlim([0, Xplt.shape[1]])
                ax.set_ylim([-13, 2*wid+13])
            if fname=='area2_bump':
                ax.text(0.44, -0.01, 'passive', fontstyle='italic', color=color,
                        transform=ax.transAxes, rotation=45)
                ax.text(-0.06, 0.04, 'active', fontstyle='italic', color='k',
                        transform=ax.transAxes, rotation=45)


        ax.axis('off')
        ax.set_title(titles[fname], color=dcolors[d], loc='left', y=1.1)

        transl = mtransforms.ScaledTranslation(-17 / 72, 6 / 72, fig.dpi_scale_trans)
        il = plot_label(ltr, il, ax, transl)

        ax = plt.subplot(grid1[-1:])
        pos = ax.get_position().bounds
        yh = pos[2]*3.3
        ax.set_position([pos[0]+0.01, pos[1]+(pos[3]-yh)/2, pos[2]*1.1, yh])
        ax.scatter(evals.real, evals.imag, s=3, color=dcolors[d])
        ax.set_ylim([-1, 1])
        ax.set_xlim([0, 1.01])
        ax.set_xticks([0, 0.5, 1])
        if d==0:
            ax.set_xlabel('real part')
            ax.set_ylabel('imaginary part')
            ax.set_title('Eigenvalues of\nDMD matrix', fontsize='medium', y=.92)

        ix = np.abs(evals)>.1
        iang = np.angle(evals[ix]) / (2*np.pi)
        iabs = -np.log10(np.abs(evals[ix]))    
        irot = iang/iabs
        ixx = evals[ix].imag>=0
        mu = irot[ixx].mean()
        sd = irot[ixx].std()
        m = np.percentile(irot[ixx], [5, 25, 50, 75, 95])
        print(ixx.sum())
        yy = -d
        axsum.plot([m[0], m[-1]], [yy, yy], color=0.4*np.ones(3))
        axsum.plot([m[1], m[-2]], [yy, yy], lw=4, color=0.4*np.ones(3))
        dy = 0.15
        axsum.plot([m[2], m[2]], [yy-dy, yy+dy], lw=1, color=dcolors[d])

        
    axsum.set_xlabel('rotations per \n10-fold attenuation')
    axsum.set_yticks(np.arange(-len(fnames)+1, 1))
    yticks = ['\n'.join(titles[fname].split(',')).split('(')[0] 
                           for fname in fnames[::-1]]
    yticks = [ytick.split(' -')[0] for ytick in yticks]
    axsum.set_yticklabels(yticks, fontsize='small')
    il = plot_label(ltr, il, axsum, mtransforms.ScaledTranslation(-100 / 72, 6 / 72, fig.dpi_scale_trans))

    return fig    