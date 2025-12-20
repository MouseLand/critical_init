from matplotlib.patches import Ellipse
from fig_utils import *
from scipy.stats import zscore
from powerlaw import fit_powerlaw_exp
from scipy.stats import ttest_rel
from matplotlib import gridspec
from scipy.interpolate import interp1d


def fig2(dat):
    areas_all = dat["areas_all"]
    evals_all = dat["evals_svca2_all"]
    evals_shuff_all = dat["evals_shuff_all"]
    evals_run = dat["evals_run_all"]

    fig = plt.figure(figsize=(14, 7), dpi=300)
    yratio = 14/7
    grid = plt.GridSpec(8, 7, wspace=0.4, hspace=0.2, figure=fig, 
                        bottom=0.04, top=0.94, left=0.02, right=1.)
    titles = ["cortical 2P imaging", "CA1 2P imaging", 
              "brainwide Neuropixels"]
    sneur = ["2,385 - 10,344 ROIs", "2,961 - 8,566 ROIs",  "1,716 - 2,914 units"]
    colors = dcolors[:3].copy()
    n_dset = len(areas_all)
    ids = np.zeros(n_dset, "int")
    for d in range(len(areas)):
        ids[np.array(areas_all)==areas[d]] = d
    ls = ["-", "-", "-", "--", "-."]
    print(areas_all, ids)
    dy = [0, 2, 1]
    alphas = np.zeros(n_dset)
    alphas_shuff = np.zeros(n_dset)
    ymax = 500
    norms = np.zeros(n_dset)

    if "imexs" in dat:
        # example images/traces (extracted from suite2p folders from example data)
        # (if available make example panels)
        imexs = dat["imexs"]
        masksexs = dat["masksexs"]    

        for d in range(3):
            ax = plt.subplot(grid[:2, d])
            pos = ax.get_position().bounds
            ax.set_position([pos[0]+0.04*d + 0.01, pos[1], pos[2]*(1.3 +0.4*(d<2)), pos[3]*1.])
            if d==2:
                im = plt.imread("allenprobes.png")
                #print(im.shape, imexs[1].shape)
                ax.imshow(im[58:-100])
            else:
                ax.imshow(imexs[d], vmin=0, vmax=0.9, 
                        cmap="gray", aspect=0.75/0.5 if d==1 else 1)
                masks0 = masksexs[d].copy()
                yo, xo = np.nonzero(masks0[:,:,-1]>0)
                masks0[yo,xo,-1] = 0.25
                ax.imshow(masks0, aspect=0.75/0.5 if d==1 else 1)
                if d==0:
                    ax.set_ylim([380, 380+90])
                    ax.set_xlim(50, 50 + 130)
                elif d==1:
                    #ax.set_ylim([350, 430])
                    #ax.set_xlim([100, 100 + 80*0.75/0.5])
                    ax.set_ylim([50, 50+90])
                    ax.set_xlim([250, 250+130*0.75/0.5])
            ax.set_title(titles[d], color=colors[d], fontstyle="italic", 
                         y=1.075, loc="left", x=-0.05)#, fontweight="bold")
            ax.text(0.5, 1.025, sneur[d], transform=ax.transAxes, 
                    fontsize="small", ha="center")
            ax.axis("off")
            il = d
            transl = mtransforms.ScaledTranslation(-25 / 72, 12 / 72, fig.dpi_scale_trans)
            il = plot_label(ltr, il, ax, transl)

    for j in range(2):
        Xexs = dat["Xexs"] if j==0 else dat["Xemb_ex"]
        Xexs = [Xexs[0], Xexs[1]] if j==0 else Xexs
        if j==0:
            titles = ['Rastermap of 2p data - V1', 'CA1']
            tmins = [5800, 42000//7+200] # 5000
            tlen = 22*60*3//7
        else:
            titles = ['Simulated 2p data - max eigenvalue = 0.998', 'max eigenvalue = 0.975']
            tmins = [5960*2, 0]
            tlen = 22*60*3//7
        for d in range(2):
            ax = plt.subplot(grid[2+3*j:5+3*j, 2*d:2*d+2])
            pos = ax.get_position().bounds
            ax.set_position([pos[0]-d*0.015, pos[1]-j*0.02, pos[2]+0.005, pos[3]*0.9])
            #Xe = zscore(Xexs[d][:, tmins[d]:tmins[d]+tlen].copy(), axis=1)
            im = ax.imshow(zscore(Xexs[d][:, tmins[d]:tmins[d]+tlen], axis=1), 
                        aspect="auto", cmap="gray_r", vmin=0., vmax=1.5)    
            ax.plot([0, 22*30/7], -Xexs[d].shape[0]*0.04*np.ones(2), color="k", lw=1.5)
            ax.plot(-0.025*tlen*np.ones(2), [0, 50], color="k", lw=1.5)
            if d==0:
                if j==0:
                    ax.text((22*30/2)/7, -50, "30 sec.", ha="center", va="center", fontsize="small")
                    ax.text(-0.05*tlen, 0, "1000 neurons", ha="center", va="bottom", fontsize="small", rotation=90)
                    cax = ax.inset_axes([0.85, -0.01, 0.15, 0.025])
                    cb = plt.colorbar(im, cax=cax, orientation="horizontal")
                    cb.set_ticks([0, 1.0])
                    cb.set_ticklabels(["0", "1.0"], fontsize="small")
                    ax.text(0.83, -0.005, "z-scored activity", fontsize="small",
                            ha="right", va="top", transform=ax.transAxes)
            ax.set_ylim([-Xexs[d].shape[0]*0.05, Xexs[d].shape[0]+0.5])
            ax.set_xlim([-0.028*tlen, tlen])
            ax.axis("off")
            ax.set_title(titles[d], fontsize='medium')
            if d==0:
                il = 3 + 3*j
                transl = mtransforms.ScaledTranslation(-10 / 72, 0 / 72, fig.dpi_scale_trans)
                il = plot_label(ltr, il, ax, transl)
    
    grid1 = gridspec.GridSpecFromSubplotSpec(3, 1, subplot_spec=grid[:5, 4], wspace=0., hspace=0.3)
    for d in range(3):
        ax = fig.add_subplot(grid1[d,0])
        pos = ax.get_position().bounds
        ax.set_position([pos[0]+0.01, pos[1]+(pos[3]-pos[2]*yratio)/2+0.01, 
                        pos[2]*0.8, pos[2]*yratio*0.8])
        pose = ax.get_position().bounds
        ix = np.nonzero(ids==d)[0]
        if d==0:
            ix = np.hstack((ix, np.nonzero(ids==4)[0])) 
            ix = np.hstack((ix, np.nonzero(ids==3)[0])) 
        lns = []
        for i in ix:
            ss = evals_all[i][:1000].copy()
            alphas[i], yp = fit_powerlaw_exp(evals_all[i], 
                                                        np.arange(10, ymax))
            ss /= yp[0]
            norms[i] = yp[0] #yp[10] * 10
            ax.loglog(np.arange(1, min(len(evals_all[i])+1, 1001)), 
                    ss, color=colors[d], lw=0.5, alpha=0.75, ls=ls[ids[i]])
            alphas_shuff[i] = fit_powerlaw_exp(evals_shuff_all[i], 
                                                        np.arange(10, ymax))[0]
            ln = ax.plot([], [], color=colors[d], lw=1, ls=ls[ids[i]])
            lns.append(ln)
        if d==0:
            ax.legend([lns[0][0], lns[-5][0], lns[-1][0]], ["V1", "sensori-\nmotor", "PPC"], 
                      frameon=False, loc="upper left", bbox_to_anchor=(0.3, 1.3), 
                      handlelength=1.2)

        ax.set_ylim(0.003, 3)
        ax.set_xlim(1, 1000)
        ax.set_xticks([1, 10, 100, 1000])
        ax.set_xticklabels(["1", "10", "100", "1,000"])
        ax.set_yticks([0.01, 0.1 ,1])
        ax.set_yticklabels(["0.01", "0.1", "1"])
        aexp =  [-0.69, -1.254]
        yc = 1
        ax.fill_between([1, 1000], [yc, yc * 1000**aexp[0]], [yc, yc * 1000**aexp[1]], 
                        color="k", lw=0, alpha=0.1)
        if d==0:
            il = 4
            transl_e = mtransforms.ScaledTranslation(-50 / 72, 12 / 72, fig.dpi_scale_trans)
            il = plot_label(ltr, il, ax, transl_e)
            ax.set_ylabel("normalized variance")
        
            axin = ax.inset_axes([1.2, 0.5, 0.7, 0.7])
            axin.fill_between([1, 1000], [yc, yc * 1000**aexp[0]], [yc, yc * 1000**aexp[1]], 
                        color="k", lw=0, alpha=0.1)
            axin.text(0.27, 0.37, "symmetric", transform=axin.transAxes, fontsize="small",
                    color=0.*np.ones(3), rotation=-(0.38)*90, fontstyle="italic")
            axin.text(0.08, 0.01, "non-symmetric", transform=axin.transAxes, fontsize="small",
                    color=0.*np.ones(3), rotation=-(0.58)*90, fontstyle="italic")
            axin.set_xlim(1, 1000)
            axin.set_ylim(0.001, 1)
            axin.set_xscale("log")
            axin.set_yscale("log")
            axin.set_xticks([]); axin.set_yticks([])
            axin.minorticks_off()
        elif d==2:
            ax.set_xlabel("PC index")
    
    ax = plt.subplot(grid[:5, -2:])
    pos = ax.get_position().bounds
    ax.set_position([pos[0]+0.03, pos[1]+(pos[3]-pos[2]*yratio)/2-0.03, 
                     pos[2]*0.8, pos[2]*yratio*0.8])
    yh = [0, 2, 1]
    for d in range(3):
        ix = np.nonzero(ids==d)[0]
        emax = min(1000, np.array([len(evals_all[i]) for i in ix]).min())
        ev_all = np.array([evals_all[i][:emax] / norms[i] for i in ix])
        ev_mean = ev_all.mean(axis=0)
        ev_std = ev_all.std(axis=0) #/ np.sqrt(len(ix)-1)
        ax.loglog(np.arange(1, len(ev_mean)+1), ev_mean, color=colors[d], lw=1)
        ax.fill_between(np.arange(1, len(ev_mean)+1), ev_mean-ev_std, ev_mean+ev_std,
                        color=colors[d], alpha=0.5, lw=0)
        ax.text(0.05, 0.1 + 0.12*yh[d], r"$\alpha$" + f" = {alphas[ix].mean():.2f}", 
                transform=ax.transAxes, color=colors[d], fontsize="large")
    ax.set_ylim(0.003, 3)
    ax.set_xlim(1, 1000)
    ax.set_xticks([1, 10, 100, 1000])
    ax.set_xticklabels(["1", "10", "100", "1,000"])
    ax.set_yticks([0.01, 0.1 ,1])
    ax.set_yticklabels(["0.01", "0.1", "1"])
    ax.set_xlabel("PC index")
    ax.set_ylabel("normalized variance")
    ax.set_title("average", y=1, loc='left')
    il = 5
    transl = mtransforms.ScaledTranslation(-40 / 72, 0 / 72, fig.dpi_scale_trans)
    il = plot_label(ltr, il, ax, transl)

    ax = ax.inset_axes([0.7, 0.9, 0.3, 0.4])
    for d in range(3):
        ix = np.nonzero(ids==d)[0]
        if d==0:
            ix = np.hstack((ix, np.nonzero(ids==4)[0])) 
            ix = np.hstack((ix, np.nonzero(ids==3)[0])) 
        ash = np.stack((alphas[ix], alphas_shuff[ix]), axis=0)
        p = ttest_rel(ash[0], ash[1]).pvalue
        for i in ix:
            ax.plot(np.arange(0,2) + dy[d] * 1.5, (alphas[i], alphas_shuff[i]),
                    color=colors[d], lw=1., alpha=0.75, ls=ls[ids[i]])
        star = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
        print(p)
        ax.text(dy[d]*1.5 + 0.5 + (dy[d]-1)*0., 0.9, f"{star}", ha="center", 
                va="center", color=colors[d], fontsize="small" if p>=0.05 else "medium")
    ax.set_ylim([0., 1.0])   
    ax.set_yticks([0., 0.5, 1.0])
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["original", "shuffled"], rotation=90, ha="center", va="top")
    ax.set_ylabel("power-law\nexponent ($\\alpha$)")
    #il += 1
    #il = plot_label(ltr, il, ax, transl)

    evals_all = dat['evals_sub'].copy()
    alphas_sim = np.zeros((2, evals_all.shape[1])) * np.nan
    ax = plt.subplot(grid[-2:, 4])
    pos = ax.get_position().bounds
    ypos = pos[1]+(pos[3]-pos[2]*yratio)/2+0.03
    ax.set_position([pose[0], ypos, pos[2]*0.9, pos[2]*yratio*0.9])
    for d in range(2):
        for i in range(len(evals_all[0])):
            ss = evals_all[d, i].copy()
            alphas_sim[d, i], yp = fit_powerlaw_exp(ss, np.arange(10, ymax))
            ss /= yp[0]
            ax.loglog(np.arange(1, min(len(ss)+1, 1001)), 
                    ss[:1000], color=colors[d], lw=0.5, alpha=0.75)
        ax.text(0.35, 0.65 + 0.15*(1-d), r"$\alpha$" + f" = {alphas_sim[d].mean():.2f}", 
                transform=ax.transAxes, color=colors[d], fontsize="medium")
    
    # ax.fill_between(np.arange(1, len(ev_mean)+1), ev_mean-ev_std, ev_mean+ev_std,
    #                     color=colors[d], alpha=0.5, lw=0)
    print(alphas_sim.mean(axis=-1))
    ax.set_title('simulated 2p data', fontsize='medium')
    ax.set_ylim(0.003, 3)
    ax.set_xlim(1, 1000)
    ax.set_xticks([1, 10, 100, 1000])
    ax.set_xticklabels(["1", "10", "100", "1,000"])
    ax.set_yticks([0.01, 0.1 ,1])
    ax.set_yticklabels(["0.01", "0.1", "1"])
    ax.set_xlabel("PC index")
    ax.set_ylabel("normalized variance")
    il += 1
    il = plot_label(ltr, il, ax, transl_e)

    transl = mtransforms.ScaledTranslation(-50 / 72, 0 / 72, fig.dpi_scale_trans)
    evals_enorms = dat['evals_enorms']
    enorms = dat['enorms']
    nnorm = len(enorms)
    n_sim = len(evals_enorms[0])
    alphas_enorms = np.zeros((nnorm, n_sim)) * np.nan
    ecolor = ['g', [0, 1, 0]]
    for d in range(nnorm):
        for i in range(len(evals_enorms[0])):
            alphas_enorms[d, i], yp = fit_powerlaw_exp(evals_enorms[d, i], np.arange(10, ymax))
    ax = plt.subplot(grid[-2:, -2])
    pos = ax.get_position().bounds
    ax.errorbar(enorms, alphas_enorms.mean(axis=1), alphas_enorms.std(axis=1)/(n_sim-1)**0.5, color=ecolor[0])
    ax1 = ax.twinx()
    ax1.set_position([pos[0] + pos[2]*0.25, ypos, 
                        pos[2]*0.5, pos[2]*yratio])
    ax1.plot(enorms, 0.5/(1-enorms) * 0.02, color=ecolor[1])
    ax1.spines['right'].set_visible(True)
    ax1.spines['right'].set_color(ecolor[1])
    ax1.tick_params(axis='y', colors=ecolor[1])
    ax1.set_ylabel('timescale (sec.)', color=ecolor[1], rotation=-90, va='bottom')
    ax1.set_yticks([0, 2, 4])
    ax.spines['left'].set_color(ecolor[0])
    ax.tick_params(axis='y', colors=ecolor[0])
    ax.set_xticks([0.975, 0.985, 0.998])
    ax.set_xlim([0.975, 0.998])
    ax.set_yticks([0.5, 0.6, 0.7])
    ax.set_ylabel('power-law exponent ($\\alpha$)', color=ecolor[0])
    ax.set_position([pos[0] + pos[2]*0.6, ypos, 
                        pos[2]*1.25, pos[2]*yratio])
    il = plot_label(ltr, il, ax, transl)

    return fig, alphas


def suppfig_svca2(dat, dat_subsample, dat_janelia, dat_london):
    fig = plt.figure(figsize=(14, 10))
    yratio = 14 / 10
    grid = plt.GridSpec(5, 5, wspace=0.4, hspace=0.6, figure=fig, 
                        bottom=0.05, top=0.92, left=0.05, right=0.95)
    ix = 5
    il = 0
    dy = 0.

    shapes = dat['shapes']
    titles = ["eigenspectrum - direct", "SVCA", "SVCA2"]
    ticks = ["direct", "SVCA", "SVCA2"]

    evals_all_list = [dat["evals_all"], dat["evals_svca_all"], dat["evals_svca2_all"]]
    areas_all = dat["areas_all"]
    n_dset = len(areas_all)
    ids = np.zeros(n_dset, "int")
    alphas_all = np.zeros((n_dset, 3))
    for d in range(5):
        ids[np.array(areas_all)==areas[d]] = d

    colors = dcolors[:3].copy()
    colors = np.vstack((colors, colors[0], colors[0]))
    ls = ["-", "-", "-", "--", "-."]
    anames = ['2P cortex', '2P CA1', 'brainwide\nephys']

    grid1 = gridspec.GridSpecFromSubplotSpec(3, 3, subplot_spec=grid[:3, :3], wspace=0.2, hspace=0.4)

    for d in range(3):
        for k in range(3):
            ax = plt.subplot(grid1[d, k])
            pos = ax.get_position().bounds
            ax.set_position([pos[0] + 0.02*(1-k), pos[1], pos[3]/yratio, pos[3]])
            ix = (ids==0) + (ids==3) + (ids==4) if d==0 else ids==d
            ix = np.nonzero(ix)[0]
            for i in ix:
                evals = evals_all_list[k][i].copy()
                alphas_all[i, k], yp = fit_powerlaw_exp(evals, np.arange(10, 500))
                evals /= yp[0]
                ax.loglog(np.arange(1, len(evals)+1), evals, color=colors[ids[i]],
                        lw=1, alpha=0.25 if d!=3 else 0.75, ls=ls[ids[i]], zorder=-30*(ids[i]==1) + 20*(ids[i]!=1))
                        
            ax.minorticks_on()
            ax.set_ylim(0.001, 3)
            ax.set_xlim(1, 3000)
            ax.set_yticks([0.01, 0.1, 1])
            ax.set_yticklabels(["0.01", "0.1", "1"], fontsize='small')
            ax.set_xticks([1, 10, 100, 1000])
            ax.set_xticklabels(['1', '10  ', '100  ', '   1,000'], fontsize='small')
            ax.text(0.3, 0.85, "$\\alpha = $" + f"{alphas_all[ix, k].mean():.2f}",
                    color=colors[d], transform=ax.transAxes, fontweight="bold")
            ax.xaxis.set_minor_locator(matplotlib.ticker.LogLocator(base=10, subs=np.arange(2,10), numticks=10))
            if k==0:
                ax.text(0.05, 0.05, anames[d], color=colors[d], 
                        transform=ax.transAxes, fontweight='bold', fontstyle='italic')
                transl = mtransforms.ScaledTranslation(-40 / 72, 5 / 72, fig.dpi_scale_trans)
                il = plot_label(ltr, il, ax, transl)
            if d==0:
                if k==0:
                    ax.set_xlabel("PC index")
                    ax.set_ylabel("normalized variance")
                    #ax.set_title("neural recordings", loc="left", x=-0.1, fontweight="bold")
                
                ax.set_title(titles[k], y=1.1, fontstyle='italic')
            

    ax = plt.subplot(grid[:3, 3])
    pos = ax.get_position().bounds
    ax.set_position([pos[0]- 0.03, pos[1]+0.25*pos[3], pos[2], pos[3]*0.8])
    xp = np.arange(3)*np.ones((len(evals_all_list[0]),1))
    xp += np.random.randn(*xp.shape)*0.05
    cols = np.tile(colors[ids][:,np.newaxis], (1,3)).reshape(-1,3)
    ax.scatter(xp.flatten(), alphas_all.flatten(), color=cols, s=10)
    for d in range(3):
        ix = (ids==0) + (ids==4) + (ids==5) if d==0 else ids==d
        ax.scatter(np.arange(3), alphas_all[ix].mean(axis=0), color=colors[d], 
                s=400, marker="_")
    ax.set_ylabel("power-law exponent ($\\alpha$)")
    ax.set_xticks(np.arange(3))
    ax.set_xticklabels(["direct", "SVCA", "SVCA2"], rotation=45, ha='right')
    transl = mtransforms.ScaledTranslation(-50 / 72, -5 / 72, fig.dpi_scale_trans)
    il = plot_label(ltr, il, ax, transl)
    ax.set_ylim([0.2, 1.4])

    ax = plt.subplot(grid[:3, 4])
    cols = [colors[0], colors[0], [0.8, 0, 0], [0.5, 0, 0.5], [1, 0, 1]]
    pos = ax.get_position().bounds
    ax.set_position([pos[0]-0.01, pos[1]+0.25*pos[3], pos[2]*1.25, pos[3]*0.8])
    ix = (ids==0) + (ids==4) + (ids==5)
    for j in range(5):
        if j<2:
            alphas = alphas_all[ix, (1-j)+1]
        elif j==2:
            alphas = dat_subsample['alphas_all']
        elif j==3:
            alphas = dat_janelia['alphas_all']
        elif j==4:
            alphas = dat_london['alphas_all']
        xp = j*np.ones(len(alphas))
        xp += np.random.randn(*xp.shape)*(0.05 + 0.05*(j==4))
        ax.scatter(xp, alphas, color=cols[j], s=10)
        ax.scatter(j, alphas.mean(), color=cols[j], s=500, marker="_")
    ax.set_ylabel("power-law exponent ($\\alpha$)")
    ax.set_ylim([0.2, 1.4])
    ax.set_xticks(np.arange(5))
    ax.set_xticklabels(["SVCA2, GCaMP8s 22Hz", "SVCA, GCaMP8s 22Hz", 
                        "SVCA, GCaMP8s 3Hz", "SVCA, GCaMP6s 3Hz\n(Janelia)",
                        "SVCA, GCaMP6s 3Hz\n(London)"], rotation=45, ha='right')
    for idx, lbl in enumerate(ax.get_xticklabels()):
        lbl.set_color(cols[idx])
    transl = mtransforms.ScaledTranslation(-50 / 72, -5 / 72, fig.dpi_scale_trans)
    il = plot_label(ltr, il, ax, transl)


    shapes = dat['shapes']
    areas_all = dat['areas_all']
    n_dset = len(shapes)
    ids = np.zeros(n_dset, "int")
    for d in range(5):
        ids[np.array(areas_all)==areas[d]] = d

    anames = ['2P cortex', '2P CA1', 'brainwide ephys']
    scolors = ['g', 'y', 'b']
    bsize = 4000
    ikeeps = np.arange(50, bsize+1, 200)
    ikeeps[-1] = 4000
    ineurs = np.arange(0.01, 1.05, 0.05)
    ineurs[-1] = 1.0    
    nneurons = ineurs * np.array(shapes)[:,:1]
    ntimes = ikeeps / bsize * np.array(shapes)[:,1:] / (22*60)
    xticks = [[[0, 2500, 5000], [0, 50, 100]],
            [[0, 2500, 5000], [0, 50, 100]],
            [[0, 1000, 2000], [0, 15, 30]]]
    fstr = ['', 'svca_', 'svca2_']
    grid1 = gridspec.GridSpecFromSubplotSpec(1, 6, subplot_spec=grid[-2:, :], 
                                             wspace=0.4, hspace=0.4)
    axs = [[plt.subplot(grid1[-2:, 2*d]) for d in range(3)], 
           [plt.subplot(grid1[-2:, 2*d+1]) for d in range(3)]]
    for k in range(3):
        evals_all = [dat[f'evals_{fstr[k]}all_times'].copy(), dat[f'evals_{fstr[k]}all_neurons'].copy()]
        for j in range(2):
            nvar = len(evals_all[j][0])
            alphas_all = np.zeros((n_dset, nvar))*np.nan
            for i in range(len(evals_all[j])):
                for n in range(len(evals_all[j][i])):
                    evals = evals_all[j][i][n].copy()
                    ymax = min(len(evals)//2, 500) #500 if len(evals) > 500 else min(len(evals), 100)
                    if ymax > 15:
                        alphas_all[i,n], yp = fit_powerlaw_exp(evals, np.arange(10, ymax))
                   
            for d in range(3):
                ax = axs[j][d]
                ix = (ids==0) + (ids==4) + (ids==5) if d==0 else ids==d
                nn = nneurons[ix].mean(axis=0) if j==1 else ntimes[ix].mean(axis=0)
                ax.errorbar(nn, np.nanmean(alphas_all[ix], axis=0), 
                            np.nanstd(alphas_all[ix], axis=0) / ((~np.isnan(alphas_all[ix])).sum(axis=0)-1)**0.5, 
                            color=scolors[k], lw=1)
                ax.set_ylim([0.2, 1.5])
                ax.tick_params(axis='both', labelsize='small')
                ax.set_xticks(xticks[d][1-j])
                ax.set_yticks([0.5, 1, 1.5])
                ax.set_xlabel('# of neurons' if j==1 else 'duration (min.)')
                if k==0 and j==0:
                    ax.set_ylabel('power-law exponent ($\\alpha$)')
                    transl = mtransforms.ScaledTranslation(-55 / 72, -5 / 72, fig.dpi_scale_trans)
                    il = plot_label(ltr, il, ax, transl)   
                    ax.set_title(anames[d], loc='left', color=colors[d], 
                                 fontstyle='italic', fontweight='bold')
                    
                if k==0:
                    pos = ax.get_position().bounds
                    ax.set_position([pos[0]+0.015*(j==0)*(k==0), pos[1], pos[2], pos[3]*0.9])

                if j==0 and d==0:
                    ax.text(1, 0.9-0.1*k, ['direct', 'SVCA', 'SVCA2'][k], color=scolors[k],
                            ha='right', transform=ax.transAxes)

    
    return fig


def suppfig_ephys(dat, dat_areas, dat_tbins):
    fig = plt.figure(figsize=(8, 9))
    yratio = 8 / 9
    grid = plt.GridSpec(7, 4, wspace=0.45, hspace=0.6, figure=fig, 
                        bottom=0.07, top=0.98, left=0.1, right=0.95)
    ix = 5
    il = 0
    dy = 0.

    shapes = dat['shapes']
    
    areas_all = dat["areas_all"]
    n_dset = len(areas_all)
    ids = np.zeros(n_dset, "int")
    alphas_all = np.zeros((n_dset, 3))
    for d in range(5):
        ids[np.array(areas_all)==areas[d]] = d


    ax = plt.subplot(grid[:3, :])
    pos = ax.get_position().bounds
    ax.set_position([pos[0]-0.05, pos[1]-0.2*pos[3], pos[2]+0.065, pos[3]*1.08])
    tmin = 1120#42000//7 - 0
    tlen = 22*60*3//7
    Xex = zscore(dat['Xexs'][2][:, tmin:tmin+tlen], axis=1)
    im = ax.imshow(Xex, aspect="auto", cmap="gray_r", vmin=0., vmax=1.25)    
    ax.plot([0, 22*30/7], -Xex.shape[0]*0.03*np.ones(2), color="k", lw=1.5)
    ax.plot(-0.015*tlen*np.ones(2), [0, 50], color="k", lw=1.5)
    ax.text((22*30/2)/7, -0.07*Xex.shape[0], "30 sec.", ha="center", va="center", fontsize="small")
    ax.text(-0.03*tlen, 0, "500 neurons", ha="center", va="bottom", fontsize="small", rotation=90)
    cax = ax.inset_axes([0.85, -0.01, 0.15, 0.025])
    cb = plt.colorbar(im, cax=cax, orientation="horizontal")
    cb.set_ticks([0, 1.0])
    cb.set_ticklabels(["0", "1.0"], fontsize="small")
    ax.text(0.83, -0.005, "z-scored activity", fontsize="small",
            ha="right", va="top", transform=ax.transAxes)
    ax.set_ylim([-Xex.shape[0]*0.05, Xex.shape[0]+0.5])
    ax.set_xlim([-0.028*tlen, tlen])
    ax.axis("off")
    ax.set_title('Rastermap of brainwide ephys activity')
    transl = mtransforms.ScaledTranslation(-20 / 72, 0 / 72, fig.dpi_scale_trans)
    il = plot_label(ltr, il, ax, transl)

    colors = dcolors[:3].copy()

    acolors = ['g', 'm', [0.25, 1, 0.25], 'r', [0.5, 0, 0]]

    ineurs = np.arange(0.01, 1.05, 0.05)
    ineurs[-1] = 1.0    
    nneurons = ineurs * np.array(shapes)[:,:1]
    evals_all = dat[f'evals_svca2_all_neurons'].copy()
    nvar = len(evals_all[0])
    alphas_all = np.zeros((n_dset, nvar))*np.nan
    for i in range(len(evals_all)):
        for n in range(len(evals_all[i])):
            evals = evals_all[i][n].copy()
            ymax = min(len(evals)//2, 500) #500 if len(evals) > 500 else min(len(evals), 100)
            if ymax > 15:
                alphas_all[i,n], yp = fit_powerlaw_exp(evals, np.arange(10, ymax))

    nh = [4, 2, 3, 1, 0]
    area_names = dat_areas['area_groups'].keys()
    evals_all = dat_areas['evals_areas'].copy()
    nareas = len(area_names)
    alphas_areas = np.zeros((nareas, 3)) * np.nan
    nn_areas = np.stack([dat_areas['nneurons'][a] for a in area_names], axis=0)
    for i in range(3):
        ax = plt.subplot(grid[3:5, i+1])
        pos = ax.get_position().bounds
        ax.set_position([pos[0] + (2-i)*0.03 + 0.01, pos[1]-0.02, pos[2]*0.95, pos[2]*yratio*0.95])
        for n, area in enumerate(area_names):
            evals = evals_all[area][i].copy()
            if len(evals) > 0:
                ymax = min(len(evals)//2, 500) #500 if len(evals) > 500 else min(len(evals), 100)
                alphas_areas[n, i], yp = fit_powerlaw_exp(evals, np.arange(10, ymax))
                evals /= yp[0]
                ax.loglog(np.arange(1, len(evals)+1), evals, color=acolors[n], lw=1)
                nstr = '# of neurons ' if nh[n]==0 else ''
                nhn = nh[n] if i<2 else nh[n] - 1*(nh[n]>1)
                ax.text(1, 0.95-0.12*nhn, f'{nstr}= {int(nn_areas[n, i]):,d}', color=acolors[n], 
                        ha='right', transform=ax.transAxes, fontsize='small')
        ax.set_ylim(0.003, 3)
        ax.set_xlim(1, 1000)
        ax.set_yticks([0.01, 0.1, 1])
        ax.set_yticklabels(["0.01", "0.1", "1"], fontsize='small')
        ax.set_xticks([1, 10, 100, 1000])
        ax.set_xticklabels(['1', '10', '100', '1,000'], fontsize='small')
        ax.set_title(f'mouse {i+1}', fontsize='medium', y=1.1)
        ax.set_xlabel("PC index")
        if i==0:
            ax.set_ylabel("normalized variance")
            transl = mtransforms.ScaledTranslation(-50 / 72, 10 / 72, fig.dpi_scale_trans)
            il = 2
            il = plot_label(ltr, il, ax, transl)
            il -= 2

                    
    d = 2
    ax = plt.subplot(grid[3:5, 0])
    pos = ax.get_position().bounds
    ax.set_position([pos[0], pos[1]-0.02, pos[2], pos[2]*yratio*1.1])
    ix = ids==d
    nn = nneurons[ix].mean(axis=0)
    ax.errorbar(nn, np.nanmean(alphas_all[ix], axis=0), 
                np.nanstd(alphas_all[ix], axis=0) / ((~np.isnan(alphas_all[ix])).sum(axis=0)-1)**0.5, 
                color=colors[d], lw=1)
    for n, area in enumerate(area_names):
        ax.scatter(nn_areas[n], alphas_areas[n], color=acolors[n], 
                    s=30, marker='x', zorder=30, alpha=0.9)
        ax.text(1.25, 1-0.12*nh[n], area, color=acolors[n], ha='right',
                transform=ax.transAxes, fontsize='small', va='bottom')
    ax.text(1.25, 0.02, u'\u2014 random\nsubsets', color=dcolors[d], ha='right',
                transform=ax.transAxes, fontweight='bold')
    ax.set_ylim([0.4, 1.8])
    ax.set_xscale('log')
    ax.set_xticks([100, 1000])
    ax.set_xticklabels(['100', '1,000'])
    ax.set_xlabel('# of neurons')
    ax.set_ylabel('power-law exponent ($\\alpha$)')


    transl = mtransforms.ScaledTranslation(-55 / 72, 0 / 72, fig.dpi_scale_trans)
    il = plot_label(ltr, il, ax, transl)
    il += 1

    tbins = dat_tbins['tbins']
    acolors = plt.get_cmap('YlOrBr_r')(np.linspace(0, 0.6, len(tbins)))
    evals_all = dat_tbins['evals_tbins']
    alphas_tbins = np.zeros((len(tbins), 3)) * np.nan
    for i in range(3):
        ax = plt.subplot(grid[5:7, i+1])
        pos = ax.get_position().bounds
        ax.set_position([pos[0] + (2-i)*0.03 + 0.01, pos[1], pos[2]*0.95, pos[2]*yratio*0.95])
        for n, tbin in enumerate(tbins):
            evals = evals_all[i][n].copy()
            if len(evals) > 0:
                ymax = min(len(evals)//2, 500) #500 if len(evals) > 500 else min(len(evals), 100)
                alphas_tbins[n, i], yp = fit_powerlaw_exp(evals, np.arange(10, ymax))
                evals /= yp[0]
                ax.loglog(np.arange(1, len(evals)+1), evals, color=acolors[n], lw=1)
        ax.set_ylim(0.003, 3)
        ax.set_xlim(1, 1000)
        ax.set_yticks([0.01, 0.1, 1])
        ax.set_yticklabels(["0.01", "0.1", "1"], fontsize='small')
        ax.set_xticks([1, 10, 100, 1000])
        ax.set_xticklabels(['1', '10', '100', '1,000'], fontsize='small')
        ax.set_title(f'mouse {i+1}', fontsize='medium')
        ax.set_xlabel("PC index")
        if i==0:
            ax.set_ylabel("normalized variance")
            transl = mtransforms.ScaledTranslation(-50 / 72, 10 / 72, fig.dpi_scale_trans)
            il += 1
            il = plot_label(ltr, il, ax, transl)
            il -= 2

    ax = plt.subplot(grid[5:7, 0])
    pos = ax.get_position().bounds
    ax.set_position([pos[0], pos[1], pos[2], pos[2]*yratio*1.1])
    ax.errorbar(tbins*1000, alphas_tbins.mean(axis=-1), 
                alphas_tbins.std(axis=-1)/(2**0.5), color='k')
    for n, tbin in enumerate(tbins):
        ax.scatter(tbin*1000, alphas_tbins[n].mean(), s=30, 
                  color=acolors[n], zorder=30)
    ax.set_xscale('log')
    ax.set_xlabel('time bin (ms)')
    ax.set_ylabel('power-law exponent ($\\alpha$)')
    ax.set_xticks([10, 100])
    ax.set_xticklabels(['10', '100'])
    ax.set_ylim([0.65, 0.85])
    ax.set_yticks([0.7, 0.8])

    transl = mtransforms.ScaledTranslation(-58 / 72, 0 / 72, fig.dpi_scale_trans)
    il = plot_label(ltr, il, ax, transl)

    return fig


def suppfig_running(dat):
    colors = [[0.7, 0, 0], [1, 0.5, 0.5]]
    fig = plt.figure(figsize=(14, 3), dpi=150)
    yratio = 14 / 3
    grid = plt.GridSpec(1, 5, wspace=0.4, hspace=0.2, figure=fig, 
                        bottom=0.18, top=0.9, left=0.05, right=0.95)
    transl = mtransforms.ScaledTranslation(-45 / 72, 5 / 72, fig.dpi_scale_trans)
    il = 0
    evals_run_all = dat['evals_run_all'].copy()
    nrun = evals_run_all.shape[1]
    alphas_all = np.zeros((2, nrun))
    ax = plt.subplot(grid[0, 0])
    pos = ax.get_position().bounds
    ax.set_position([pos[0], pos[1], pos[2], pos[2]*yratio])
    rstr = ['running', 'not running']
    for j in range(2):
        for i in range(nrun):
            alphas_all[j, i], yp = fit_powerlaw_exp(evals_run_all[j, i], np.arange(10, 500))
            evals_run_all[j, i] /= yp[0]
        evals = np.nanmean(evals_run_all[j], axis=0)
        evals_std = np.nanstd(evals_run_all[j], axis=0) / (nrun-1)**0.5
        ax.loglog(np.arange(1, len(evals)+1), evals, lw=1, color=colors[j])
        ax.fill_between(np.arange(1, len(evals)+1), evals - evals_std, evals + evals_std, 
                   edgecolor='none', facecolor=colors[j], alpha=0.25)
        alpha = fit_powerlaw_exp(evals, np.arange(10, 500))[0]
        ax.text(1, 1-0.1*j, f'{rstr[j]}, $\\alpha$={alpha:.2f}', ha='right',
                transform=ax.transAxes, color=colors[j])
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_ylim(0.003, 0.003*700)
    ax.set_xlim(1, 700)
    ax.set_yticks([0.01, 0.1, 1])
    ax.set_yticklabels(["0.01", "0.1", "1"], fontsize='small')
    ax.set_xticks([1, 10, 100])
    ax.set_xticklabels(['1', '10', '100'], fontsize='small')
    ax.set_xlabel("PC index")
    ax.set_ylabel("normalized variance")
    il = plot_label(ltr, il, ax, transl)

    ax = plt.subplot(grid[0, 1])
    pos = ax.get_position().bounds
    ax.set_position([pos[0] + pos[2]*0.1, pos[1], pos[2]*0.5, pos[2]*yratio])
    areas = dat['areas']
    ls = {'V1': '-', 'sensorimotor': '--', 'PPC': '-.'}
    for i in range(nrun):
        ax.plot(np.arange(0,2), alphas_all[:,i],
                    color='k', lw=1., alpha=0.75, ls=ls[areas[i]])
    p = ttest_rel(alphas_all[0], alphas_all[1]).pvalue
    star = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
    print(p)
    ax.text(0.5, 1., f"{star}", ha="center", 
            va="center", color='k', fontsize="small" if p>=0.05 else "medium")
    ax.set_ylim([0., 1.0])   
    ax.set_yticks([0., 0.5, 1.0])
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["running", "not\nrunning"], rotation=0, ha="center", va="top")
    ax.set_ylabel("power-law exponent ($\\alpha$)")
    lns = []
    area_names = ['V1', 'sensorimotor', 'PPC']
    for area in area_names:
        ln = ax.plot([], [], color='k', lw=1, ls=ls[area])
        lns.append(ln[0])
    ax.legend(lns, area_names, 
                frameon=False, loc="lower left", #bbox_to_anchor=(0.2, 1.1), 
                handlelength=1.2)
    il = plot_label(ltr, il, ax, transl)

    iex = 2
    for j in range(2):
        ax = plt.subplot(grid[0, 2+j])
        pos = ax.get_position().bounds
        ax.set_position([pos[0], pos[1], pos[2], pos[2]*yratio])
        e = dat['dmd_evals_all'][j, iex].copy()
        ix = np.abs(e)>.25
        iang = np.angle(e[ix]) / (2*np.pi)
        iabs = -np.log10(np.abs(e[ix]))    
        irot = iang/iabs
        ixx = e[ix].imag>=0
        mu = irot[ixx].mean()
        sd = irot[ixx].std()
        m = np.percentile(irot[ixx], [5, 25, 75, 95])
        med = np.median(irot[ixx])
        print(m, med)
        ax.scatter(e.real, e.imag, s=10, color=colors[j])
        ax.set_ylim([-1, 1])
        ax.set_xlim([0.25, 1])
        ax.set_xlabel('real part')
        ax.set_ylabel('imaginary part')
        ax.text(0.5, 0.8, rstr[j], color=colors[j], ha='center', transform=ax.transAxes)
        if j==0:
            il = plot_label(ltr, il, ax, transl)
            ax.set_title('Eigenvalues of DMD matrix (dt=0.23s), example mouse', loc='left', y=1.02, fontsize='medium')

    ax = plt.subplot(grid[0, -1])
    pos = ax.get_position().bounds
    ax.set_position([pos[0], pos[1], pos[2], pos[2]*yratio])
    igood = np.nonzero(~np.isnan(dat['dmd_evals_all'][0,:,0]))[0]
    for i, iex in enumerate(igood):
        for j in range(2):
            e = dat['dmd_evals_all'][j, iex].copy()
            ix = np.abs(e)>.25
            iang = np.angle(e[ix]) / (2*np.pi)
            iabs = -np.log10(np.abs(e[ix]))    
            irot = iang/iabs
            ixx = e[ix].imag>=0
            mu = irot[ixx].mean()
            sd = irot[ixx].std()
            m = np.percentile(irot[ixx], [5, 25, 75, 95])
            med = np.median(irot[ixx])
            ax.plot([m[0], m[-1]], -(4*i - j) * np.ones(2), lw=1, color=colors[j])
            ax.plot([m[1], m[2]], -(4*i - j) * np.ones(2), lw=4, color=colors[j])
            if j==0:
                ax.text(-0.1, -4*i, f'mouse {len(igood) - i}', rotation=90, ha='center', va='center', fontsize='small')
    ax.set_xlim([-0.1, 1])
    ax.set_ylim([3, -9])
    ax.spines['left'].set_visible(False)
    ax.set_yticks([])
    ax.set_xlabel('rotations per 10-fold attenuation')
    il = plot_label(ltr, il, ax, transl)

    return fig




# def suppfig_svca(dat):
#     fig = plt.figure(figsize=(14, 14), dpi=150)
#     yratio = 14 / 14
#     grid = plt.GridSpec(5, 6, wspace=0.4, hspace=0.2, figure=fig, 
#                         bottom=0.05, top=1, left=0.08, right=0.95)
#     ix = 5
#     transl = mtransforms.ScaledTranslation(-45 / 72, ix / 72, fig.dpi_scale_trans)
#     il = 0
#     dy = 0.
    
#     titles = ["eigenspectrum - direct", "SVCA", "SVCA2"]
#     ticks = ["direct", "SVCA", "SVCA2"]
    
#     evals_all_list = [dat["evals_all"], dat["evals_svca_all"], dat["evals_svca2_all"]]
#     areas_all = dat["areas_all"]
#     ids = np.zeros(len(areas_all), "int")
#     alphas_all = np.zeros((len(evals_all_list[0]), 3))
#     for d in range(5):
#         ids[np.array(areas_all)==areas[d]] = d

#     colors = dcolors[:3].copy()
#     colors.append(colors[0])
#     colors.append(colors[0])
#     ls = ["-", "-", "-", "--", "-."]
    
#     for d in range(5):
#         ax = plt.subplot(grid[-1, d])
#         pos = ax.get_position().bounds
#         ax.set_position([pos[0], pos[1], pos[2], pos[2]*yratio])
#         for i in range(len(evals_all_list[d])):
#             evals = evals_all_list[d][i].copy()
#             alphas_all[i, d], yp = fit_powerlaw_exp(evals, np.arange(10, 500))
#             evals /= yp[0]
#             ax.loglog(np.arange(1, len(evals)+1), evals, color=dcolors[ids[i]],
#                     lw=1, alpha=1, ls=ls[d])
#         ax.set_ylim(0.001, 3)
#         ax.set_xlim(1, 3000)
#         ax.set_yticks([0.01, 0.1, 1])
#         ax.set_yticklabels(["0.01", "0.1", "1"])
#         ax.set_xticks([1, 10, 100, 1000])
#         ax.set_xticklabels(["1", "10", "100", "1,000"])
#         for k in range(3):
#             ax.text(0.05, 0.32 - 0.12*k, "$\\alpha = $" + f"{alphas_all[ids==k, d].mean():.2f}",
#                     color=dcolors[k], transform=ax.transAxes, fontweight="bold")
#         ax.set_xlabel("PC index")
#         if d==0:
#             ax.set_ylabel("normalized variance")
#             il = plot_label(ltr, il, ax, transl)
#             ax.set_title("neural recordings", loc="left", x=-0.1, fontweight="bold")

#     ax = plt.subplot(grid[-1, 3])
#     pos = ax.get_position().bounds
#     ax.set_position([pos[0] + 0.025, pos[1]+dy-0.02, pos[2], pos[2]*yratio + 0.04])
#     xp = np.arange(3)*np.ones((len(evals_all_list[0]),1))
#     xp += np.random.randn(*xp.shape)*0.05
#     cols = np.tile(dcolors[ids][:,np.newaxis], (1,3)).reshape(-1,3)
#     ax.scatter(xp.flatten(), alphas_all.flatten(), color=cols, s=10)
#     for k in range(3):
#         ax.scatter(np.arange(3), alphas_all[ids==k].mean(axis=0), color=dcolors[k], 
#                 s=400, marker="_")
#     ax.set_ylabel("power-law exponent ($\\alpha$)")
#     ax.set_xticks(np.arange(3))
#     ax.set_xticklabels(["direct", "SVCA", "SVCA2"], rotation=0)
#     il = plot_label(ltr, il, ax, transl)

#     return fig

# def suppfig_svca2_sizes(dat_sim, alphas=None):
#     evals_svca2_all = dat_sim["evals_svca2_all"]
#     nonsyms = dat_sim["nonsyms"]
#     nneurons = dat_sim["nneurons"]
#     ntimes = dat_sim["ntimes"].astype('float32') / 23 # convert to seconds
#     noise_levels = dat_sim["noise_levels"]
#     n_sim = len(evals_svca2_all)

#     if alphas is None:
#         alphas = np.zeros((evals_svca2_all.shape[:-1]))
#         for i in range(n_sim):
#             for ni, nonsym in enumerate(nonsyms):
#                 for nl, noise_level in enumerate(noise_levels):
#                     for ii, nneur in enumerate(nneurons):
#                         for jj, ntime in enumerate(ntimes):
#                                 evals = evals_svca2_all[i,ni,nl,ii,jj].copy()
#                                 ymax = min((~np.isnan(evals)).sum(), int(nneur*0.4), int(ntime*0.4))
#                                 evals = evals_svca2_all[i,ni,nl,ii,jj].copy()
#                                 yrange = np.arange(10, min(500, (~np.isnan(evals)).sum()-1))
#                                 alpha = fit_powerlaw_exp(evals, yrange)[0]
#                                 alphas[i,ni,nl,ii,jj] = alpha


#     fig = plt.figure(figsize=(14, 9), dpi=150)
#     yratio = 14 / 9
#     grid = plt.GridSpec(4, 7, wspace=0.6, hspace=0.7, figure=fig, 
#                         bottom=0.05, top=0.95, left=0.06, right=0.98)
#     grid1 = gridspec.GridSpecFromSubplotSpec(4, 6, subplot_spec=grid[:, 2:], wspace=0.3, hspace=0.7)
#     il = 0 

#     n_sim = len(evals_svca2_all)

#     for ni, nonsym in enumerate(nonsyms):
#         for j in range(2):
#             if j==0:
#                 lcolors = plt.get_cmap("Oranges")(np.linspace(0.5, 1, len(nneurons)//2))
#             else:
#                 lcolors = plt.get_cmap("Greens")(np.linspace(0.5, 1, len(ntimes)//2))
            
#             ax = plt.subplot(grid[ni, j])
#             pos = ax.get_position().bounds
#             ax.set_position([pos[0], pos[1], pos[2], pos[2]*yratio])
#             if ni==0:
#                 transl = mtransforms.ScaledTranslation(-40 / 72, -4 / 72, fig.dpi_scale_trans)
#             else:
#                 transl = mtransforms.ScaledTranslation(-40 / 72, -10 / 72, fig.dpi_scale_trans)
#             il = plot_label(ltr, il, ax, transl)
            
#             if j==0:
#                 if ni==0:
#                     x0, y0 = -0.65, 1.3
#                 else:
#                     x0, y0 = -0.65, 1.2
#                 ax.text(x0, y0, ["symmetric", "1/3 non-symmetric", "2/3 non-symmetric", "non-symmetric"][ni] + " connectivity",
#                                     transform=ax.transAxes, fontsize="medium", fontweight="bold", fontstyle='italic')    
#             if ni == 0:
#                 ax.text(0.7, 1.05, '# neurons = ' if j==0 else 'time =', 
#                         color='k', transform=ax.transAxes, ha='right')
                
#             for ii, nn in enumerate(nneurons) if j ==0 else enumerate(ntimes):
#                 if ii%2==1:
#                     continue
#                 evals = evals_svca2_all[0,ni,0,ii,-1].copy() if j==0 else evals_svca2_all[0,ni,0,-1,ii].copy()
#                 ymax = min((~np.isnan(evals)).sum(), int(nn*0.4))
#                 evals = evals[:ymax]
#                 alpha = plot_spectrum(ax, evals, color=lcolors[ii//2], lw=1, plot_fit=False)
#                 if ii%4==0 and ni==0:
#                     if j==0:
#                         ax.text(0.75, 0.65 + 0.1*ii/4,  f"{nn:.0f}",
#                                 color=lcolors[ii//2], transform=ax.transAxes, fontsize="small")
#                     else:
#                         ax.text(0.75, 1.05 - 0.1*ii/4,  f"{nn/60:.1f} min",
#                                 color=lcolors[ii//2], transform=ax.transAxes, fontsize="small")
#             if ni==0:
#                 if j==0:
#                     ax.set_xlabel("PC index")
#                     ax.set_ylabel("normalized\nvariance")
#                 else:
#                     ax.set_xlabel("PC index")

#         for j in range(len(noise_levels)):
#             ax = plt.subplot(grid1[ni, j])
#             pos = ax.get_position().bounds
#             ax.set_position([pos[0]+0.008*(len(noise_levels) - j), pos[1], pos[2], pos[2]*yratio])
#             im = ax.imshow(alphas.mean(axis=0)[ni, j].T, vmin=0.5, vmax=2, cmap='viridis',
#                     aspect='auto')
#             #ax.invert_yaxis()
            
#             xticks = np.array(1 * 2**np.arange(0, 9, 2))
#             f = interp1d(ntimes, np.arange(len(ntimes)))
#             ax.set_xticks(f(xticks * 60))
#             yticks = np.array(200 * 2**np.arange(0, 6, 1))
#             yticks = np.hstack((yticks, 10000))
#             f = interp1d(nneurons, np.arange(len(nneurons)))
#             ax.set_yticks(f(yticks))
#             ax.tick_params(labelsize='small')
#             #ax.axis('square')
#             ax.set_title(['low', 'medium', 'high'][j%3], fontsize='small')
#             if j==1:
#                 ax.text(0.5, 1.2, 'Gaussian noise + smoothing', fontsize='medium', 
#                             ha='center', transform=ax.transAxes, fontstyle='italic')
#             elif j==4:
#                 ax.text(0.5, 1.2, 'Poisson noise', fontsize='medium', 
#                             ha='center', transform=ax.transAxes, fontstyle='italic')

#             if ni==0:
#                 if j==1:
#                     axin = ax.inset_axes([0.05, -0.25, 0.8, 0.1]);
#                     cb = plt.colorbar(im, cax=axin, orientation='horizontal')
#                     cb.ax.tick_params(labelsize='small')
#                     axin.text(1.05, 0.5, 'power-law exponent ($\\alpha$)', fontsize='small', 
#                             ha='left', transform=axin.transAxes, va='center')
#                     cb.ax.set_xticks([0.5, 1, 1.5, 2])
#                     #cb.ax.yaxis.label.set_size('medium')
#                 elif j==0:
#                     ax.set_ylabel('# of neurons')
#                     ax.set_xlabel('time (min)')
                    
#             if j==0:
#                 transl = mtransforms.ScaledTranslation(-50 / 72, -5 / 72, fig.dpi_scale_trans)
#                 il = plot_label(ltr, il, ax, transl)
#                 ax.set_yticklabels(yticks)
#                 ax.set_xticklabels(xticks, rotation=30, ha='right', va='top')
#             else:
#                 ax.set_yticklabels([])
#                 ax.set_xticklabels([])
                
#     return fig