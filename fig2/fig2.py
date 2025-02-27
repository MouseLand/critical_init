from matplotlib.patches import Ellipse
from fig_utils import *
from scipy.stats import zscore
from powerlaw import fit_powerlaw_exp
from scipy.stats import ttest_rel


def fig2(dat):
    areas_all = dat["areas_all"]
    evals_all = dat["evals_svca2_all"]
    evals_shuff_all = dat["evals_shuff_all"]
    Xexs = dat["Xexs"]

    fig = plt.figure(figsize=(14, 7), dpi=300)
    yratio = 14/7
    grid = plt.GridSpec(3, 6, wspace=0.65, hspace=0.2, figure=fig, 
                        bottom=0.02, top=0.96, left=0.02, right=0.98)
    titles = ["V1 two-photon imaging", "CA1 two-photon imaging", 
              "whole-brain Neuropixels"]
    sneur = ["6,433 - 10,595 ROIs", "3,981 - 8,519 ROIs",  "1,716 - 2,914 units"]
    colors = dcolors[:3].copy()
    n_dset = len(areas_all)
    tmins = [131000, 0, 42000] # 100000, 27000
    tlen = 8000
    ids = np.zeros(n_dset, "int")
    for d in range(3):
        ids[np.array(areas_all)==areas[d]] = d
    dy = [0, 2, 1]
    alphas = np.zeros(n_dset)
    alphas_shuff = np.zeros(n_dset)
    norms = np.zeros(n_dset)

    if "Fexs" in dat:
        # example traces (extracted from suite2p folders from example data)
        # (if available make example panels)
        Fexs = dat["Fexs"]
        spkexs = dat["spkexs"]
        imexs = dat["imexs"]
        masksexs = dat["masksexs"]    

        for d in range(3):
            ax = plt.subplot(grid[dy[d], 0])
            pos = ax.get_position().bounds
            ax.set_position([pos[0], pos[1]-0.055, pos[2]*1.3, pos[3]*1.3])
            if d==2:
                im = plt.imread("allenprobes.png")
                ax.imshow(im)
            else:
                ax.imshow(imexs[d], vmin=0, vmax=0.9, 
                        cmap="gray", aspect=0.75/0.5 if d==1 else 1)
                masks0 = masksexs[d].copy()
                yo, xo = np.nonzero(masks0[:,:,-1]>0)
                masks0[yo,xo,-1] = 0.25
                ax.imshow(masks0, aspect=0.75/0.5 if d==1 else 1)
                if d==0:
                    ax.set_ylim([380, 380+90])
                    ax.set_xlim(75, 75 + 100)
                elif d==1:
                    #ax.set_ylim([350, 430])
                    #ax.set_xlim([100, 100 + 80*0.75/0.5])
                    ax.set_ylim([50, 50+80])
                    ax.set_xlim([280, 280+90*0.75/0.5])
            ax.set_title(titles[d], fontstyle="italic", y=1.075, loc="left")
            ax.text(0.5, 1.025, sneur[d], transform=ax.transAxes, 
                    fontsize="small", ha="center")
            ax.axis("off")
            il = dy[d]
            transl = mtransforms.ScaledTranslation(-20 / 72, 12 / 72, fig.dpi_scale_trans)
            il = plot_label(ltr, il, ax, transl)

            ax = plt.subplot(grid[dy[d], 1])
            pos = ax.get_position().bounds
            ax.set_position([pos[0], pos[1], pos[2]*1.4, pos[3]])
            if d==2:
                for i in range(len(spkexs[d])):
                    ax.plot(spkexs[d][i] + i, color=colors[d], lw=1)
            else:
                for i in range(len(spkexs[d])):
                    ax.plot(Fexs[d][i] + i, color=colors[d], lw=1)
                    ax.plot(spkexs[d][i] + i, color="k", lw=0.75)
            ax.plot([0, 22], -0.35*np.ones(2), color="k", lw=1.5)
            if d==0:
                ax.text(22/2, -1, "1 sec.", ha="center", va="center", fontsize="small")
            if d==0 or d==1:
                ax.text(0.0, 0.95, "fluorescence", color=colors[d], transform=ax.transAxes,)
                ax.text(0.55, 0.95, "deconvolved", color="k", transform=ax.transAxes,)
            else:
                ax.text(0.5, 0.95, "spiking (binned at 22Hz)", color=colors[d], 
                        transform=ax.transAxes, ha = "center")
            ax.set_ylim([-0.5, len(spkexs[d])+1])
            ax.axis("off")

    for d in range(3):
        ax = plt.subplot(grid[dy[d], 2:4])
        Xe = zscore(Xexs[d][:,tmins[d]:tmins[d] + tlen].copy(), axis=1)
        im = ax.imshow(Xe, 
                    aspect="auto", cmap="gray_r", vmin=0., 
                    vmax=0.75)    
        ax.plot([0, 22*30], -8*np.ones(2) if d==2 else -15*np.ones(2), color="k", lw=1.5)
        ax.plot(-200*np.ones(2), [0, 50], color="k", lw=1.5)
        if d==0:
            ax.text(22*30/2, -50, "30 sec.", ha="center", va="center", fontsize="small")
            ax.text(-400, 0, "1000 neurons", ha="center", va="bottom", fontsize="small", rotation=90)
            cax = ax.inset_axes([0.85, -0.01, 0.15, 0.025])
            cb = plt.colorbar(im, cax=cax, orientation="horizontal")
            cb.set_ticks([0, 0.5])
            cb.set_ticklabels(["0", "0.5"], fontsize="small")
            ax.text(0.92, -0.12, "z-scored activity", fontsize="small",
                    ha="center", va="top", transform=ax.transAxes)
        ax.set_ylim([-22, Xexs[d].shape[0]+0.5])
        ax.set_xlim([-250, tlen])
        ax.axis("off")
        if dy[d]==0:
            il = 3
            transl = mtransforms.ScaledTranslation(-10 / 72, -4 / 72, fig.dpi_scale_trans)
            il = plot_label(ltr, il, ax, transl)

        ax = fig.add_subplot(grid[dy[d], 4])
        pos = ax.get_position().bounds
        ax.set_position([pos[0], pos[1]+(pos[3]-pos[2]*yratio)/2+0.01, 
                        pos[2], pos[2]*yratio])
        ix = np.nonzero(ids==d)[0]
        for i in ix:
            ss = evals_all[i][:1000].copy()
            alphas[i], yp = fit_powerlaw_exp(evals_all[i], 
                                                        np.arange(10, 500))
            ss /= yp[0]
            norms[i] = yp[0] #yp[10] * 10
            ax.loglog(np.arange(1, min(len(evals_all[i])+1, 1001)), 
                    ss, color=colors[d], lw=0.75)
            alphas_shuff[i] = fit_powerlaw_exp(evals_shuff_all[i], 
                                                        np.arange(10, 500))[0]
        ax.set_ylim(0.003, 3)
        ax.set_xlim(1, 1000)
        ax.set_xticks([1, 10, 100, 1000])
        ax.set_xticklabels(["1", "10", "100", "1,000"])
        ax.set_yticks([0.01, 0.1 ,1])
        ax.set_yticklabels(["0.01", "0.1", "1"])
        aexp =  [-0.69, -1.254]
        yc = 1
        ax.fill_between([1, 1000], [yc, yc * 1000**aexp[0]], [yc, yc * 1000**aexp[1]], 
                        color="k", lw=0, alpha=0.05)
        if dy[d]==0:
            il = 4
            transl = mtransforms.ScaledTranslation(-50 / 72, 8 / 72, fig.dpi_scale_trans)
            il = plot_label(ltr, il, ax, transl)
            ax.set_xlabel("PC index")
            ax.set_ylabel("normalized variance")
            axin = ax.inset_axes([0.9, 0.5, 0.75, 0.75])
            axin.fill_between([1, 1000], [yc, yc * 1000**aexp[0]], [yc, yc * 1000**aexp[1]], 
                        color="k", lw=0, alpha=0.05)
            axin.text(0.27, 0.37, "symmetric", transform=axin.transAxes, 
                    color=0.*np.ones(3), rotation=-(0.38)*90, fontstyle="italic")
            axin.text(0.08, 0.01, "non-symmetric", transform=axin.transAxes, 
                    color=0.*np.ones(3), rotation=-(0.58)*90, fontstyle="italic")
            axin.set_xlim(1, 1000)
            axin.set_ylim(0.001, 1)
            axin.set_xscale("log")
            axin.set_yscale("log")
            axin.set_xticks([]); axin.set_yticks([])
            axin.minorticks_off()
    
    grid1 = matplotlib.gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=grid[:,-1],
                                                            wspace=0., hspace=0.)

    ax = plt.subplot(grid1[0])
    pos = ax.get_position().bounds
    ax.set_position([pos[0], pos[1]+0.05, pos[2], pos[2]*yratio])
    dh = [0, 2, 1]
    for d in range(3):
        ix = np.nonzero(ids==d)[0]
        emax = min(1000, np.array([len(evals_all[i]) for i in ix]).min())
        ev_all = np.array([evals_all[i][:emax] / norms[i] for i in ix])
        ev_mean = ev_all.mean(axis=0)
        ev_std = ev_all.std(axis=0) #/ np.sqrt(len(ix)-1)
        ax.loglog(np.arange(1, len(ev_mean)+1), ev_mean, color=colors[d], lw=1)
        ax.fill_between(np.arange(1, len(ev_mean)+1), ev_mean-ev_std, ev_mean+ev_std,
                        color=colors[d], alpha=0.5, lw=0)
        ax.text(0.5, 0.62 + 0.12*dh[d], r"$\alpha$" + f" = {alphas[ix].mean():.2f}", 
                transform=ax.transAxes, color=colors[d])
    ax.set_ylim(0.003, 3)
    ax.set_xlim(1, 1000)
    ax.set_xticks([1, 10, 100, 1000])
    ax.set_xticklabels(["1", "10", "100", "1,000"])
    ax.set_yticks([0.01, 0.1 ,1])
    ax.set_yticklabels(["0.01", "0.1", "1"])
    ax.set_xlabel("PC index")
    ax.set_ylabel("normalized variance")
    ax.set_title("average")
    il = 5
    transl = mtransforms.ScaledTranslation(-50 / 72, 5 / 72, fig.dpi_scale_trans)
    il = plot_label(ltr, il, ax, transl)

    ax = plt.subplot(grid1[1])
    pos = ax.get_position().bounds 
    ax.set_position([pos[0], pos[1]+0.17, pos[2], pos[2]*yratio])
    for d in range(3):
        ix = np.nonzero(ids==d)[0]
        ash = np.stack((alphas[ix], alphas_shuff[ix]), axis=0)
        p = ttest_rel(ash[0], ash[1], alternative="greater").pvalue
        ax.plot(np.tile((np.arange(0,2) + dy[d] * 1.5)[:,np.newaxis], (1, len(ix))),
                ash, color=colors[d], lw=1.5)
        star = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
        print(p)
        ax.text(dy[d]*1.5 + 0.5 + (dy[d]-1)*0., 0.9, f"{star}", ha="center", 
                va="center", color=colors[d], fontsize="small" if p>=0.05 else "medium")
    ax.set_ylim([0., 0.95])   
    ax.set_yticks([0.2, 0.4, 0.6, 0.8])
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["original", "shuffled"], rotation=90, ha="center", va="top")
    ax.set_ylabel("power-law\nexponent ($\\alpha$)")
    il += 1
    transl = mtransforms.ScaledTranslation(-50 / 72, 5 / 72, fig.dpi_scale_trans)
    il = plot_label(ltr, il, ax, transl)
    #ax.set_xticks([0.5, 2.5, 4.5])
    #ax.set_xticklabels([areas[dy[0]], areas[dy[1]], areas[dy[2]]])

    return fig


def suppfig_svca(dat_sim, dat_neural=None, save_fig=True):
    fig = plt.figure(figsize=(9.333, 14), dpi=150)
    yratio = 9.333 / 14
    grid = plt.GridSpec(5, 4, wspace=0.4, hspace=0.2, figure=fig, 
                        bottom=0.05, top=1, left=0.08, right=0.95)
    ix = 5
    transl = mtransforms.ScaledTranslation(-45 / 72, ix / 72, fig.dpi_scale_trans)
    il = 0
    dy = 0.
    evals_all_list = [dat_sim["evals_all"], dat_sim["evals_svca_all"],
                      dat_sim["evals_svca2_all"]]

    n_sim = len(evals_all_list[0])
    alphas_all = np.zeros((n_sim, 3, 4))
    titles = ["eigenspectrum - direct", "SVCA", "SVCA2"]
    nonsyms = dat_sim["nonsyms"]
    for k in range(4):
        for d in range(3):
            ax = plt.subplot(grid[k, d])
            pos = ax.get_position().bounds
            ax.set_position([pos[0], pos[1]+dy, pos[2], pos[2]*yratio])
            for i in range(n_sim):
                evals = evals_all_list[d][i, k].copy()
                alphas_all[i, d, k], yp = fit_powerlaw_exp(evals, np.arange(10, 500))
                evals /= yp[0]
                ax.loglog(np.arange(1, len(evals)+1), evals, color="k", lw=0.75, alpha=0.5)
            ax.set_ylim(0.001, 3)
            ax.set_xlim(1, 3000)
            ax.set_yticks([0.01, 0.1, 1])
            ax.set_yticklabels(["0.01", "0.1", "1"])
            ax.set_xticks([1, 10, 100, 1000])
            ax.set_xticklabels(["1", "10", "100", "1,000"]) 
            ax.text(0.5, 0.75, "$\\alpha = $" + f"{alphas_all[:, d, k].mean():.2f}",
                    transform=ax.transAxes)
            ax.set_xlabel("PC index")
            if k==0:
                ax.set_title(titles[d],  x=0.5, y=1.2, fontstyle="italic",
                             fontsize="x-large")
            if d==0:
                ax.text(0, 1.05, ["symmetric", "1/3 non-symmetric", "2/3 non-symmetric", "non-symmetric"][k] + " connectivity",
                        transform=ax.transAxes, fontsize="large", fontweight="bold")
                ax.set_ylabel("normalized variance")
                il = plot_label(ltr, il, ax, transl)
        
        alphas_gt = np.zeros(n_sim)
        for i in range(n_sim):
            alphas_gt[i] = fit_powerlaw_exp(dat_sim["evals_gt_all"][i,k], np.arange(10, 500))[0]
        transl = mtransforms.ScaledTranslation(-45 / 72, ix / 72, fig.dpi_scale_trans)
        ax = plt.subplot(grid[k, -1])
        pos = ax.get_position().bounds
        ax.set_position([pos[0] + 0.025, pos[1]+dy-0.02, pos[2], pos[2]*yratio + 0.04])
        xp = np.arange(3)*np.ones((n_sim,1))
        xp += np.random.randn(*xp.shape)*0.05
        ax.scatter(xp.flatten(), alphas_all[:,:,k].flatten(), color="k", s=10)
        ax.scatter(np.arange(3), alphas_all[:,:,k].mean(axis=0), color="k", s=400, marker="_")
        ax.set_ylabel("power-law exponent ($\\alpha$)")
        ax.plot([-0.3, 2.3], alphas_gt.mean()*np.ones(2), color=0.75*np.ones(3),
                lw=3, ls="--")
        ax.set_xticks(np.arange(3))
        ax.set_xticklabels(["direct", "SVCA", "SVCA2"], rotation=0)
        ax.text(-0.25, alphas_gt.mean()+0.05, "ground-truth", color=0.75*np.ones(3), 
                ha="left", va="bottom", fontstyle="italic")
        ax.set_ylim([0.5, 2.1])
        il = plot_label(ltr, il, ax, transl)
        
    if dat_neural is not None:
        evals_all_list = [dat_neural["evals_all"], dat_neural["evals_svca_all"],
                        dat_neural["evals_svca2_all"]]
        areas_all = dat_neural["areas_all"]
        ids = np.zeros(len(areas_all), "int")
        alphas_all = np.zeros((len(evals_all_list[0]), 3))
        for d in range(3):
            ids[np.array(areas_all)==areas[d]] = d
        
        for d in range(3):
            ax = plt.subplot(grid[-1, d])
            pos = ax.get_position().bounds
            ax.set_position([pos[0], pos[1], pos[2], pos[2]*yratio])
            for i in range(len(evals_all_list[d])):
                evals = evals_all_list[d][i].copy()
                alphas_all[i, d], yp = fit_powerlaw_exp(evals, np.arange(10, 500))
                evals /= yp[0]
                ax.loglog(np.arange(1, len(evals)+1), evals, color=dcolors[ids[i]],
                        lw=1, alpha=1)
            ax.set_ylim(0.001, 3)
            ax.set_xlim(1, 3000)
            ax.set_yticks([0.01, 0.1, 1])
            ax.set_yticklabels(["0.01", "0.1", "1"])
            ax.set_xticks([1, 10, 100, 1000])
            ax.set_xticklabels(["1", "10", "100", "1,000"])
            for k in range(3):
                ax.text(0.05, 0.32 - 0.12*k, "$\\alpha = $" + f"{alphas_all[ids==k, d].mean():.2f}",
                        color=dcolors[k], transform=ax.transAxes, fontweight="bold")
            ax.set_xlabel("PC index")
            if d==0:
                ax.set_ylabel("normalized variance")
                il = plot_label(ltr, il, ax, transl)
                ax.set_title("neural recordings", loc="left", x=-0.1, fontweight="bold")

        ax = plt.subplot(grid[-1, -1])
        pos = ax.get_position().bounds
        ax.set_position([pos[0] + 0.025, pos[1]+dy-0.02, pos[2], pos[2]*yratio + 0.04])
        xp = np.arange(3)*np.ones((len(evals_all_list[0]),1))
        xp += np.random.randn(*xp.shape)*0.05
        cols = np.tile(dcolors[ids][:,np.newaxis], (1,3)).reshape(-1,3)
        ax.scatter(xp.flatten(), alphas_all.flatten(), color=cols, s=10)
        for k in range(3):
            ax.scatter(np.arange(3), alphas_all[ids==k].mean(axis=0), color=dcolors[k], 
                    s=400, marker="_")
        ax.set_ylabel("power-law exponent ($\\alpha$)")
        ax.set_xticks(np.arange(3))
        ax.set_xticklabels(["direct", "SVCA", "SVCA2"], rotation=0)
        il = plot_label(ltr, il, ax, transl)

    return fig