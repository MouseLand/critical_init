from fig_utils import *
from powerlaw import fit_powerlaw_exp

def eval_panel(ax, evals_all, sparsities, colors, sparse=True):
    alphas = np.zeros((len(sparsities), len(evals_all)))
    k = 0
    for i, sparsity in enumerate(sparsities):
        ss_all = evals_all[:,i].copy()
        for j in range(len(ss_all)):
            alpha, yp = fit_powerlaw_exp(ss_all[j], np.arange(10, 500))
            ss_all[j] /= yp[0]
            alphas[i, j] = alpha
        ax.loglog(np.arange(1, ss_all.shape[-1] + 1), ss_all.mean(axis=0), 
                  color=colors[i], lw=1, alpha=1)
        txt = f"{round(sparsity*100, 3):.15g}" if sparse else f"{sparsity:.15g}"
        ax.text(0.35 + 0.35*(i//6), 
                1.1 - 0.1*(i%6), txt, fontsize="x-small", 
            color=colors[i], transform=ax.transAxes, ha="left")
        
    sstr = "p(conn.) (%) = " if sparse else "p(local) / p(global) = "
    ax.text(0.7, 1.25, sstr, transform=ax.transAxes, 
            ha="right", fontsize="small")
    yc = 1#ss_all.mean(axis=0)[1]
    # ax.plot([1, 10000], [yc, 10000**(-0.69)], 
    #     color=0.75*np.ones(3), lw=2, ls="--", alpha=1)
    ax.set_ylim(0.001, 10)
    ax.set_xlim(1, 10000)
    ax.set_xticks([1, 10, 100, 1000])
    ax.set_xticklabels(["1", "10", "100", "    1,000"])
    ax.set_yticks([0.01, 0.1 ,1, 10])
    ax.set_yticklabels(["0.01", "0.1", "1", "10"])
    ax.set_xlabel("PC index")
    ax.set_ylabel("normalized variance")
    return alphas
    
def fig4(dat_sparse, dat_clustered, dat_local, dat_neural):
    fig = plt.figure(figsize=(9.33,9))
    yratio = 9.33/9
    grid = plt.GridSpec(4, 5, wspace=0.65, hspace=0.65, figure=fig, 
                        bottom=0.0, top=0.94, left=0.07, right=0.96)
    transl = mtransforms.ScaledTranslation(-24 / 72, 17 / 72, fig.dpi_scale_trans)
    il = 0
    alphas_all = []
    dy = 0.02
    titles = ["sparse connectivity", "clustered", "local"]   
    for d in range(3):
        if d==0:
            evals_all_sparsities = dat_sparse["evals"]
            sparsities = dat_sparse["sparsities"]
            iex = 6
            nn = 10000
            Aex = (np.random.rand(100, 100) < sparsities[iex]).astype("float32")
            Aex -= sparsities[iex]
            Aex -= np.triu(Aex)
            Aex = 0.5 * (Aex + Aex.T)
            Aex /= (nn * sparsities[iex])**0.5
            iplot = np.arange(0, 100)
        elif d==1:
            sparsities = 0.5 / dat_clustered["pglobals"]
            evals_all_clusters = dat_clustered["evals"]
            Aex = dat_clustered["Aex"]
            vmax = Aex.max()
            iplot = np.arange(2000, 3500, 20)
        elif d==2:
            evals_all_locals = dat_local["evals_local_all"]
            sparsities = 0.5 / dat_local["pglobals"]
            Aex = dat_local["Aex"]
            ypos = dat_local["ypos_ex"]
            xpos = dat_local["xpos_ex"]
            pcs_local_all = dat_local["pcs_local_all"]
            idist = ypos.argsort()
            idist = idist[np.abs(xpos[idist] - xpos.mean())<100]
            idist = idist[np.linspace(0, len(idist)-1, 100).astype("int")]
            vmax = Aex.max()*0.5
            iplot = idist
            
        ax = plt.subplot(grid[0, d + (d>0)])
        pos = ax.get_position().bounds
        ax.set_position([pos[0] , pos[1]+0*dy, pos[2], pos[3]])
        conn_panel(ax, Aex, iplot, vmax=0.02, colorbar=d==0, title=titles[d])
        il = plot_label(ltr, il, ax, transl)
        pos0 = ax.get_position().bounds
        ax.set_position([pos0[0]-0.25*pos0[2], pos0[1]-0.15*pos0[3], pos0[2]*1.25, pos0[3]*1.25])

        colors = plt.get_cmap("gray")(np.linspace(0, 0.8, len(sparsities)))[::-1]
        ax = plt.subplot(grid[1, d + (d>0)])
        pos = ax.get_position().bounds
        ax.set_position([pos0[0], pos[1]+2*dy, pos0[2], pos0[3]])
        alphas_all.append(eval_panel(ax, evals_all_sparsities if d==0 else evals_all_clusters if d==1 else evals_all_locals,
                                    sparsities, colors, sparse=(d==0)))

        if d==0:
            ax = plt.subplot(grid[:2, 1])
            pos = ax.get_position().bounds
            ax.set_position([pos[0], pos[1]+pos[3]*0.25, pos[2]*0.9, pos[3]*0.55])
            amean = alphas_all[0].mean(axis=-1)
            astd = alphas_all[0].std(axis=-1) #/ np.sqrt(10-1)
            ax.errorbar(sparsities*100, amean, yerr=astd, color="k", 
                        marker="o", ms=1, lw=1)
            ax.text(1, 0.95, "uniform\nrandom", ha="right", va="bottom", 
                    transform=ax.transAxes, color=0.75*np.ones(3), fontstyle="italic")
            ax.plot([sparsities[0]*100, sparsities[-1]*100], 0.695*np.ones(2), color=0.75*np.ones(3), 
                    lw=2, ls="--")
            ax.set_xscale("log")
            ax.minorticks_off()
            ax.set_ylabel("power-law exponent ($\\alpha$)")
            ax.set_xlabel("p(conn.) (%)")
            ax.set_xticks([0.1, 1, 10])
            ax.set_xticklabels(["0.1", "1", "10"])
            ax.set_ylim([0., 0.75])
            ax.xaxis.set_minor_locator(locmin)
            ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        elif d==2:
            kcolors = plt.get_cmap("hot")(np.linspace(0., 0.33, 3))
            ax = plt.subplot(grid[:2, -1])
            pos = ax.get_position().bounds  
            ax.set_position([pos[0]+0.01, pos[1]+pos[3]*0.25, pos[2]*0.9, pos[3]*0.55])
            for k in range(1, 3):
                amean = alphas_all[k].mean(axis=-1)
                astd = alphas_all[k].std(axis=-1) #/ np.sqrt(10-1)
                ax.errorbar(sparsities, amean, yerr=astd, color=kcolors[k],
                            marker="o", ms=1, lw=1)
                ax.text(0.1, 0.4 - 0.12*k, ["sparse", "clustered", "local"][k], 
                        ha="left", color=kcolors[k], transform=ax.transAxes)
            ax.plot([sparsities[0],  sparsities[-1]], 0.695*np.ones(2), color=0.75*np.ones(3), 
                    lw=2, ls="--")
            ax.text(1, 0.95, "uniform\nrandom", ha="right", va="bottom", 
                    transform=ax.transAxes, color=0.75*np.ones(3), fontstyle="italic")
            ax.set_xscale("log")
            ax.set_ylim([0., 0.75])
            ax.set_xticks([1, 100])
            ax.set_xticklabels(["1", "100"])
            #ax.set_yticks([0.5, 0.6, 0.7])
            ax.minorticks_off()
            ax.set_ylabel("power-law exponent ($\\alpha$)")
            ax.set_xlabel("p(local) / p(global)   ")
            ax.xaxis.set_minor_locator(locmin)
            ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        
    #dat_locala = np.load("strongpairs_dat_locala.npy", allow_pickle=True).item()
    
    cov_bin_all = dat_local["cov_bin_all"]
    drand_all = dat_local["drand_all"]
    crand_all = dat_local["crand_all"]    
    nbins = cov_bin_all.shape[-1]
    bin_size = dat_local["bin_size"]
    bcent = np.arange(0, nbins*bin_size, bin_size) + bin_size/2
    ax = plt.subplot(grid[2, 0])
    pos = ax.get_position().bounds
    ax.set_position([pos[0]+0.0, pos[1]+0.02, pos[2]-0., pos[3]-0.01])
    #ax.scatter(drand_all[0], crand_all[0], color=colors[i], s=1, alpha=0.25)
    isparse = np.tile(np.arange(0, len(sparsities))[:,np.newaxis], (1, drand_all.shape[-1]))
    irand = np.random.randint(0, drand_all.shape[-1]*drand_all.shape[-2],
                              20000)
    cols = colors[isparse.flatten()[irand]]
    dr = drand_all[0].flatten()[irand]
    cr = crand_all[0].flatten()[irand]
    ax.scatter(dr, cr, color=cols, s=1, alpha=0.05, zorder=0, rasterized=True)
    for i, sparsity in enumerate(sparsities):
         ax.errorbar(bcent, cov_bin_all[0,i].mean(axis=0), 
                        cov_bin_all[0,i].std(axis=0), 
                 color=colors[i], lw=1+0.1*i, zorder=30-i)
    ax.set_xlim([0, 3000])
    ax.set_ylim([-0.1, 0.1])
    ax.set_ylabel("correlation")
    ax.set_xlabel("distance ($\mu$m)")
    ax.set_title("local simulation", loc="center", x=0.5, fontsize="medium")
    ax.text(-0.38, 1.25, "spatial distribution of correlations", fontsize="large", fontstyle="italic",
            transform=ax.transAxes)
    transl = mtransforms.ScaledTranslation(-47 / 72, 18 / 72, fig.dpi_scale_trans)
    il = plot_label(ltr, il, ax, transl)

    colors_data = plt.get_cmap("winter")(np.linspace(0, 1, len(dat_neural["cov_bin_all"])))
    cov_bin_all = dat_neural["cov_bin_all"]
    crand_all = np.array(dat_neural["crand_all"])
    drand_all = np.array(dat_neural["drand_all"])
    idat = np.tile(np.arange(0, drand_all.shape[0])[:,np.newaxis], (1, drand_all.shape[-1]))
    irand = np.random.randint(0, drand_all.shape[-1]*drand_all.shape[-2],
                              20000)
    cols = colors_data[idat.flatten()[irand]]
    dr = drand_all.flatten()[irand]
    cr = crand_all.flatten()[irand]
    ax = plt.subplot(grid[2, 1])
    pos = ax.get_position().bounds
    ax.set_position([pos[0]-0.025, pos[1]+0.02, pos[2]-0.0, pos[3]-0.01])
    ax.scatter(dr, cr, color=cols, s=1, alpha=0.05, zorder=0, rasterized=True)
    for i in range(len(cov_bin_all)):
        #print(cov_bin_all[i].shape)
        ax.errorbar(bcent, cov_bin_all[i][:,:15].mean(axis=0), 
                    cov_bin_all[i][:,:15].std(axis=0), 
                    color=colors_data[i],
                    lw=1+0.1*i, zorder=30-i)
    ax.set_xlim([0, 3000])
    ax.set_ylim([-0.1, 0.1])
    ax.set_title("neural data", loc="center", x=0.5, fontsize="medium")

    ypos_local_all = dat_local["ypos_local_all"]
    xpos_local_all = dat_local["xpos_local_all"]
    pcs_local_all = dat_local["pcs_local_all"]
    pcs_all = dat_neural["pcs_all"]
    ypos_all = dat_neural["ypos_all"]
    xpos_all = dat_neural["xpos_all"]
    
    perc_local_all = dat_local["perc_local_all"]
    perc = dat_local["perc"]
    ax = plt.subplot(grid[2, 2])
    pos = ax.get_position().bounds
    ax.set_position([pos[0]-0.005, pos[1]+0.02, pos[2]+0.01, pos[3]])
    for i, sparsity in enumerate(sparsities):
        ax.loglog(perc*100, perc_local_all[:,i].mean(axis=0) / dat_local["pconn"][i], 
                    color=colors[i], lw=1)
    ax.plot([1, 1], [1, 2e3], color="k", ls="--", lw=2)
    ax.text(0.02, 1, "strong\npairs", transform=ax.transAxes, 
            fontsize="small", va="top")
    ax.set_xticks([1e-1, 1, 10])
    ax.set_xticklabels(["0.1%", "1%", "10%"])
    ax.set_title("connectivity recovery\nin simulations", y=1.05,
                fontstyle="italic", x=-0.38, loc="left")
    ax.set_ylabel("odds of true connection\nvs chance")
    #ax.set_yticks([1, 10, 100, 1000])
    #ax.set_yticklabels(["1", "10", "100", "1,000"])
    ax.set_xlabel("top % of correlations")
    il = plot_label(ltr, il, ax, transl)

    transl = mtransforms.ScaledTranslation(-40 / 72, 18 / 72, fig.dpi_scale_trans)
    ax = plt.subplot(grid[2, 3])
    pos = ax.get_position().bounds
    ax.set_position([pos[0]+0.025, pos[1]+0.02, pos[2], pos[3]])
    dbin_strong_all = dat_local["dbin_strong_all"]
    dbin_other_all = dat_local["dbin_other_all"]
    dbin_strong = dbin_strong_all.mean(axis=-2).copy()
    dbin_other = dbin_other_all.mean(axis=-2).copy()
    pstrong = dat_local["pstrong"]
    nn = dbin_strong_all.shape[-2]
    nstrong = int(np.round(pstrong * nn))
    nother = nn - nstrong
    dnorm_sim = dbin_strong / nstrong / (dbin_other / nother)
    for i, sparsity in enumerate(sparsities):
        ax.plot(bcent, dnorm_sim[:,i].mean(axis=0),
                color=colors[i], lw=1)
    ax.plot(bcent, np.ones(len(bcent)), color="r", ls="--", lw=2)
    ax.set_ylabel("strong pair odds\nvs chance")
    ax.set_xlabel("distance ($\mu$m)")
    ax.set_title("simulation", fontsize="medium", loc="center", x=0.5)
    ax.text(-0.25, 1.25, "spatial distribution of strong pairs", fontsize="large", 
            fontstyle="italic", transform=ax.transAxes)
    il = plot_label(ltr, il, ax, transl)
    #ax.set_ylim([0, 0.16])
    sr_sim = dnorm_sim[:,:,:1].mean(axis=-1) / dnorm_sim[:,:,-4:].mean(axis=-1)
    print(sr_sim.shape)
    axin = ax.inset_axes([0.8, 0.45, 0.5, 0.5])
    for i in range(sr_sim.shape[1]):
        axin.scatter(sparsities[i], sr_sim[:,i].mean(axis=0), 
                      color=colors[i], marker="o", s=10)
    axin.set_xlabel("p(local) /\n p(global)", fontsize="x-small")
    axin.set_ylabel("strong pair odds\n(near vs far)", fontsize="x-small")
    #axin.fill_between([sparsities[0], sparsities[-1]], (5.85 - 1.7)*np.ones(2), 
    #                  (5.85 + 1.7) * np.ones(2), color=colors_data[3], alpha=0.25)
    axin.plot([sparsities[0], sparsities[-1]], 5.85*np.ones(2), color=colors_data[3], 
              lw=2, ls="--")
    axin.text(0.65, 0.5, "data", fontsize="small", color=colors_data[3], 
              fontstyle="italic", transform=axin.transAxes)
    axin.set_yscale("log")
    axin.set_xscale("log")
    axin.set_yticks([1, 10])
    axin.set_yticklabels(["1", "10"], fontsize="x-small")
    axin.set_xticks([1, 100])
    axin.set_xticklabels(["1", "100"], fontsize="x-small")
    ax.xaxis.set_minor_locator(locmin)
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        
    dnorm_data = np.zeros((len(dat_neural["dbin_strong_all"]), len(bcent)))
    ax = plt.subplot(grid[2, 4])
    pos = ax.get_position().bounds
    ax.set_position([pos[0]+0.025, pos[1]+0.02, pos[2], pos[3]])
    dbin_strong_all = dat_neural["dbin_strong_all"]
    dbin_other_all = dat_neural["dbin_other_all"]
    for i in range(len(dbin_strong_all)):
        dbin_strong = dbin_strong_all[i].mean(axis=0).copy() 
        dbin_other = dbin_other_all[i].mean(axis=0).copy()
        nn = dbin_strong_all[i].shape[-2]
        nstrong = int(np.round(pstrong * nn))
        nother = nn - nstrong
        dnorm = dbin_strong / nstrong / (dbin_other / nother)
        dnorm_data[i] = dnorm[:len(bcent)]
        ax.plot(bcent, dnorm_data[i], 
                color=colors_data[i], lw=0.5+0.1*i, zorder=30-i)
    ax.set_xlabel("distance ($\mu$m)")
    ax.set_title("neural data", fontsize="medium", loc="center", x=0.5)
    ax.plot(bcent, np.ones(len(bcent)), color="r", ls="--", lw=2)
    sratio_data = dnorm_data[:, :1].mean(axis=-1) / dnorm_data[:, -4:].mean(axis=-1)
    
    transl = mtransforms.ScaledTranslation(-44 / 72, 20 / 72, fig.dpi_scale_trans)
    grid1 = matplotlib.gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=grid[-1, :],
                                                            wspace=0.05, hspace=0.)      

    titles = [f"p(local) / p(global) = {int(sparsities[3])}", 
              f"p(local) / p(global) = {int(sparsities[6])}",
              "mouse V1 recording"]
    jex = [3, 6, 12]
    for j in range(3):
        grid2 = matplotlib.gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=grid1[j],
                                                            wspace=0.15, hspace=0.2)
        if j < 2:
            ypos = ypos_local_all[jex[j]]
            xpos = xpos_local_all[jex[j]]
            U = pcs_local_all[jex[j]].copy()
            vmax = 0.03
        else:
            U = pcs_all[jex[j]].copy()
            ypos = ypos_all[jex[j]]
            xpos = xpos_all[jex[j]]
        vmax = 0.05
        np.random.seed(1)
        irand = np.random.randint(0, U.shape[0], 10000)
        #U -= U.mean(axis=0)
        U /= (U[irand]**2).sum(axis=0)**0.5
        from scipy.stats import skew
        U /= np.sign(skew(U, axis=0))
        for k in range(3):
            ax = plt.subplot(grid2[k])
            pos = ax.get_position().bounds
            ax.set_position([pos[0] - (j==0)*0.05 + (j==2)*0.035, pos[1]+0.04, pos[2], pos[2]*yratio])
            # ax.set_position([pos[0] - j*0.01 - 0.01, 
            #                  pos[1]-0.01*(k//2) - 0.02, 
            #                  pos[3], pos[3]*yratio])
            im = ax.scatter(ypos[irand], xpos[irand], c=U[irand,k], 
                            vmin=-vmax, vmax=vmax, cmap="bwr", s=1,
                            rasterized=True)
            ax.axis("off")
            yrange = [0, max(ypos.max(), xpos.max())]
            ax.set_ylim(yrange); ax.set_xlim(yrange)
            ax.text(0.5, 1.1, f"PC {k+1}", fontsize="small", ha="center", va="center",
                    transform=ax.transAxes)
            if k==0 and j==0:
                ax.text(0.1, 1.55, "spatial distribution of principal components", 
                            fontsize="large", fontstyle="italic", transform=ax.transAxes)
                transl = mtransforms.ScaledTranslation(-10 / 72, 27 / 72, fig.dpi_scale_trans)
                il = plot_label(ltr, il, ax, transl)
            elif k==0 and j==2:
                cax = ax.inset_axes([-0.35, 0.7, 0.35, 0.35])
                cax.plot([-1, 1], [0, 0], color="k", lw=1)
                cax.plot([0, 0], [-1, 1], color="k", lw=1)
                cax.set_ylim([-1, 1]); cax.set_xlim([-1, 1])
                cax.text(0.5, 1.3, "A", fontsize="small", ha="center", va="center",
                         transform=cax.transAxes)
                cax.text(1.3, 0.5, "P", fontsize="small", ha="center", va="center",
                         transform=cax.transAxes)
                cax.text(0.5, -0.3, "L", fontsize="small", ha="center", va="center",
                         transform=cax.transAxes)
                cax.text(-0.3, 0.5, "M", fontsize="small", ha="center", va="center",
                         transform=cax.transAxes)
                cax.axis("off")
            elif k==1:
                ax.set_title(titles[j], y=1.18, x=0.5, loc="center",
                             fontsize="medium")
            elif k==2 and j==0:
                cax = ax.inset_axes([1.07, 0.55, 0.07, 0.45])
                cb = plt.colorbar(im, cax=cax)
                cb.ax.tick_params(labelsize="small")
                cb.set_ticks([-vmax, vmax])

    return fig
    