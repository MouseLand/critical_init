from matplotlib.patches import Ellipse
from fig_utils import *
from scipy.stats import zscore
from powerlaw import fit_powerlaw_exp

def plot_spectrum(ax, ss, color=0.5*np.ones(3), plot_fit=True, lw=3):
    alpha, yp = fit_powerlaw_exp(ss[:1000], np.arange(10, 500))
    ss /= yp[0]
    yp /= yp[0]
    ax.loglog(np.arange(1, 1001), ss[:1000], lw=lw, color=color)
    if plot_fit:
        ax.loglog(np.arange(1, 1001), yp , lw=lw*0.5, color=0.*np.ones(3), ls="--")
    ax.set_ylim(0.002, 2)
    ax.set_xlim(1, np.exp(np.log(2) - np.log(0.002)))
    ax.set_xticks([1, 10, 100, 1000])
    ax.set_xticklabels(["1", "10", "100", "1,000"])
    ax.set_yticks([0.01, 0.1 ,1])
    ax.set_yticklabels(["0.01", "0.1", "1"])
    return alpha

def fig1(dat):
    Asub = dat["Asym_ex"].copy()
    Apos = Asub.copy() - Asub.min()
    Aasym = dat["Anonsym_ex"].copy()
    evals_asym = dat["enonsym_ex"].copy()
    evals_cov_asym = dat["evals_nonsym"].copy()
    evals_cov_sym = dat["evals_sym"].copy()
    evals_sym = dat["esym_ex"].copy()
    Xt = dat["Xt"].copy()

    fig = plt.figure(figsize=(14,5.5))
    yratio = 14/5.5
    grid = plt.GridSpec(2, 7, wspace=0.1, hspace=0.5, figure=fig, 
                            bottom=0.07, top=0.93, left=0.05, right=0.95)

    transl = mtransforms.ScaledTranslation(-35 / 72, 20 / 72, 
                                           fig.dpi_scale_trans)
    il = 0
    ax = plt.subplot(grid[0, :])
    pos = ax.get_position().bounds
    ax.set_position([-0.02, pos[1]-0.05, 1.02, 1-pos[1]+0.08])
    ax.add_patch(plt.Rectangle((0, 0), 1, 1, color=0.9*np.ones(3)))
    ax.axis("off")
    ax = plt.subplot(grid[1, -3:])
    pos = ax.get_position().bounds
    ax.set_position([pos[0]-0.06, -0.05, 1-pos[0]+0.06, 0.6])
    ax.add_patch(plt.Rectangle((0, 0), 1, 1, color=0.9*np.ones(3)))
    ax.axis("off")

    dy = 0.03
    nplot = 20
    ax = plt.subplot(grid[0, 0])
    pos = ax.get_position().bounds
    ax.set_position([pos[0], pos[1]-dy, pos[2], pos[3]])
    ax.spines["bottom"].set_visible(False)
    ax.spines["top"].set_visible(True)
    vmax = 0.03
    Apos -= np.diag(np.diag(Apos))
    im = ax.imshow(Apos[:nplot, :nplot], cmap="bwr", vmin=-vmax, vmax=vmax)
    # unicode arrow
    ax.set_ylabel(r"$\leftarrow$ neurons")
    ax.set_title(r"neurons $\rightarrow$", fontsize="medium", loc="center")
    ax.set_xticks([])
    ax.set_yticks([])
    cax = ax.inset_axes([1.05, 0.75, 0.05, 0.25])
    cb = plt.colorbar(im, cax=cax)
    cb.set_ticks([-0.03, 0.03])
    ax.text(-0.14, 1.28, "symmetric connectivity matrix", transform=ax.transAxes, 
            ha="left", va="top", fontsize="large")
    il = plot_label(ltr, il, ax, transl)

    ax = plt.subplot(grid[0, 1])
    pos = ax.get_position().bounds 
    ax.set_position([pos[0]+0.15*pos[2], pos[1]-dy, pos[2]*0.7, pos[3]])
    ax.annotate("", xy=(0., 0.5), xytext=(1, 0.5), arrowprops=dict(arrowstyle="<-"))
    ax.text(0.5, 0.58, "subtract mean", fontsize="large", transform=ax.transAxes,
            ha="center", va="center", fontstyle="italic")
    ax.text(0.5, 0.35, '("inhibitory\nstabilization")', fontsize="large", 
            transform=ax.transAxes, ha="center", va="center", fontstyle="italic")
    ax.axis("off")
    ax.set_ylim([0, 1])
    ax.set_xlim([0, 1])

    ax = plt.subplot(grid[0, 2])
    pos = ax.get_position().bounds
    ax.set_position([pos[0]+0.*pos[2], pos[1]-dy, pos[2], pos[3]])
    ax.spines["bottom"].set_visible(False)
    ax.spines["top"].set_visible(True)
    vmax = 0.015
    im = ax.imshow(Asub[:nplot, :nplot], cmap="bwr", vmin=-vmax, vmax=vmax)
    ax.set_xticks([]); ax.set_yticks([])
    cax = ax.inset_axes([1.05, 0.75, 0.05, 0.25])
    cb = plt.colorbar(im, cax=cax)
    cb.set_ticks([-0.01, 0.01])
    ax.set_title(r"neurons $\rightarrow$", fontsize="medium", loc="center")
    ax.set_ylabel(r"$\leftarrow$ neurons")
    ax.text(-0.1, 1, "A  = ", fontsize="xx-large", transform=ax.transAxes, 
            ha="right", va="center")
    pos0 = ax.get_position().bounds
    il = plot_label(ltr, il, ax, transl)

    ax = plt.subplot(grid[0, 3])
    pos = ax.get_position().bounds 
    ax.set_position([pos[0]+0.15*pos[2], pos[1]-dy, pos[2]*0.7, pos[3]])
    ax.annotate("", xy=(0., 0.5), xytext=(1, 0.5), arrowprops=dict(arrowstyle="<-"))
    ax.text(0.5, 0.58, "dynamics",fontsize="large", transform=ax.transAxes,
            ha="center", va="center", fontstyle="italic")
    ax.axis("off")
    ax.set_ylim([0, 1])
    ax.set_xlim([0, 1])

    ax = plt.subplot(grid[0, 4])
    pos = ax.get_position().bounds
    ax.set_position([pos[0], pos0[1], pos[2], pos[2]*yratio])
    ax.plot(Xt[5:8, :50].T)
    tstr = r"$\tau \dot{x} = -x + A x + \epsilon_t$"
    tstr += "\n"
    tstr += r"$\epsilon_t \sim N(0, I)$"
    ax.text(0.5, 1, tstr, fontsize="large", transform=ax.transAxes,
            ha="center", va="bottom")#, va="center")
    ax.set_ylabel(r"$x_i(t)$")
    ax.set_yticks([])
    ax.set_xticks([])
    ax.set_xlabel(r"time $t$")
    il = plot_label(ltr, il, ax, transl)

    ax = plt.subplot(grid[0, 5])
    pos = ax.get_position().bounds 
    ax.set_position([pos[0]+0.1*pos[2], pos[1]-dy, pos[2]*0.7, pos[3]])
    ax.annotate("", xy=(0., 0.5), xytext=(1, 0.5), arrowprops=dict(arrowstyle="<-"))
    ax.text(0.5, 0.65, "stationary\ndistribution", fontsize="large", transform=ax.transAxes,
            ha="center", va="center", fontstyle="italic")
    tstr = r"$(A-I)\Sigma$ + "
    tstr += "\n" 
    tstr += r"  $\Sigma(A-I)^\top$" 
    tstr += "\n" 
    tstr += r"$= -I$"
    ax.text(0.1, 0.3, tstr, fontsize="large", 
            transform=ax.transAxes, ha="left", va="center")
    ax.axis("off")
    ax.set_ylim([0, 1])
    ax.set_xlim([0, 1])

    ax = plt.subplot(grid[0, 6])
    pos = ax.get_position().bounds
    ax.set_position([pos[0], pos0[1], pos[2], pos[2]*yratio])
    for size in [0.0625, 0.125, 0.25][::-1]:
        ax.add_patch(Ellipse((0.5, 0.5), size, 2*size, angle=-45, 
                             facecolor=[0.9, 0.9, 1], edgecolor="k", 
                             lw=1, ls="--"))
    ax.set_xlim([0.2, 0.8])
    ax.set_ylim([0.2, 0.8])
    ax.set_xlabel(r"$x_i$")
    ax.set_ylabel(r"$x_j$")
    ax.set_yticks([]); ax.set_xticks([])
    tstr = r"covariance matrix"
    tstr += "\n"
    tstr += r"$\Sigma = \frac{1}{2}(I - A)^{-1}$"
    ax.set_title(tstr, fontsize="large", loc="center", y=1)
    il = plot_label(ltr, il, ax, transl)

    ax = fig.add_axes([pos[0], pos0[1]-0.14, pos[2], 0.08])
    ax.annotate("", xy=(0.5, 1.), xytext=(0.5, 0), arrowprops=dict(arrowstyle="<-"))
    ax.set_ylim([0, 1])
    ax.set_xlim([0, 1])
    ax.axis("off")

    il = 6
    ax = plt.subplot(grid[1, 0])
    ax.spines["bottom"].set_visible(False)
    ax.spines["top"].set_visible(True)
    vmax = 0.03
    ax.imshow(Aasym[:nplot, :nplot], cmap="bwr", vmin=-vmax, vmax=vmax)
    ax.set_title(r"neurons $\rightarrow$", fontsize="medium", 
                 loc="center", y=1.01)
    ax.set_ylabel(r"$\leftarrow$ neurons")
    ax.set_xticks([]); ax.set_yticks([])
    pos0 = ax.get_position().bounds
    ax.text(-0.14, 1.28, "non-symmetric connectivity matrix", transform=ax.transAxes,
            ha="left", va="top", fontsize="large")
    il = plot_label(ltr, il, ax, transl)

    axin = ax.inset_axes([1.5, 0.75, 0.35, 0.35])
    irand = np.random.permutation(len(evals_asym))[:1000]
    re = np.real(evals_asym)[irand]
    im = np.imag(evals_asym)[irand]
    axin.scatter(re, im, s=1, color=dcolors[3], alpha=0.5, 
                 rasterized=True)
    axin.set_ylabel(r"Im($\lambda_A$)")
    axin.set_xlabel(r"Re($\lambda_A$)")
    #axin.set_title(r"$\lambda_A$", loc="center")

    ax = plt.subplot(grid[1, 1])
    pos = ax.get_position().bounds
    ax.set_position([pos[0]+0.15*pos[2], pos0[1]-0.05, pos[2]*0.7, pos[3]])
    ax.annotate("", xy=(0., 0.5), xytext=(1, 0.5), arrowprops=dict(arrowstyle="<-"))
    ax.set_ylim([0, 1])
    ax.set_xlim([0, 1])
    ax.axis("off")

    transl = mtransforms.ScaledTranslation(-48 / 72, 10 / 72, 
                                           fig.dpi_scale_trans)
    
    ax = plt.subplot(grid[1, 2])
    ss = evals_cov_asym.copy()
    pos = ax.get_position().bounds
    ax.set_position([pos[0]+0.2*pos[2], pos0[1], pos[2], pos[2]*yratio])
    alpha = plot_spectrum(ax, ss)
    ax.set_ylabel("normalized variance")
    ax.set_xlabel(r"PC index")
    ax.set_title("eigenvalues of $\Sigma$", fontsize="large")
    astr = rf"$1 / n^\alpha$, $\alpha$ = {alpha:.3f}"
    ax.text(0.7, 0.8, astr, transform=ax.transAxes,
            fontsize="large", ha="center", va="center")
    il = plot_label(ltr, il, ax, transl)

    il = 4
    ax = plt.subplot(grid[1, 4])
    pos = ax.get_position().bounds
    ax.set_position([pos[0], pos0[1], pos[2], pos[2]*yratio])
    ax.hist(evals_sym, bins=50, color=dcolors[4], density=True)
    ax.set_ylabel("density")
    ax.set_xlabel(r"$\lambda$")
    ax.set_title(r"eigenvalues of $A$", fontsize="large")
    il = plot_label(ltr, il, ax, transl)
    
    ax = plt.subplot(grid[1, 5])
    pos = ax.get_position().bounds
    ax.set_position([pos[0]+0.0*pos[2], pos0[1], pos[2]*0.6, pos[3]])
    ax.annotate("", xy=(0., 0.5), xytext=(1, 0.5), arrowprops=dict(arrowstyle="<-"))
    ax.text(0.5, 0.65, r"$\lambda_\Sigma = \frac{1}{2(1 - \lambda_A)}$", fontsize="x-large", transform=ax.transAxes,
            ha="center", va="center")   
    ax.set_ylim([0, 1])
    ax.set_xlim([0, 1])
    ax.axis("off")

    ax = plt.subplot(grid[1, 6])
    pos = ax.get_position().bounds
    ax.set_position([pos[0], pos0[1], pos[2], pos[2]*yratio])
    alpha = plot_spectrum(ax, evals_cov_sym, color=dcolors[4])
    ax.set_title("eigenvalues of $\Sigma$", fontsize="large")
    ax.set_ylabel("normalized variance")
    ax.set_xlabel(r"PC index")
    astr = rf"$1 / n^\alpha$, $\alpha$ = {alpha:.3f}"
    ax.text(0.6, 0.85, astr, transform=ax.transAxes,
            fontsize="large", ha="center", va="center")
    il = plot_label(ltr, il, ax, transl)

    return fig


def suppfig_probs(dat):
    fig = plt.figure(figsize=(9.33, 4.2))
    yratio = 9.33/4.2
    grid = plt.GridSpec(2, 4, wspace=0.75, hspace=0.3, figure=fig, 
                        bottom=0.05, top=0.95, left=0.09, right=0.96)
    transl = mtransforms.ScaledTranslation(-30 / 72, 17 / 72, fig.dpi_scale_trans)
    il = 0
    Aexs = dat["Aexs"]
    evals_all = dat["evals_all"]
    titles = dat["distributions"]
    for k in range(4):
        ax = plt.subplot(grid[0, k])
        pos = ax.get_position().bounds
        ax.set_position([pos[0], pos[1]-0.02, pos[2], pos[2]*yratio])
        conn_panel(ax, Aexs[k], np.arange(25), vmax=0.02 if k>0 else 0.005,
                    title=titles[k], xt=-0.15, colorbar=k<2)
        pos0 = ax.get_position().bounds
        il = plot_label(ltr, il, ax, transl)

        ax = plt.subplot(grid[1, k])
        pos = ax.get_position().bounds
        ax.set_position([pos0[0], pos[1]+0.07, pos0[2], pos0[3]])
        alphas = np.zeros(len(evals_all))
        for i in range(evals_all.shape[0]):
            evals = evals_all[i, k].copy()
            alphas[i], yp = fit_powerlaw_exp(evals, np.arange(10, 500))
            ax.loglog(np.arange(1, len(evals)+1), evals / yp[0], 
                      color=dcolors[4], lw=0.5)
        ax.set_ylim(0.001, 3)
        ax.set_xlim(1, 3000)
        ax.set_yticks([0.001, 0.01, 0.1, 1])
        ax.set_yticklabels(["0.001", "0.01", "0.1", "1"])
        ax.set_xticks([1, 10, 100, 1000])
        ax.set_xticklabels(["1", "10", "100", "  1,000"])
        ax.set_xlabel("PC index")
        ax.text(0.5, 0.7, f"$\\alpha$ = {alphas.mean():.3f}", transform=ax.transAxes)
        if k==0:
            ax.set_ylabel("normalized variance")
        ax.xaxis.set_minor_locator(locmin)
        ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())

    return fig

def suppfig_sim(dat, save_fig=True):
    fig = plt.figure(figsize=(9.333, 5), dpi=150)
    yratio = 9.333 / 5
    grid = plt.GridSpec(2, 4, wspace=0.4, hspace=0.1, figure=fig, 
                        bottom=0.11, top=1, left=0.09, right=0.98)
    transl = mtransforms.ScaledTranslation(-45 / 72, 7 / 72, 
                                           fig.dpi_scale_trans)
    il = 0
    ax = plt.subplot(grid[0, 0])
    pos = ax.get_position().bounds
    ax.set_position([pos[0], pos[1], pos[2], pos[2]*yratio])
    nonsyms= dat["nonsyms"]
    evals_all = dat["evals_all"]
    Xembs = [dat["Xsym_ex"], dat["Xnonsym_ex"]]
    print(evals_all.shape)
    n_sim = evals_all.shape[0]
    colors = np.linspace(0, 0.8, len(nonsyms))[:,np.newaxis] * np.ones((1,3))
    alphas_all = np.zeros(evals_all.shape[:2])
    for i in range(n_sim):
        for j in range(len(nonsyms)):
            evals = evals_all[i, j].copy()
            alphas_all[i,j], yp = fit_powerlaw_exp(evals, np.arange(10, 500))
            evals /= yp[0]
            ax.loglog(np.arange(1, len(evals)+1), evals, color=colors[j], 
                      lw=0.75, alpha=1)
    ax.set_ylim(0.001, 10)  
    ax.set_xlim(1, 10000)
    ax.set_yticks([0.001, 0.1, 10])
    ax.set_yticklabels(["0.001", "0.1", "10"])
    ax.set_xticks([1, 100, 10000])
    ax.set_xticklabels(["1", "100", "10,000"])
    ax.minorticks_on()
    ax.set_xlabel("PC index")
    ax.set_title("eigenspectrum", fontstyle="italic", loc="left", 
                 x=-0.24, y=1.03)
    ax.xaxis.set_minor_locator(locmin)
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    il = plot_label(ltr, il, ax, transl)
    lbls = ["symmetric", "1/3 non-symm.", "2/3 non-symm.", "nonsymmetric"]
    for j in range(len(nonsyms)):
        ax.text(0.5, 0.95 - 0.12*j, lbls[j], color=colors[j], 
                transform=ax.transAxes, ha="left")
    ax.text(-0.3, 0.5, "    normalized variance", rotation=90, ha="center", 
            va="center", transform=ax.transAxes)

    ax = plt.subplot(grid[1, 0])
    pos = ax.get_position().bounds
    ax.set_position([pos[0], pos[1], pos[2], pos[2]*yratio])
    xp = np.arange(len(nonsyms))*np.ones((n_sim,1))
    xp += np.random.randn(*xp.shape)*0.05
    ax.scatter(xp.flatten(), alphas_all.flatten(), 
               color=(colors[np.newaxis,:,:]*np.ones((n_sim,1,1))).reshape(-1,3), s=10)
    ax.scatter(np.arange(len(nonsyms)), alphas_all.mean(axis=0), color=colors, 
               s=400, marker="_")
    print(alphas_all.mean(axis=0))
    ax.set_ylabel("power-law exponent ($\\alpha$)")
    ax.set_xticks(np.arange(len(nonsyms)))
    ax.set_xticklabels(["symm.", "1/3 non-symm.", "2/3 non-symm.", "non-symm."], 
                       rotation=20, ha="right")
    pos = ax.get_position().bounds

    transl = mtransforms.ScaledTranslation(-17 / 72, 3 / 72, 
                                           fig.dpi_scale_trans)
    for d in range(2):
        ax = plt.subplot(grid[d, 1:])
        pos1 = ax.get_position().bounds
        ax.set_position([pos1[0], pos1[1], pos1[2], pos[3]])
        vmax = 1.5
        im = ax.imshow(zscore(Xembs[d][:,:8000], axis=1), cmap="gray_r", 
                  vmin=0, vmax=vmax, aspect="auto")
        ax.plot([0, 22*30], -10*np.ones(2), color="k", lw=1.5)
        ax.plot(-80*np.ones(2), [0, 40], color="k", lw=1.5)
        if d==0:
            ax.text(22*30/2, -40, "30 sec.", ha="center", va="center", fontsize="small")
            ax.text(-200, 0, "1000 neurons", ha="center", va="bottom", fontsize="small", rotation=90)
        ax.set_ylim([-14, Xembs[d].shape[0]+0.5])
        ax.set_xlim([-100, 8000])
        ax.axis("off")
        ax.set_title(["symmetric connectivity rastermap", "non-symmetric connectivity rastermap"][d], 
                     fontsize="large", fontstyle="italic", loc="left", x=0.0)
        if d==0:
            cax = ax.inset_axes([0.8, -0.04, 0.1, 0.03])
            cb = plt.colorbar(im, cax=cax, orientation="horizontal", 
                              label="z-scored activity")
            cb.ax.tick_params(labelsize="small")
        il = plot_label(ltr, il, ax, transl)
    return fig


def suppfig_tbin(dat):
    titles = ["symmetric", "non-symmetric"]
    nt = (200*(60000-4000))//23
    fig = plt.figure(figsize=(7, 2.3), dpi=150)
    yratio = 7/2.3
    grid = plt.GridSpec(1, 3, wspace=0.6, hspace=0.65, figure=fig,
                        bottom=0.05, top=0.95, left=0.1, right=0.95)
    evals_bin_all = dat["evals_bin_all"]
    tbins = dat["tbins"]
    alphas_all = np.zeros(evals_bin_all.shape[:3])
    colors = [[0, 0.5, 0], [0.7, 0.6, 0.9]]
    dy = 0.2
    il = 0
    transl = mtransforms.ScaledTranslation(-50 / 72, 15 / 72, fig.dpi_scale_trans)
    for k in range(2):
        if k==0:
            lcolors = plt.get_cmap("YlGn")(np.linspace(0.5, 1, len(tbins)))
        else:
            lcolors = plt.get_cmap("Purples")(np.linspace(0.5, 1, len(tbins)))
        ax = plt.subplot(grid[0, k])
        pos = ax.get_position().bounds
        ax.set_position([pos[0], pos[1]+dy, pos[2]*0.9, pos[2]*yratio*0.9])
        for i, tbin in enumerate(tbins):
            evals = evals_bin_all[:,k,i].copy().mean(axis=0)
            ntmax = min(nt // tbin - 30, len(evals))
            #print(ntmax)
            alpha, yp = fit_powerlaw_exp(evals[:ntmax], 
                                                       np.arange(10, min(ntmax, 500)))
            ax.loglog(np.arange(1, ntmax+1), evals[:ntmax]/yp[0], 
                      color=lcolors[i], lw=1, zorder=i)
            if i%4==0 and k==0:
                ax.text(0.65, 0.94 - 0.1*i/4,  f"{tbin*23.*2:.0f} ms",
                        color=lcolors[i], transform=ax.transAxes, fontsize="small")
            for j in range(evals_bin_all.shape[0]):
                evals = evals_bin_all[j,k,i].copy()
                alphas_all[j, k, i], yp = fit_powerlaw_exp(evals[:ntmax],
                                                                        np.arange(10, min(ntmax, 500)))
        if k==0:
            ax.text(0.64, 0.94, "time bin = ", transform=ax.transAxes, 
                    ha="right")
        ax.set_ylim(0.001, 3)
        ax.set_xlim(1, 1000)
        ax.set_yticks([0.01, 0.1, 1])
        ax.set_yticklabels(["0.01", "0.1", "1"])
        ax.set_xticks([1, 10, 100, 1000])
        ax.set_xticklabels(["1", "10", "100", "1,000"])
        ax.set_xlabel("PC index")
        ax.set_ylabel("normalized variance")
        ax.xaxis.set_minor_locator(locmin)
        ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.set_title(titles[k], color=colors[k], fontweight="bold", 
                     fontsize="medium", y=1.1)
        il = plot_label(ltr, il, ax, transl)

    ax = plt.subplot(grid[0, -1])
    pos = ax.get_position().bounds
    ax.set_position([pos[0]+0.01, pos[1]+dy, pos[2]*0.9, pos[2]*yratio*0.9])
    nbins = len(tbins)
    #xp = np.arange(2)*np.ones((nbins,1))
    for k in range(2):
        print(alphas_all[:,k].mean(axis=0))
        ax.errorbar(tbins * 2 * 23/1000, alphas_all[:,k].mean(axis=0), 
                    alphas_all[:,k].std(axis=0), color=colors[k], lw=1)
    ax.set_xscale("log")
    ax.set_ylabel("power-law exponent ($\\alpha$)")
    ax.set_xlabel("time bin (sec.)")
    ax.set_ylim([0, 2])
    ax.set_xticks([0.1, 1, 10])
    ax.set_xticklabels(["0.1", "1", "10"])
    il = plot_label(ltr, il, ax, transl)

    return fig