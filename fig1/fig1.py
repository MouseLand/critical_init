from matplotlib.patches import Ellipse
from fig_utils import *
from scipy.stats import zscore
from powerlaw import fit_powerlaw_exp


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
    alpha = plot_spectrum(ax, ss)[0]
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
    alpha = plot_spectrum(ax, evals_cov_sym, color=dcolors[4])[0]
    ax.set_title("eigenvalues of $\Sigma$", fontsize="large")
    ax.set_ylabel("normalized variance")
    ax.set_xlabel(r"PC index")
    astr = rf"$1 / n^\alpha$, $\alpha$ = {alpha:.3f}"
    ax.text(0.6, 0.85, astr, transform=ax.transAxes,
            fontsize="large", ha="center", va="center")
    il = plot_label(ltr, il, ax, transl)

    return fig


def suppfig_poisson(dat2):
    il = 0
    fig = plt.figure(figsize=(14,6))
    yratio = 14/6
    grid = plt.GridSpec(2, 6, wspace=0.4, hspace=0.65, figure=fig, 
                            bottom=0.09, top=0.91, left=0.05, right=0.95)

    transl = mtransforms.ScaledTranslation(-48 / 72, 5 / 72, 
                                           fig.dpi_scale_trans)
    
    ylims = [[0.3, 1.2], [0.3, 1.5], [0.5, 1.8], [0.5, 1.8]]
    
    colors = ['g', 'y', 'b']

    evals_concat = [dat2['evals_gt_all'], dat2['evals_all'], dat2['evals_svca_all'], dat2['evals_svca2_all']]

    ymax = 500

    lbls = ["symmetric", "1/3 non-symm.", "2/3 non-symm.", "nonsymmetric"]
    ntype = ['2p data', 'poisson noise']

    transl = mtransforms.ScaledTranslation(-48 / 72, 2 / 72, fig.dpi_scale_trans)
    
    grid1 = matplotlib.gridspec.GridSpecFromSubplotSpec(1, 5, subplot_spec=grid[-1, :],
                                                            wspace=0.7, hspace=0.3)

    colors = ['g', 'y', 'b']

    evals_concat = [dat2['evals_gt_all'], dat2['evals_all'], dat2['evals_svca_all'], dat2['evals_svca2_all']]

    ymax = 500

    for ps in [0, 1]:
        grid1 = matplotlib.gridspec.GridSpecFromSubplotSpec(1, 5, subplot_spec=grid[ps, :],
                                                            wspace=0.7, hspace=0.3)
        ni = 0
        nl = 5 if ps==0 else 1
        ss_gt = dat2['evals_gt_all'][:1, ni].mean(axis=0)
        alpha_gt, yp = fit_powerlaw_exp(ss_gt, np.arange(10, min(len(ss_gt)-50, ymax)))
        ss_gt /= yp[0]
        ax = plt.subplot(grid1[0, 0])
        pos = ax.get_position().bounds
        ax.set_position([pos[0] + 0.02, *pos[1:]])    
        hs = []
        h = ax.loglog(np.arange(1, len(ss_gt)+1), ss_gt, color='k', lw=3, ls=':')
        hs.append(h)
        for k in range(3):
            ss = evals_concat[k+1][:1, ni, nl, -1, 0].mean(axis=0)
            alpha, h = plot_spectrum(ax, ss, color=colors[k], plot_fit=False, ymax=ymax, lw=2)
            hs.append(h)
        ax.set_ylabel("normalized variance")
        ax.set_xlabel(r"PC index")
        if ni==0:
            print(hs)
            leg = ax.legend([h[0] for h in hs], ['ground-truth', 'direct', 'SVCA', 'SVCA2'],
                    loc='upper left', bbox_to_anchor=(0.6, 1.2), frameon=False)
            for txt, col in zip(leg.get_texts(), ['k'] + colors):
                txt.set_color(col)
        ax.text(-0.24, 1.3, f"simulated {ntype[ps]} (symmetric case)", transform=ax.transAxes,
                ha="left", va="top", fontsize="large", fontstyle="italic")
        il = plot_label(ltr, il, ax, transl)

        n_sim = len(evals_concat[0])
        alphas = np.zeros((n_sim, 4, 3))
        for nl in range(3,6) if ps==0 else range(0,3):
            for k in range(4):
                for i in range(n_sim):
                    if k > 0:
                        ss = evals_concat[k][i, ni, nl, -1, 0]
                    else:
                        ss = evals_concat[k][i, ni]
                    alphas[i,k,nl-3], yp = fit_powerlaw_exp(ss, np.arange(10, min(len(ss)//2, ymax)))
        
        alpha_gt = alphas[:,0,0].mean()
        ylim = [0.3, 1.4]
        for j in range(2):
            var = dat2['ntimes']/(60*23) if j==0 else dat2['nneurons']
            ax = plt.subplot(grid1[0, 1 + j])
            pos = ax.get_position().bounds
            ax.set_position([pos[0] + 0.04*(j==0) + 0.035, *pos[1:]])
            ax.plot(var, alpha_gt * np.ones(len(var)), color='k', lw=3, ls=':')
            for k in range(3):
                alpha0 = np.zeros((n_sim, len(var)))
                for i in range(n_sim):
                    for t in range(len(var)):
                        if j==0:
                            ss = evals_concat[k+1][i, ni, nl, -1, t]
                        else:
                            ss = evals_concat[k+1][i, ni, nl, t, 0]
                        if np.isnan(ss).sum() > 0:
                            ss = ss[:np.nonzero(np.isnan(ss))[0][0]]
                        alpha0[i, t], yp = fit_powerlaw_exp(ss, np.arange(10, min(len(ss)//2, ymax)))
                ax.errorbar(var, alpha0.mean(axis=0), alpha0.std(axis=0) / (n_sim-1)**0.5, color=colors[k])
            ax.set_ylim(ylim)
            ax.set_xlabel('duration (min.)' if j==0 else '# of neurons')
            if j==0:
                ax.set_ylabel(r'power law exponent ($\alpha$)')
                transl = mtransforms.ScaledTranslation(-50 / 72, 2 / 72, fig.dpi_scale_trans)
                il = plot_label(ltr, il, ax, transl)

        ax = plt.subplot(grid1[0, 3])
        pos = ax.get_position().bounds
        ax.set_position([pos[0] + 0.0, *pos[1:]])
        ax.plot([0, 2], alphas[:,0,0].mean(axis=0)*np.ones(2), lw=3, ls=':', color='k')
        print(alphas.mean(axis=0), alphas[:,-1, -1])
        for k in range(3):
            ax.errorbar(np.arange(3), alphas[:,k+1].mean(axis=0), alphas[:,k+1].std(axis=0) / (n_sim-1)**0.5, color=colors[k], lw=2)
        ax.set_ylim(ylim)
        ax.set_xticks([0, 1, 2])
        ax.set_xticklabels(['low', 'medium', 'high'])
        #ax.set_ylabel(r'power-law exponent ($\alpha$)')
        ax.set_xlabel('shot noise level' if ps==0 else 'poisson level')
        #il = plot_label(ltr, il, ax, transl)

        ax = plt.subplot(grid1[0, 4])
        alphas = np.zeros((n_sim, 4, 4))
        
        ylim = [0.5, 1.6]
        ax.plot(ylim, ylim, lw=2, color='k')
        markers = ['o', 's', 'x']
        for nl in range(3, 6) if ps==0 else range(0, 3):
            for ni in range(4):
                for k in range(4):
                    for i in range(n_sim):
                        if k>0:
                            ss = evals_concat[k][i, ni, nl, -1, 0]
                        else:
                            ss = evals_concat[k][i, ni]
                        alphas[i,k,ni], yp = fit_powerlaw_exp(ss, np.arange(10, min(len(ss)//2, ymax)))
                    if k > 0:
                        ax.scatter(alphas[:, 0, ni], alphas[:, k, ni], color=colors[k-1], 
                                s=10, alpha=0.5, marker=markers[nl-3], facecolors=colors[k-1] if nl==5 else 'none')
                        #ax.scatter(alphas[:, 0, ni].mean()*np.ones(n_sim) + np.random.randn(n_sim)*0.01, 
                        #           alphas[:, k, ni], color=colors[k-1], s=10)
        handles = [ax.scatter([], [], marker=m, color='k', facecolors='k' if i==2 else 'none',
                            edgecolors='k', s=40) for i, m in enumerate(markers)]
        leg = ax.legend(handles, ['low', 'medium', 'high'], loc='upper left', frameon=False,
                        handletextpad=0.2, handlelength=0.6, labelspacing=0.2, borderpad=0.2,
                        bbox_to_anchor=(0.0, 1.18))
        ax.set_xlabel(r'ground-truth $\alpha$')
        ax.set_ylabel(r'estimated $\alpha$')
        ax.set_xlim(ylim)
        ax.set_ylim(ylim)
        ticks = [0.5, 1.0, 1.5]
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        tick_labels = [f'{t:.1f}' for t in ticks]
        ax.set_xticklabels(tick_labels)
        ax.set_yticklabels(tick_labels)
        transl = mtransforms.ScaledTranslation(-48 / 72, 2 / 72, fig.dpi_scale_trans)
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

def suppfig_sim(dat, dat2, save_fig=True):
    fig = plt.figure(figsize=(9.333, 5), dpi=150)
    yratio = 9.333 / 5
    grid = plt.GridSpec(2, 4, wspace=0.4, hspace=0.1, figure=fig, 
                        bottom=0.12, top=1, left=0.09, right=0.98)
    transl = mtransforms.ScaledTranslation(-45 / 72, 7 / 72, 
                                           fig.dpi_scale_trans)
    il = 0
    ax = plt.subplot(grid[0, 0])
    pos = ax.get_position().bounds
    ax.set_position([pos[0], pos[1], pos[2], pos[2]*yratio])
    nonsyms= dat2["nonsyms"]
    evals_all = dat2["evals_gt_all"]
    Xembs = [dat["Xemb_sym"], dat["Xemb_nonsym"]]
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
        ax.set_position([pos1[0], pos1[1]-0.05*pos[3], pos1[2], pos[3]*1.1])
        vmax = 1.5
        xmax = 22*60*3//7
        im = ax.imshow(zscore(Xembs[d][:,:xmax], axis=1), cmap="gray_r", 
                  vmin=0, vmax=vmax, aspect="auto")
        ax.plot([0, 22*30//7], -10*np.ones(2), color="k", lw=1.5)
        ax.plot(-5*np.ones(2), [0, 50], color="k", lw=1.5)
        if d==0:
            ax.text(22*30/7/2, -40, "30 sec.", ha="center", va="center", fontsize="small")
            ax.text(-10, 0, "1000 neurons", ha="center", va="bottom", fontsize="small", rotation=90)
        ax.set_ylim([-12, Xembs[d].shape[0]+0.5])
        ax.set_xlim([-10, xmax])
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
    #nt = (200*(60000-4000))//23
    fig = plt.figure(figsize=(7, 2.3), dpi=150)
    yratio = 7/2.3
    grid = plt.GridSpec(1, 3, wspace=0.6, hspace=0.65, figure=fig,
                        bottom=0.05, top=0.95, left=0.1, right=0.95)
    evals_bin_all = dat["evals_all"].copy()
    tbins = dat["tbins"]
    alphas_all = np.zeros(evals_bin_all.shape[:3])
    n_sim = evals_bin_all.shape[0]
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
            for j in range(n_sim):
                evals = evals_bin_all[j,k,i]
                if len(np.nonzero(np.isnan(evals))[0]) > 0:
                    ymax = min(500, np.nonzero(np.isnan(evals))[0][0] - 50)
                else:
                    ymax = 500
                alphas_all[j,k,i], yp = fit_powerlaw_exp(evals, np.arange(10, ymax))
                evals /= yp[0]
            evals_mean = evals_bin_all[:,k,i].mean(axis=0)
            ax.loglog(np.arange(1, len(evals)+1), evals_mean, 
                      color=lcolors[i], lw=1, zorder=i)
            if i%4==0 and k==0:
                ax.text(0.65, 0.94 - 0.1*i/4,  f"{tbin*1000:.0f} ms",
                        color=lcolors[i], transform=ax.transAxes, fontsize="small")
            
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
        ax.errorbar(tbins, alphas_all[:,k].mean(axis=0), 
                    alphas_all[:,k].std(axis=0), color=colors[k], lw=1)
    ax.set_xscale("log")
    ax.set_ylabel("power-law exponent ($\\alpha$)")
    ax.set_xlabel("time bin (sec.)")
    ax.set_ylim([0, 2])
    ax.set_xticks([0.01, 0.1, 1, 10])
    ax.set_xticklabels(["0.01", "0.1", "1", "10"])
    ax.xaxis.set_minor_locator(locmin)
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    il = plot_label(ltr, il, ax, transl)

    return fig