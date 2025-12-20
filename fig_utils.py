"""
Copyright © 2023 Howard Hughes Medical Institute, Authored by Carsen Stringer and Marius Pachitariu.
"""
import string
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import numpy as np
from matplotlib import rcParams
from matplotlib.colors import ListedColormap
from powerlaw import fit_powerlaw_exp

default_font = 12
rcParams["font.family"] = "Arial"
rcParams["savefig.dpi"] = 300
rcParams["axes.spines.top"] = False
rcParams["axes.spines.right"] = False
rcParams["axes.titleweight"] = "normal"
rcParams["font.size"] = default_font
dcolors = np.array([[1,.5,.5], [.5,.5,1],[1,.75,.25], [.75, .75, .75], [.4, .4, .4]])
areas = ["V1", "CA1", "ephys", "PPC", "sensorimotor"]
locmin = matplotlib.ticker.LogLocator(base=10.0,subs=np.arange(0.1, 1., 0.1), 
                                          numticks=10)
    
ltr = string.ascii_lowercase
fs_title = 16
weight_title = "normal"

def plot_spectrum(ax, ss, color=0.5*np.ones(3), plot_fit=True, lw=3, ymax=500):
    alpha, yp = fit_powerlaw_exp(ss[:1000], np.arange(10, min(len(ss), ymax)))
    ss /= yp[0]
    yp /= yp[0]
    h = ax.loglog(np.arange(1, min(len(ss)+1, 1001)), ss[:1000], lw=lw, color=color)
    if plot_fit:
        ax.loglog(np.arange(1, len(yp)+1), yp , lw=lw*0.5, color=0.*np.ones(3), ls="--")
    ax.set_ylim(0.002, 2)
    ax.set_xlim(1, np.exp(np.log(2) - np.log(0.002)))
    ax.set_xticks([1, 10, 100, 1000])
    ax.set_xticklabels(["1", "10", "100", "1,000"])
    ax.set_yticks([0.01, 0.1 ,1])
    ax.set_yticklabels(["0.01", "0.1", "1"])
    return alpha, h


def plot_label(ltr, il, ax, trans, fs_title=20):
    ax.text(
        0.0,
        1.0,
        ltr[il],
        transform=ax.transAxes + trans,
        va="bottom",
        fontsize=fs_title,
        fontweight="bold",
    )
    il += 1
    return il


def conn_panel(ax, Aex, iplot, vmax=0.02, xt=-0.05, colorbar=True, title="sparse connectivity"):
    ax.spines["bottom"].set_visible(False)
    ax.spines["top"].set_visible(True)
    im = ax.imshow(Aex[np.ix_(iplot, iplot)], cmap="bwr", vmin=-vmax, vmax=vmax)
    # unicode arrow
    ax.set_ylabel(r"$\leftarrow$ neurons")
    ax.set_title(r"neurons $\rightarrow$", fontsize="medium", loc="center")
    ax.set_xticks([])
    ax.set_yticks([])
    if colorbar:
        cax = ax.inset_axes([1.05, 0.75, 0.05, 0.25])
        cb = plt.colorbar(im, cax=cax)#, fontsize="small")
        cb.ax.set_ylim(-vmax, vmax)
        cb.set_ticks([-vmax, vmax])
        cb.ax.tick_params(labelsize="small")
    ax.text(xt, 1.32, title, transform=ax.transAxes, 
            ha="left", va="top", fontsize="large", fontstyle="italic")
