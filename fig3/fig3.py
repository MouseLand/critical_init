from matplotlib import pyplot as plt

import sys, os
import importlib as imp 

#from fig_utils import *
import matplotlib.gridspec as gridspec

import numpy as np 
import string
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms

from matplotlib import rcParams
from matplotlib.colors import ListedColormap
cmap_emb = ListedColormap(plt.get_cmap("gist_ncar")(np.linspace(0.05, 0.95), 100))

kp_colors = np.array([[0.55,0.55,0.55],
                      [0.,0.,1],
                      [0.8,0,0],
                      [1.,0.4,0.2],
                      [0,0.6,0.4],
                      [0.2,1,0.5],
                      ])

default_font = 12
rcParams["font.family"] = "Arial"
rcParams["savefig.dpi"] = 300
rcParams["axes.spines.top"] = False
rcParams["axes.spines.right"] = False
rcParams["axes.titlelocation"] = "left"
rcParams["axes.titleweight"] = "normal"
rcParams["font.size"] = default_font

ltr = string.ascii_lowercase
fs_title = 16
weight_title = "normal"


ax_titles = ['2p V1', 'ephys brainwide', '2p CA1']

short_titles = ['2p V1', 'ephys\n brainwide', '2p CA1', 'sim\n symm', 'sim\n non-symm']

#dcolors = [[.5,0,0], [.5,.5,0], [0,0,.5], [.75, .75, .75], [.5, .5, .5]]
#dcolors = ['r', 'y', 'b', [.75, .75, .75], [.5, .5, .5]]
dcolors2 = [[1,0,0], [1,.65,0], [0,0,1]]
dcolors = [[1,.5,.5], [1,.75,.25], [.5,.5,1], [.4, .4, .4],  [.75, .75, .75]]

subsets = [np.arange(6), np.arange(18,21),np.arange(6,10), np.arange(10,14), np.arange(14,18)]
nmax = [400, 100, 100]
    
yti = .95
yti2 = .95

ltr = string.ascii_lowercase

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


def my_cmap(xs, cmax):
    cc = np.outer(xs, cmax) + np.outer(1-xs, np.ones(3))
    return cc

def fig00(fig, il, grid, evals):
    cmap = mpl.colormaps['inferno']
    
    for j in range(3):
        ax = plt.subplot(grid[0,j])

        if j==0:
            ax.text(0, 1.125, 'PC auto-correlograms (noise-normalized)', fontsize='large',
                     transform = ax.transAxes, fontstyle='italic')
            plt.ylabel('correlation')
            plt.text(8, .95, 'PC index:')
            
        
        ax.set_title('\n %s'%ax_titles[j], y = yti,fontsize = 'medium')

        ixx = np.round(np.exp(np.linspace(np.log(.5),np.log(nmax[j]-1),50))).astype('int32')
        #ixx = np.unique(ixx)
        n_lines = len(ixx)

        colors = my_cmap(np.linspace(.1, 1, n_lines)[::-1], dcolors2[j])#[::-1]

        axin1 = ax.inset_axes([0.9, 0.6, 0.05, 0.4])
        axin1.yaxis.tick_right()
        axin1.spines[['left']].set_visible(False)
        axin1.spines[['bottom']].set_visible(False)
        axin1.set_xticks([])
        i0 = (ixx>=10).nonzero()[0][0]-1
        i1 = (ixx>=99).nonzero()[0][0]
        axin1.set_yticks([0.5, .5+i0, .5+i1], labels=[ixx[0]+1, 1+ixx[i0], 1+ixx[i1]])
        
        imm = colors[:,np.newaxis, :3]
        #imm = np.arange(n_lines)[:,np.newaxis, np.newaxis].astype('float32') / n_lines
        #imm = np.tile(imm, (1,1,3))
        axin1.imshow(imm, aspect = 'auto', alpha = .5)



        acg = avg_acg(evals, subsets[j])
        for kk in range(len(ixx)):
            plt.semilogx(np.arange(1,100), acg[ixx[kk],1:100].T, c = colors[kk]);
        
        if j==0:
            plt.xlabel('timelag (sec)')
            transl = mtransforms.ScaledTranslation(-30 / 72, 10 / 72, fig.dpi_scale_trans)
            il = plot_label(ltr, il, ax, transl)
            #plt.text(-.2, 1.1, 'a', fontsize = 'large', fontweight='bold', transform = ax.transAxes)
        #plt.xticks([1, 10, 100], labels = ['1', '10', '100'])
        plt.xticks([22/16, 22/4, 22, 22*4], labels = ['0.06', '0.25', '1', '4'])
    
        plt.plot([5,5], [0, 1], color = 'k')

    return il

def avg_acg(evals, ind):
    acg = np.zeros((400,100))
    for j in ind:
        acg += evals[j]['acg'][:400,:100]
    acg = acg/len(ind)
    return acg

def fig01(ax, evals):

    ax.set_title('PC auto-correlation @ $\\Delta = 0.23$s\n, example (2p V1)', y = yti)
    acg = evals[4]['acg']
    ac_index = acg[:400,4:7].mean(-1)
        
    r = np.corrcoef(np.log(ac_index), np.log(1+np.arange(len(ac_index))))[0,1]

    plt.loglog(1+np.arange(400), ac_index, '.', markersize=8, c = dcolors[0])
    plt.xlabel('sorted PCs')
    plt.xticks([1, 10, 100], labels = ['1', '10', '100'])
    plt.text(1, .8, 'r=%2.2f'%r, transform=ax.transAxes, ha = 'right')
    ax.set_yticks([.2, .3,.4,.6, 1], labels = ['.2', '.3', '.4', '.6', '1'])

def fig02(fig, il, grid_all,evals):
    grid = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=grid_all[0,3:])

    yl = [[.2, 1], [.1, 1], [.2, 1]]
    for j in range(3):
        acg = avg_acg(evals, subsets[j])
        ax = plt.subplot(grid[0,j])
        if j==0:
            ax.text(0, 1.125, 'PC auto-correlation (dt=0.23s)', transform = ax.transAxes, 
                    fontsize='large', fontstyle = 'italic')
            plt.ylabel('correlation')
            plt.xlabel('PC index')
            transl = mtransforms.ScaledTranslation(-30 / 72, 10 / 72, fig.dpi_scale_trans)
            il = plot_label(ltr, il, ax, transl)
        
        ax.set_title('\n %s'%ax_titles[j], y = yti, fontsize='medium')
        ac_index = acg[:nmax[j],4:7].mean(-1)
        plt.loglog(1+np.arange(len(ac_index)), ac_index, '.', 
                markersize=12,c = dcolors[j])
        
        r = np.corrcoef(np.log(ac_index), np.log(1+np.arange(len(ac_index))))[0,1]
        
        plt.xticks([1, 10, 100], labels = ['1', '10', '100'])
        plt.ylim([yl[j][0], yl[j][1]])
        
        if j==1:
            ax.set_yticks([.1, .2, .3,.4,.6, 1], labels = ['.1', '.2', '.3', '.4', '.6', '1'])
        else:
            ax.set_yticks([.2, .3,.4,.6, 1], labels = ['.2', '.3', '.4', '.6', '1'])
        plt.text(1, .8, 'r=%2.2f'%r, transform=ax.transAxes, ha = 'right')

    return il 

def plot_references():
    cmap = plt.get_cmap('jet')
    cmap = cmap([.4, .3, .4]) 

    for k in [0]:
        rr = np.linspace(.1, 1, 101)[:-1]
        yang = - 2 * np.pi  / (1 / np.log10(rr))
        yy = np.sin(yang / (k+1)) * rr
        xx = np.cos(yang / (k+1)) * rr
        ix = (yy<1 ) * (yy>0)
        xx, yy = xx[ix], yy[ix]
        plt.plot(xx, yy, c = 'k')
        plt.plot(xx, -yy, c = 'k')

def fig10(fig, il, grid, evals):
    ax = plt.subplot(grid[1,0])

    ax.set_title('Dynamic Mode Decomp.\n', y = yti2, fontstyle = 'italic')
    transl = mtransforms.ScaledTranslation(-30 / 72, 10 / 72, fig.dpi_scale_trans)
    il = plot_label(ltr, il, ax, transl)

    ax.plot([0, 1, 1, 0, 0], [0, 0, .2, .2, 0],color= 'k')
    for j in range(10):
        ax.plot([j*.1, j*.1], [0, .2], 'k') 
    plt.text(.05, -.05, '$... x_t ..... x_{t+dt} ...$')

    ax.text(0, .3, 
            'argmin$_B \\| x_{t+dt} - Bx_t\\|^2\\sim$\n$\\sim e^{(A-I)\\;dt\\,/\\,\\tau}$')
    ax.text(-.1,.1, 'neurons', rotation = 90, ha = 'center', rotation_mode='anchor')

    ax.set_ylim([-.05, .4])

    connectionstyle =  "arc3,rad=-0.5"
    for j in range(4):
        ax.annotate("",
                    xy=(.5+j*0.1, .2), xycoords='data',
                    xytext=(.05+j*0.1, .2), textcoords='data',
                    arrowprops=dict(arrowstyle="->", color="0.5",
                                    shrinkA=5, shrinkB=5,
                                    patchA=None, patchB=None,
                                    connectionstyle=connectionstyle,
                                    ),
                    )
    
    ax.axis('off')

    return il

import os
def fig11(fig, il, grid,evals):
    ax = plt.subplot(grid[1,1])

    ax.set_title('symmetric random matrix', y = yti2, fontsize='medium')
    ax.text(0, 1.125, 'Eigenvalues of DMD matrix (dt=0.23s)', 
            transform = ax.transAxes, fontstyle = 'italic', fontsize = 'large')

    plot_references()
    plt.scatter(evals[10]['eA'].real, evals[10]['eA'].imag, s = 8, color = dcolors[3])
    plt.xlabel('real part')
    plt.ylabel('imaginary part')
    plt.text(1, .5, '1 rotation per 10-fold\n attenuation', 
             c= 'k', fontsize='small', ha = 'right')
    # c = plt.get_cmap('jet')(0.4)
    plt.ylim([-1, 1])
    plt.xlim([.25, 1])

    transl = mtransforms.ScaledTranslation(-30 / 72, 10 / 72, fig.dpi_scale_trans)
    il = plot_label(ltr, il, ax, transl)
    
    ax = plt.subplot(grid[1,2])
    ax.set_title(' \n non-symmetric', y = yti2, fontsize='medium')
    plot_references()
    plt.scatter(evals[14]['eA'].real, evals[14]['eA'].imag, s = 8, color = dcolors[4])
    plt.ylim([-1, 1])
    plt.xlim([.25, 1])

    return il

def fig20(fig, il, grid, evals):
    axe_titles = ['example 2p V1', 'ex. ephys brainwide', 'ex. 2p CA1']

    iex = [0, 18, 6]
    for j in range(3):
        ax = plt.subplot(grid[2,j])

        if iex[j]<10:
            e = evals[iex[j]]['e']
        else:
            e = evals[iex[j]]['eA']

        plot_references()
        plt.scatter(e.real, e.imag, s = 8, color = dcolors[j])
        plt.ylim([-1, 1])
        plt.xlim([.25, 1])

        if j==0:
            ax.text(0, 1.125, 'Eigenvalues of DMD matrix (dt=0.23s)', transform = ax.transAxes, 
                 fontstyle = 'italic', fontsize='large')
            plt.xlabel('real part')
            plt.ylabel('imaginary part')
            transl = mtransforms.ScaledTranslation(-30 / 72, 10 / 72, fig.dpi_scale_trans)
            il = plot_label(ltr, il, ax, transl)
        
        ax.set_title('\n %s'%axe_titles[j], y = yti2, fontsize = 'medium')
    return il
        
def fig23(fig, il, grid, evals):
    ax = plt.subplot(grid[1:,3:])
    ax.set_title('Number of rotations per 10-fold attenuation',  fontstyle = 'italic')

    imap = np.array([0,0,0,0,0,0,2,2,2,2,3,3,3,3,4,4,4,4,1,1,1])

    yy = 0
    for k in range(5):
        ix = (imap==k).nonzero()[0]

        plt.plot([-.05,-.05], [yy, yy+len(ix)-1], color = dcolors[k])
        txt = plt.text(-.1, yy + (len(ix)-1)/2, short_titles[k], rotation = 90,
                 color = dcolors[k], va = 'bottom', ha = 'center', rotation_mode = 'anchor')
        for j in ix:
            if j<10:
                e = evals[j]['e']
            else:
                e = evals[j]['eA']
            
            ix = np.abs(e)>.25
            iang = np.angle(e[ix]) / (2*np.pi)
            iabs = -np.log10(np.abs(e[ix]))    
            irot = iang/iabs
            ixx = e[ix].imag>=0
            mu = irot[ixx].mean()
            sd = irot[ixx].std()
            m = np.percentile(irot[ixx], [5, 25, 75, 95])
            
            plt.plot([m[0], m[-1]], [yy, yy], color=dcolors[k])
            plt.plot([m[1], m[2]], [yy, yy], lw=4, color=dcolors[k])
            yy+=1 
        yy+=1 

    plt.xlabel('rotations per 10-fold attenuation')
    ax.spines[['left']].set_visible(False)
    ax.set_yticks([])

    transl = mtransforms.ScaledTranslation(-30 / 72, 10 / 72, fig.dpi_scale_trans)
    il = plot_label(ltr, il, ax, transl)

    axin1 = ax.inset_axes([0.4, 0.2, 0.5, 0.5])
    theta = np.linspace(0, 2*np.pi, 101)
    #axin1.plot(np.cos(theta), np.sin(theta), 'k')
    axin1.plot([0, 1], [0, 0], 'k')
    
    theta0 = 0.115 * (2*np.pi)
    r = 0.1**(theta0/(2*np.pi)) #0.85
    for j in range(1, int((2*np.pi)//theta0)+1):
        axin1.annotate("", xy=(r**j * np.cos(j*theta0), r**j * np.sin(j*theta0)), xytext=(0, 0),
            arrowprops=dict(arrowstyle="->"))
        
    axin1.plot(r**(theta/theta0) * np.cos(theta), r**(theta/theta0) * np.sin(theta), c = 'k') 
               #,c = plt.get_cmap('jet')(0.4))
    axin1.plot([0, r**(2*np.pi/theta0)], [0, 0], 'k', lw = 3)
    
    r = (0.1**(1/3))**(theta0/(2*np.pi)) #0.85
    axin1.plot(r**(theta/theta0) * np.cos(theta), r**(theta/theta0) * np.sin(theta), c = 'k')
               #,c = plt.get_cmap('jet')(0.3))
    
    r = (0.1**(5))**(theta0/(2*np.pi)) #0.85
    axin1.plot(r**(theta/theta0) * np.cos(theta), r**(theta/theta0) * np.sin(theta), c = 'k') 
               #,c = plt.get_cmap('jet')(0.5))
    
    
    axin1.axis('off')
    axin1.text(0, 1., 'rotations per 10-fold attenuation:' , transform=axin1.transAxes)

    a = -.75
    axin1.annotate("", xy=(a+.3,a), xytext=(a, a),
            arrowprops=dict(arrowstyle="->"))
    axin1.annotate("", xy=(a,a+.3), xytext=(a, a),
            arrowprops=dict(arrowstyle="->"))
    
    axin1.text(.7, .7, '3')#, color = plt.get_cmap('jet')(0.3))
    axin1.text(.25, .625, '1')#, color = plt.get_cmap('jet')(0.4))
    axin1.text(.6, .2, '0.2')#, color = plt.get_cmap('jet')(0.5))
    
    axin1.text(a,a-.15, 'real')
    axin1.text(a-.1,a, 'imaginary', rotation=90, rotation_mode = 'anchor')
    
    axin1.set_xlim([-1,1])
    axin1.set_ylim([-1,1])
    #axin1.axis('square')
            
    return il
