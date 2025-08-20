import os, sys, time, tqdm, importlib
from datetime import datetime
import numpy as np
import pandas as pd
import pathlib
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import style
import seaborn as sns
import scanpy as sc
import squidpy as sq
import scipy.stats as ss
from scipy.spatial.distance import cdist
from scipy.ndimage import gaussian_filter
from scipy.interpolate import griddata, RBFInterpolator
from sklearn import neighbors
import gseapy

from assocplots.qqplot import *
from operator import itemgetter
import matplotlib.patches as mpatches
from matplotlib.cm import ScalarMappable
from matplotlib.collections import PatchCollection
from matplotlib.colors import Normalize, is_color_like, ListedColormap
from matplotlib import colormaps
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from legendkit import SizeLegend, Colorbar
from legendkit.layout import vstack, hstack


### legend title on the left
def legend_title_left(leg):
    c = leg.get_children()[0]
    title = c.get_children()[0]
    hpack = c.get_children()[1]
    c._children = [hpack]
    hpack._children = [title] + hpack.get_children()


### QQ plot
def qqplot(data, labels, n_quantiles=200, alpha=0.95, error_type='theoretical', 
           distribution = 'binomial', log10conv=True, 
           color=['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'tab:brown', 'C9', 'tab:olive', 'tab:cyan', 'tab:gray'], 
           shape=['.','+','.','+','.','+'],
           fill_dens=[0.1 for _ in range(10)], type = 'uniform', title='None', 
           ms=5, lw=0.5, alp=0.5, legloc=2, xlim=None, ylim=None, tick_font=10, label_font=12, title_font=14,
           xticks=None, yticks=None,
           showXlabel=True, showYlabel=True, showXticks=True, showYticks=True, showLeg=True, ax=None):
    
    xmax = 0
    ymax = 0
    if type == 'uniform':
        # we expect distribution from 0 to 1
        for j in range(len(data)):
            # define quantiles positions:
            q_pos = np.concatenate([np.arange(99.)/len(data[j]), np.logspace(-np.log10(len(data[j]))+2, 0, n_quantiles)])
            # define quantiles in data
            q_data = mquantiles(data[j], prob=q_pos, alphap=0, betap=1, limit=(0, 1)) # linear interpolation
            # define theoretical predictions
            q_th = q_pos.copy()
            # evaluate errors
            q_err = np.zeros([len(q_pos),2])
            if np.sum(alpha) > 0:
                for i in range(0, len(q_pos)):
                    if distribution == 'beta':
                        q_err[i, :] = beta.interval(alpha, len(data[j])*q_pos[i], len(data[j]) - len(data[j])*q_pos[i])
                    elif distribution == 'binomial':
                        q_err[i, :] = binom.interval(alpha=alpha, n=len(data[j]), p=q_pos[i])
                    elif distribution == 'normal':
                        q_err[i, :] = norm.interval(alpha, len(data[j])*q_pos[i], np.sqrt(len(data[j])*q_pos[i]*(1.-q_pos[i])))
                    else:
                        print('Distribution is not defined!')
                q_err[i, q_err[i, :] < 0] = 1e-15
                if (distribution == 'binomial') | (distribution == 'normal'):
                    q_err /= 1.0*len(data[j])
                    for i in range(0, 100):
                        q_err[i,:] += 1e-15
            # print(q_err[100:, :])
            slope, intercept, r_value, p_value, std_err = linregress(q_th, q_data)
            # print(labels[j], ' -- Slope: ', slope, " R-squared:", r_value**2)
            #print(q_data.shape,q_th.shape,n_quantiles)
            ax.plot(-np.log10(q_th[n_quantiles-1:]), -np.log10(q_data[n_quantiles-1:]), '-', color=color[j], alpha=0.7)
            ax.scatter(-np.log10(q_th[:n_quantiles]), -np.log10(q_data[:n_quantiles]), edgecolor=color[j], facecolor=color[j], linewidth=lw, marker=shape[j], s=ms, label=labels[j], alpha=alp)
            xmax = np.max([xmax, - np.log10(q_th[1])])
            ymax = np.max([ymax, - np.log10(q_data[0])])
            #print(ymax)
            # print(- np.log10(q_th[:]))
            if np.sum(alpha)>0:
                if error_type=='experimental':
                    ax.fill_between(-np.log10(q_th), -np.log10(q_data/q_th*q_err[:,0]), -np.log10(q_data/q_th*q_err[:,1]), color=color[j], alpha=fill_dens[j], label='%1.2f CI'%alpha)
        if np.sum(alpha)>0:
            if error_type=='theoretical':
                ax.fill_between(-np.log10(q_th), -np.log10(q_err[:,0]), -np.log10(q_err[:,1]), color='grey', alpha=fill_dens[j], label='%1.2f CI'%alpha)
    ax.legend(loc=legloc)
    if not showLeg:
        ax.get_legend().remove()
    if showXlabel:
        ax.set_xlabel('Expected $-\log_{10} P$', fontsize=label_font)
    if showYlabel:
        ax.set_ylabel('Observed $-\log_{10} P$', fontsize=label_font)
    ax.plot([0, 100], [0, 100],'--k',linewidth=0.5)
    if xlim is None:
        ax.set_xlim([0, np.ceil(xmax)])
    else:
        ax.set_xlim(xlim)
    if ylim is None:
        ax.set_ylim([0, np.ceil(ymax*1.05)])#np.ceil(ymax*1.05)])
    else:
        ax.set_ylim(ylim)
    ax.set_title(title, fontsize=title_font)
    if not showXticks:
        ax.set_xticks([])
    if not showYticks:
        ax.set_yticks([])
    if xticks is not None:
        ax.set_xticks(xticks)
    if yticks is not None:
        ax.set_yticks(yticks)
    ax.tick_params(axis='both', which='major', length=3, width=1, labelsize=tick_font)


### niche cell type V radar plot
def draw_v_radar(vdf, comps, v_thres=0.2, colors=None, alpha=0.6, width=0.2, offset=None, xticklabels=None, rlabel_angle=0, figsize=(5,5), dpi=300, xtick_fs=16, xlabel_pad=20, ylim=(0,1.05), yticks=[0.2,0.4,0.6,0.8,1.0], yticklabels=['0.2','0.4','0.6','0.8','1.0'], ytick_fs=12, leg_loc='upper right', leg_pos=(1.3, 1.1), leg_ncol=1, leg_title=None, leg_fs=16, leg_title_fs=16, leg_title_left=False, title=None, title_fs=20, title_y=1.15):

    sns.set_theme(style='white')
    
    categories = vdf.index.tolist()
    if xticklabels is None:
        xticklabels = categories
    niches = []
    for x in comps:
        if isinstance(x, int):
            niches.append(f'comp{x}')
        else:
            niches.append(x)
    angles = np.linspace(0, 2*np.pi, len(categories), endpoint=False)
    
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi, subplot_kw=dict(projection='polar'))
    if offset is None:
        offset = 2*np.pi/72
    counter = {k:0  for k in categories}
    for i, niche in enumerate(niches):
        angles_ = np.zeros(len(categories)) 
        keys_ = vdf.loc[vdf[niche]>=v_thres].index.tolist()
        index_ = [categories.index(x) for x in keys_]
        for j, k in zip(index_, keys_):
            angles_[j] = angles[j] + counter[k] * offset
            counter[k] += 1
        values_ = vdf[niche].values.copy()
        values_[values_ < v_thres] = 0
        if len(colors[i]) == 3:
            fc = list(colors[i])+[alpha*0.5]
        elif len(colors[i]) == 4:
            fc = list(colors[i])
            fc[-1] = alpha*0.5
        ax.bar(angles_, values_, width=width, align='center', label=i+1, facecolor=fc, edgecolor=colors[i])

    ax.set_ylim(ylim)
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticklabels, fontsize=ytick_fs)
    ax.set_rlabel_position(rlabel_angle)
    ax.set_rorigin(0)
    ax.set_xticks(angles)
    ax.set_xticklabels(xticklabels, fontsize=xtick_fs, ha='center', va='center')
    ax.tick_params(axis='x', which='major', pad=xlabel_pad)
    ax.set_theta_zero_location("S")         
    ax.set_theta_direction("clockwise")

    if leg_loc is not None:
        handles, labels = ax.get_legend_handles_labels()
        leg = ax.legend(handles, labels, bbox_to_anchor=leg_pos, loc=leg_loc, ncol=leg_ncol, borderpad=0, borderaxespad=0, columnspacing=0.8, framealpha=0, markerscale=1, fontsize=leg_fs, handletextpad=0.3, title=leg_title, title_fontsize=leg_title_fs, alignment='left')
        if leg_title_left:
            legend_title_left(leg)

    plt.suptitle(title, y=title_y, fontsize=title_fs)
    plt.show()


### spatial distribution of niche score across whole tissue section
def draw_niche_score_spatial(adata, score_df, score_column, target_ct=None, draw_bg=True, bg_color='#F8F8F8', ms_bg=2, sort_score=False, cmap=mpl.colormaps['magma'], target_ec='#C0C0C0', ms=4, lw=0.4, window=None, window_lw=4, window_ls='--', window_lc='k', ylabel=None, ylabel_fs=12, title=None, title_fs=12, show_colorbar=False, cb_label=None, cb_tick_fs=20, figsize=(6, 6), dpi=300, no_ticks=True, aspect=['equal', 'auto'], invert_yaxis=True):
    
    if sort_score:
        score_df = score_df.sort_values(score_column).reset_index(drop=True)
    lims = [adata.obs.x.min(), adata.obs.x.max(), adata.obs.y.min(), adata.obs.y.max()]
    
    fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=dpi)

    ## background
    if draw_bg:
        bg_df = adata[adata.obs['cell_type']!=target_ct].obs.copy()
        ax.scatter(bg_df['x'], bg_df['y'], ms_bg, facecolor=bg_color, edgecolor=bg_color, lw=0)

    ## target cell type
    sca_lims = {'vmin': 0, 'vmax': score_df[score_column].max()}
    ax.scatter(score_df['x'], score_df['y'], ms, score_df[score_column], edgecolor=target_ec, lw=lw, cmap=cmap, **sca_lims)
    
    if show_colorbar:
        axins = inset_axes(
            ax, width=0.25, height=2, loc="upper left",
            bbox_transform=ax.transAxes, borderpad=0,
        )
        cb = fig.colorbar(mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(0,1), cmap=cmap), cax=axins)
        cb.set_ticks([0,1])
        cb.set_ticklabels([0,1], fontsize=cb_tick_fs)
        cb.outline.set_visible(False)
        cb.ax.tick_params(length=0, which='major')
        cb.ax.yaxis.set_ticks_position('right')
        cb.ax.yaxis.set_label_position('right')

    ax.set_xlim([lims[0], lims[1]])
    ax.set_ylim([lims[2], lims[3]])
    if window is not None:
        x0, y0, xl, yl = window
        rect = mpatches.Rectangle((x0, y0), xl, yl, linewidth=window_lw, linestyle=window_ls, edgecolor=window_lc, facecolor='none')
        ax.add_patch(rect)
    if invert_yaxis:
        ax.invert_yaxis()
    if no_ticks:
        ax.set_xticks([])
        ax.set_yticks([])
    for s in ['top', 'bottom', 'right', 'left']:
        ax.spines[s].set_visible(False)
    ax.set_ylabel(ylabel, fontsize=ylabel_fs)
    ax.set_title(title, fontsize=title_fs)
    plt.show()


### niche sub cell type count and enrichment fold
def sub_ct_enrich(comp, score_df, subcts, quantiles, q_order):

    col = f'S_comp{comp}'
    base_count = score_df.groupby('sub_cell_type').size().loc[subcts]
    count_df = pd.DataFrame({'base': base_count})

    quants = []
    for q in quantiles:
        quant = np.quantile(score_df[col], q)
        # print(q, quant)
        high_df = score_df.loc[score_df[col]>quant]
        high_count = high_df.groupby('sub_cell_type').size().loc[subcts]
        q_name = str(q).replace('.', '')
        count_df[f'obs_high_{q_name}'] = high_count
        count_df[f'exp_high_{q_name}'] = (count_df['base'] * count_df[f'obs_high_{q_name}'].sum() / count_df['base'].sum()).astype(int)
        count_df[f'exp_high_{q_name}'] = np.maximum(count_df[f'exp_high_{q_name}'], 1)
        count_df[f'fold_{q_name}'] = count_df[f'obs_high_{q_name}'] / count_df[f'exp_high_{q_name}']
        count_df[f'fold_{q_name}'] = np.round(count_df[f'fold_{q_name}'], 3)
    cols_reorder = ['base'] + [f"fold_{str(q).replace('.', '')}" for q in quantiles] + [f"obs_high_{str(q).replace('.', '')}" for q in quantiles] + [f"exp_high_{str(q).replace('.', '')}" for q in quantiles]
    count_df = count_df[cols_reorder].sort_values(f"fold_{str(q_order).replace('.', '')}", ascending=False)
    
    return count_df

