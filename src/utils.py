import os, sys, time, tqdm, importlib
from datetime import datetime
import numpy as np
import pandas as pd
import pathlib
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import style
import matplotlib.lines as mlines
import seaborn as sns
import scanpy as sc
import squidpy as sq
import scipy.stats as ss
from scipy.spatial.distance import cdist
from scipy.ndimage import gaussian_filter
from scipy.interpolate import griddata, RBFInterpolator
from sklearn import neighbors
import gseapy
import cmasher as cmr

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
def draw_v_radar(vdf, niches, niche1s=[], niche2s=[], v_thres=0.2, colors=None, alpha=0.6, width=0.2, offset=None, xticklabels=None, rlabel_angle=0, figsize=(5,5), dpi=300, xtick_fs=16, xlabel_pad=20, ylim=(0,1.05), yticks=[0.2,0.4,0.6,0.8,1.0], yticklabels=['0.2','0.4','0.6','0.8','1.0'], ytick_fs=12, leg_loc='center', leg_pos=(0.5,1.2), leg_ncol=1, leg_title=None, leg_pos_spec=None, leg_ncol_spec=1, leg_title_spec=None, leg_fs=16, leg_title_fs=16, leg_title_left=False, title=None, title_fs=20, title_y=1.15):

    sns.set_theme(style='white')
    
    for i, x in enumerate(niches):
        if 'comp' not in str(x):
            niches[i] = f'comp{str(x)}'
    for i, x in enumerate(niche1s):
        if 'comp' not in str(x):
            niche1s[i] = f'comp{str(x)}'
    for i, x in enumerate(niche2s):
        if 'comp' not in str(x):
            niche2s[i] = f'comp{str(x)}'
    all_niches = niches + niche1s + niche2s
    
    categories = vdf.index.tolist()
    if xticklabels is None:
        xticklabels = categories
    angles = np.linspace(0, 2*np.pi, len(categories), endpoint=False)
    
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi, subplot_kw=dict(projection='polar'))
    if offset is None:
        offset = 2*np.pi/72
    counter = {k:0  for k in categories}
    for i, niche in enumerate(all_niches):
        angles_ = np.zeros(len(categories)) 
        keys_ = vdf.loc[vdf[niche]>=v_thres].index.tolist()
        index_ = [categories.index(x) for x in keys_]
        for j, k in zip(index_, keys_):
            angles_[j] = angles[j] + counter[k] * offset
            counter[k] += 1
        values_ = vdf[niche].values.copy()
        values_[values_ < v_thres] = 0

        fc = colors[i]
        if isinstance(colors[i], str):
            fc = mpl.colors.hex2color(colors[i])
        if len(fc) == 3:
            fc = list(fc)+[alpha]
        elif len(fc) == 4:
            fc = list(fc)
            fc[-1] = alpha
        if niche in niches:
            ax.bar(angles_, values_, width=width, align='center', label=i+1, facecolor=fc, edgecolor=colors[i])
        elif niche in niche1s:
            ax.bar(angles_, values_, width=width, align='center', facecolor='none', edgecolor=colors[i], hatch='//')
        elif niche in niche2s:
            ax.bar(angles_, values_, width=width, align='center', facecolor='none', edgecolor=colors[i], hatch='\\\\')

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
        if leg_pos_spec is not None:
            ax.add_artist(leg)
    if leg_pos_spec is not None:
        legend_elements = []
        for i, niche in enumerate(niche1s):
            if len(niche1s) == 1:
                label = 'Pri'
            else:
                label = f'Pri{i+1}'
            legend_elements.append(mpatches.Patch(facecolor='none', edgecolor=colors[len(niches)+i], hatch='//', label=label))
        for i, niche in enumerate(niche2s):
            if len(niche2s) == 1:
                label = 'Met'
            else:
                label = f'Met{i+1}'
            legend_elements.append(mpatches.Patch(facecolor='none', edgecolor=colors[len(niches)+len(niche1s)+i], hatch='//', label=label))
        leg = ax.legend(handles=legend_elements, bbox_to_anchor=leg_pos_spec, loc=leg_loc, ncol=leg_ncol_spec, borderpad=0, borderaxespad=0, columnspacing=0.8, framealpha=0, markerscale=1, fontsize=leg_fs, handletextpad=0.3, title=leg_title_spec, title_fontsize=leg_title_fs, alignment='left')

    plt.suptitle(title, y=title_y, fontsize=title_fs)
    plt.show()


### niche gene dotmap
def draw_u_dotmap(udf, focus_comp, draw_comps, n_top_gene=10, add_genes=None, sizes=(5, 150), xticklabels=None, aspect_equal=False, fs=11, h=None, w=3, title=None, mx=0.2, my=0.03, leg_pos=(1.1,0.95), no_leg=False, dpi=200):

    if 'comp' not in str(focus_comp):
        focus_comp = f'comp{focus_comp}'
    for i, x in enumerate(draw_comps):
        if 'comp' not in str(x):
            draw_comps[i] = f'comp{x}'
    gene_names = udf[draw_comps].sort_values(focus_comp, ascending=False).head(n_top_gene).index.tolist()
    if add_genes is not None:
        gene_names = list(set(gene_names) | set(add_genes))
    ucolor_df0 = udf[draw_comps].loc[gene_names].sort_values(focus_comp, ascending=False)
    ucolor_df = ucolor_df0.stack().reset_index(name="u")
    
    plt.rcParams.update({"figure.dpi": dpi})
    sns.set_theme(style="whitegrid")
    sns.set_context('paper',font_scale=1.)
    if h is None:
        h = 0.25*len(ucolor_df0)
    asp = w/h
    csm = ucolor_df.u.max()
    g = sns.relplot(
        data=ucolor_df, 
        x="level_1", y="level_0", hue="u", size="u",
        palette="vlag", hue_norm=(-csm, csm), edgecolor=".7", linewidth=1,
        height=h, aspect=asp, sizes=sizes, size_norm=(0, csm)
    )
    g.set(xlabel="", ylabel="", xticklabels=xticklabels, title=None)
    if aspect_equal:
        g.set(aspect='equal')
    if title is not None:
        g._axes[0][0].set_title(title, fontsize=fs, loc='center', pad=10)
    
    g.set_xticklabels(size = fs, rotation=0)
    g.set_yticklabels(size = fs, style='italic')
    g.despine(left=True, bottom=True)
    g.ax.margins(x=mx, y=my)
    if len(xticklabels) > 1:
        g.ax.text(-1, len(ucolor_df0)+0.05, 'MCN', fontsize=fs, ha='right', va='top')

    if no_leg:
        g._legend.remove()
    else:
        sns.move_legend(
            g, "upper right", bbox_to_anchor=leg_pos, fontsize=fs, ncols=1, frameon=False,
        )

    return g


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


### niche score spatial distribution by kernel density estimation
def fit_kde_diff(score_df, col, z_quantile=1, bw=20, kernel=['gaussian', 'tophat']):
    
    assert col in score_df.columns, f'{col} not in score_df.columns!'
    x = score_df.x.values
    y = score_df.y.values
    z = score_df[col].values
    xy = np.vstack([x, y]).T # coordinates of points
    keep_ids = np.where(z <= np.quantile(z, z_quantile))[0]
    xy = xy[keep_ids,:]
    z = z[keep_ids]
    z = (z - z.min()) / (z.max() - z.min())

    kde1 = neighbors.KernelDensity(kernel=kernel, bandwidth=bw).fit(xy, sample_weight=z)
    kde0 = neighbors.KernelDensity(kernel=kernel, bandwidth=bw).fit(xy)
    
    return kde1, kde0


def sample_kde_diff(kde1, kde0, lims, nx=100, ny=100):
    
    xmin, xmax, ymin, ymax = lims
    xgrid, ygrid = np.mgrid[xmin:xmax:eval(f'{nx}j'), ymin:ymax:eval(f'{ny}j')]
    grid_coords = np.vstack([xgrid.ravel(), ygrid.ravel()]).T
    
    Z1 = kde1.score_samples(grid_coords)
    Z0 = kde0.score_samples(grid_coords)
    Z1 = np.reshape(np.exp(Z1), xgrid.shape)
    Z0 = np.reshape(np.exp(Z0), xgrid.shape)
    Z = Z1 - Z0
    Z = Z - Z.mean()
    
    return Z, Z1, Z0
    

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


### gene expression spatial
def draw_gene_spatial(adata, gene, ct=None, title=None, cmap=None, vmax=None, expr_quant=1, ms=4, figsize=(15,9), dpi=300):

    lims = [adata.obs.x.min(), adata.obs.x.max(), adata.obs.y.min(), adata.obs.y.max()]
    values = sc.get.obs_df(adata, keys=gene)
    df = pd.DataFrame({'x': adata.obs['x'].copy(), 
                       'y': adata.obs['y'].copy(), 
                       'cell_type': adata.obs['cell_type'].copy(),
                       'value': values.copy()})
    df = df.sort_values('value', ascending=True)
    if ct is not None:
        bg_df = df.loc[df['cell_type']!=ct]
        df = df.loc[df['cell_type']==ct]

    if vmax is None:
        vmax = np.quantile(values, expr_quant)
    vmax_round = np.round(vmax, 1)
    sca_lims = {'vmin': 0, 'vmax': vmax}
    if cmap is None:
        cmap = mpl.cm.Reds
    
    fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=dpi)
    if ct is not None:
        ax.scatter(bg_df['x'], bg_df['y'], ms, c='#F8F8F8', edgecolor='#F8F8F8', lw=0)
    ax.scatter(df['x'], df['y'], ms, c=df['value'], edgecolor='#C0C0C0', lw=0, cmap=cmap, **sca_lims)
    
    axins = inset_axes(
        ax, width=0.25, height=2, loc="upper left",
        # bbox_to_anchor=(0, 1, 1, 1), 
        bbox_transform=ax.transAxes, borderpad=0,
    )
    norm = mpl.colors.Normalize(vmin=0, vmax=vmax)
    cb = fig.colorbar(mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(0, vmax), cmap=cmap), cax=axins)
    cb.set_ticks([0.0,vmax_round])
    cb.set_ticklabels([0.0,vmax_round], fontsize=20)
    cb.outline.set_visible(False)
    cb.ax.tick_params(length=0, which='major')
    cb.ax.yaxis.set_ticks_position('right')
    cb.ax.yaxis.set_label_position('right')
    
    if title is None:
        ax.set_title(gene, fontsize=32, style='italic')
    else:
        ax.set_title(title, fontsize=32, style='italic')
    ax.set_xlim([lims[0], lims[1]])
    ax.set_ylim([lims[2], lims[3]])
    ax.invert_yaxis()
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ['top', 'bottom', 'right', 'left']:
        ax.spines[s].set_visible(False)
    plt.show()


## get cell ids within one niche
def get_niche_cell_ids(meta, meta_key, adata, comp_col, niche_quantile=0.5, sigma=40, cutoff=0.05):
    
    index_df = meta[meta_key].copy()[['cell_id', 'x', 'y', comp_col]]
    tmp = meta[meta_key][comp_col]
    tmp = tmp[tmp > 0]
    thres = np.quantile(tmp, niche_quantile)
    index_df = index_df.loc[index_df[comp_col] > thres]
    print(f'Keep {len(index_df)} / {len(tmp)} target cells.')
    all_df = adata.obs.copy()[['cell_id', 'x', 'y', 'cell_type']]
    K_index = cdist(index_df[['x', 'y']].values, all_df[['x', 'y']].values)
    K_index = np.exp(-K_index**2/sigma**2)
    K_index[K_index < cutoff] = 0
    keep_niche_cell_ids = np.where(K_index.sum(0)>0)[0]
    keep_niche_cell_ids = all_df.cell_id.values[keep_niche_cell_ids]
    
    return keep_niche_cell_ids


## get cell ids in tumors' neighborhood
def get_tumor_neighbor_cell_ids(adata, target_ct = 'Tumor', max_chunk_target=10000, sigma = 40, cutoff = 0.05):
    
    loc_target = adata.obs.loc[adata.obs.cell_type==target_ct, ['x', 'y']].values
    loc_all = adata.obs[['x','y']].values
    print(f'{len(loc_all)} cells, {len(loc_target)} {target_ct}.')
    if loc_target.shape[0]>max_chunk_target:
        n_chunk = np.ceil(loc_target.shape[0]/max_chunk_target).astype(int)
        loc_target_chunks = np.array_split(loc_target, n_chunk, axis=0)
        K_target_chunks = []
        for i, loc_target_chunk in enumerate(loc_target_chunks):
            K_target_chunk = cdist(loc_target_chunk, loc_all)
            K_target_chunk = np.exp(-K_target_chunk**2/sigma**2)
            K_target_chunk[K_target_chunk < cutoff] = 0
            K_target_chunks.append(K_target_chunk)
        K_target = np.vstack(K_target_chunks)
    else:
        K_target = cdist(loc_target, loc_all)
        K_target = np.exp(-K_target**2/sigma**2)
        K_target[K_target < cutoff] = 0
    print(f'K_target: {K_target.shape}.')
    keep_niche_cell_ids = np.where(K_target.sum(0)>0)[0]
    keep_niche_cell_ids = adata.obs.cell_id.values[keep_niche_cell_ids]
    print(f'{len(keep_niche_cell_ids)} cells in the neighborhood of {target_ct}.\n')
    return keep_niche_cell_ids


## draw cells in one niche
def draw_niche_cells(adata, keep_niche_cell_ids, cmap_ct_dict, ct_name_map,
                     ms_bg = 4, color_bg = '#F0F0F0', lw_bg = 0, 
                     ms = 6, ecolor = '#C0C0C0', lw = 0.4,
                     title=None, title_fs=30, figsize=(7,10), dpi=300):
    
    fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=dpi)
    # background
    bg_df = adata.obs.loc[~adata.obs.cell_id.isin(keep_niche_cell_ids)]
    ax.scatter(bg_df['x'], bg_df['y'], ms_bg, facecolors=color_bg, edgecolor=color_bg, lw=lw_bg)
    # cells in niche
    niche_df = adata.obs.loc[adata.obs.cell_id.isin(keep_niche_cell_ids)]
    cts = niche_df.cell_type.value_counts().index.tolist()
    for i, ct in enumerate(cts):
        color = cmap_ct_dict[ct]
        df = niche_df.loc[niche_df.cell_type==ct]
        ax.scatter(df['x'], df['y'], ms_bg, facecolors=color, edgecolor=color, lw=0, label=ct_name_map[ct])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=title_fs, pad=10)
    for s in ['top', 'bottom', 'right', 'left']:
        ax.spines[s].set_visible(False)
    # leg = ax.legend(loc='upper center', bbox_to_anchor=(0.5,0), ncol=5, fontsize=12, markerscale=4)
    handles, labels = ax.get_legend_handles_labels()
    order = np.argsort(labels)
    handles = [handles[i] for i in order]
    labels = [labels[i] for i in order]
    # leg = ax.legend(handles, labels,
    #                 loc='upper center', bbox_to_anchor=(0.5,0), ncol=5, fontsize=12, markerscale=4)
    leg = ax.legend(handles, labels,
                    loc='upper left', bbox_to_anchor=(1,1), ncol=1, fontsize=20, markerscale=4)

    ax.margins(0,0)
    plt.show()


## heatmap for one source-target cell type pair in primary/met, ordered by magnitude rank
def lr_heatmap_prim_met(lr_prim_met, source, target, filter_thres1s, filter_thres2s, n_top_orders, 
                        source_name=None, target_name=None,
                        cmap_lr=None, leg_bbox=None, mx=None, my=None, rot_x=90, sizes=(10,250), fs=10, w=3, dpi=300):

    datasets = ['Pri', 'Met']
    order_metric = 'magnitude_rank'
    filter_metric1 = 'specificity_rank'
    filter_metric2 = 'magnitude_rank'

    ## use common lrs
    common_lrs = []
    for i, lr_res in enumerate(lr_prim_met):
        lrs_ = lr_res.loc[(lr_res.source==source)&(lr_res.target==target)].lr.unique().tolist()
        if len(common_lrs) == 0:
            common_lrs = lrs_
        else:
            common_lrs = sorted(list(set(common_lrs) & set(lrs_)))
    print(f'Use common {len(common_lrs)} LRs.')
    for i in range(len(lr_prim_met)):
        lr_prim_met[i] = lr_prim_met[i].loc[lr_prim_met[i].lr.isin(common_lrs)]
    
    terms = []
    for i, (lr_res, dataset) in enumerate(zip(lr_prim_met, datasets)):

        df0 = lr_res.loc[(lr_res.source==source)&(lr_res.target==target)&(lr_res[filter_metric1]<=filter_thres1s[i])&(lr_res[filter_metric2]<=filter_thres2s[i])]
        df_ = df0.sort_values(order_metric).head(n_top_orders[i])[['lr', 'specificity_rank', 'magnitude_rank']]
        df_['dataset'] = dataset
        terms.extend(df_.lr.tolist())
        print(dataset, len(df0), len(df_))
    terms = list(set(terms))
    dfs = []
    for i, (lr_res, dataset) in enumerate(zip(lr_prim_met, datasets)):
        df_ = lr_res.loc[(lr_res.source==source)&(lr_res.target==target), ['lr', 'specificity_rank', 'magnitude_rank']]
        df_ = df_.set_index('lr').loc[terms][['specificity_rank', 'magnitude_rank']].reset_index()
        df_['dataset'] = dataset
        dfs.append(df_)
    concat_df = pd.concat(dfs).reset_index(drop=True)
    
    spec_long = concat_df[['lr', 'dataset', 'specificity_rank']]
    spec_wide = spec_long.pivot(index='lr', columns='dataset', values='specificity_rank').fillna(1)
    spec_wide['min'] = spec_wide.min(1)
    spec_wide = spec_wide.sort_values('min')
    spec_wide = spec_wide.iloc[:,:len(datasets)]
    mag_long = concat_df[['lr', 'dataset', 'magnitude_rank']]
    mag_wide = mag_long.pivot(index='lr', columns='dataset', values='magnitude_rank').fillna(1)
    mag_wide['min'] = mag_wide.min(1)
    mag_wide = mag_wide.sort_values('min')
    mag_wide = mag_wide.iloc[:,:len(datasets)]
    print('Combine:', len(mag_wide))
    mag_df = mag_wide.stack().reset_index(name='mag')
    spec_df = spec_wide.stack().reset_index(name='spec')
    mag_spec_df = mag_df.merge(spec_df, on=['lr', 'dataset'])
    mag_spec_df['mag'] = -np.log10(mag_spec_df['mag'].values)
    mag_spec_df['spec'] = -np.log10(mag_spec_df['spec'].values)
    mag_spec_df['dataset'] = pd.Categorical(mag_spec_df['dataset'], categories=datasets)

    ## heatmap
    plt.rcParams.update({"figure.dpi": dpi})
    sns.set_theme(style="whitegrid")
    sns.set_context('paper',font_scale=1.)
    if cmap_lr is None:
        cmap_lr = cmr.get_sub_cmap(sns.cubehelix_palette(rot=0.1, dark=0.2, light=0.8, as_cmap=True, reverse=True), 0, 1)
    
    xticklabels = datasets
    h = 0.3 * len(mag_wide)
    if mx is None or my is None:
        mx = 0.5/(len(datasets)-1)
        my = 0.5/(len(mag_wide)-1) 
    hue_min = mag_spec_df['spec'].min()
    hue_max = mag_spec_df['spec'].max()
    size_min = mag_spec_df['mag'].min()
    size_max = mag_spec_df['mag'].max()
    g = sns.relplot(
        data=mag_spec_df, 
        x="dataset", y="lr", hue="spec", size="mag",
        palette=cmap_lr, hue_norm=(0,hue_max), size_norm=(0,size_max), edgecolor=".7",
        height=h, aspect=1, sizes=sizes, legend=True, 
    )
    if source_name is None:
        source_name = source
    if target_name is None:
        target_name = target
    title = f'{source_name} $\\rightarrow$ {target_name}'
    g.set(xlabel="", ylabel="", xticklabels=xticklabels, title=None, aspect='equal')
    g._axes[0][0].set_title(title, fontsize=fs+2, color='darkblue')
    g.set_xticklabels(size = fs, rotation=rot_x)
    g.set_yticklabels(size = fs, style='italic')
    g.despine(left=True, bottom=True)
    g.ax.margins(x=mx, y=my)
    if leg_bbox is not None:
        for i in range(len(g._legend.texts)):
            if g._legend.texts[i].get_text() == 'spec':
                g._legend.texts[i].set_text('Spec.')
            if g._legend.texts[i].get_text() == 'mag':
                g._legend.texts[i].set_text('Mag.')
        sns.move_legend(
            g, "center", bbox_to_anchor=leg_bbox, fontsize=fs, ncols=1, frameon=False, alignment='left',
        )
    else:
        g._legend.remove()
    plt.show()

    return g, concat_df, mag_spec_df


## heatmap for one target cell type, ordered by magnitude rank
def lr_heatmap_mag_target(lr_res, sources, target, filter_thres1s, filter_thres2s, n_top_orders, 
                          source_names=None, target_name=None,
                          add_lrs=None, cmap_lr=None, leg_bbox=None, mx=None, my=None, rot_x=90, dpi=300):
    
    order_metric = 'magnitude_rank'
    filter_metric1 = 'specificity_rank'
    filter_metric2 = 'magnitude_rank'

    keep_cols = ['lr', 'source', 'target', 'specificity_rank', 'magnitude_rank']
    dfs = []
    for i, source in enumerate(sources):
        df = lr_res.loc[(lr_res.source==source)&(lr_res.target==target)]
        df0 = df.loc[(df[filter_metric1]<=filter_thres1s[i])&(df[filter_metric2]<=filter_thres2s[i])]
        df_ = df0[keep_cols].sort_values(order_metric).head(n_top_orders[i])
        lrs_ = df_.lr.values.tolist()
        df_ = lr_res.loc[(lr_res.lr.isin(lrs_))&(lr_res.source.isin(sources))&(lr_res.target==target)][keep_cols]
        print(source, len(lrs_), df_.shape)
        if add_lrs is not None:
            for lr in add_lrs:
                if lr not in df_.lr.values:
                    if lr in df.lr.values:
                        df_ = pd.concat([df_, df.loc[df.lr==lr][keep_cols]])
                    else:
                        df_ = pd.concat([df_, pd.DataFrame([[lr, source, target, 1, 1]], columns=keep_cols)])
        df_ = df_.sort_values(order_metric)
        dfs.append(df_)
    concat_df = pd.concat(dfs).drop_duplicates()
    print(len(concat_df), concat_df.lr.unique().shape)
    
    spec_long = concat_df[['lr', 'source', 'specificity_rank']]
    spec_wide = spec_long.pivot(index='lr', columns='source', values='specificity_rank').fillna(1)
    spec_wide['min'] = spec_wide.min(1)
    spec_wide = spec_wide.sort_values('min')
    spec_wide = spec_wide.iloc[:,:len(sources)]
    mag_long = concat_df[['lr', 'source', 'magnitude_rank']]
    mag_wide = mag_long.pivot(index='lr', columns='source', values='magnitude_rank').fillna(1)
    mag_wide['min'] = mag_wide.min(1)
    mag_wide = mag_wide.sort_values('min')
    mag_wide = mag_wide.iloc[:,:len(sources)]
    print('Combine:', len(mag_wide))
    mag_df = mag_wide.stack().reset_index(name='mag')
    spec_df = spec_wide.stack().reset_index(name='spec')
    mag_spec_df = mag_df.merge(spec_df, on=['lr', 'source'])
    mag_spec_df['mag'] = -np.log10(mag_spec_df['mag'].values)
    mag_spec_df['spec'] = -np.log10(mag_spec_df['spec'].values)
    mag_spec_df['source'] = pd.Categorical(mag_spec_df['source'], categories=sources)

    ## heatmap
    plt.rcParams.update({"figure.dpi": dpi})
    sns.set_theme(style="whitegrid")
    sns.set_context('paper',font_scale=1.)
    # cmap_lr = cmr.get_sub_cmap(mpl.cm.magma, 0.05,0.95)
    if cmap_lr is None:
        cmap_lr = cmr.get_sub_cmap(sns.cubehelix_palette(rot=0.1, dark=0.2, light=0.8, as_cmap=True, reverse=True), 0, 1)

    if source_names is None:
        source_names = sources
    xticklabels = source_names
    fs = 10
    w = 3
    h = 0.3 * len(mag_wide)
    if mx is None or my is None:
        mx = 0.5/(len(sources)-1) if len(sources) > 1 else 5 
        my = 0.5/(len(mag_wide)-1) 
    sizes = (10, 250)
    hue_max = mag_spec_df['spec'].max()
    size_max = mag_spec_df['mag'].max()
    g = sns.relplot(
        data=mag_spec_df, 
        x="source", y="lr", hue="spec", size="mag",
        palette=cmap_lr, hue_norm=(0,hue_max), size_norm=(0,size_max), edgecolor=".7",
        height=h, aspect=1, sizes=sizes, legend=True, 
    )
    if target_name is None:
        target_name = target
    title = f'$\\rightarrow$ {target_name}'
    g.set(xlabel="", ylabel="", xticklabels=xticklabels, title=None, aspect='equal')
    g._axes[0][0].set_title(title, fontsize=fs+2, color='darkblue')
    g.set_xticklabels(size = fs, rotation=rot_x)
    g.set_yticklabels(size = fs, style='italic')
    g.despine(left=True, bottom=True)
    g.ax.margins(x=mx, y=my)
    if leg_bbox is not None:
        for i in range(len(g._legend.texts)):
            if g._legend.texts[i].get_text() == 'spec':
                g._legend.texts[i].set_text('Spec.')
            if g._legend.texts[i].get_text() == 'mag':
                g._legend.texts[i].set_text('Mag.')
        sns.move_legend(
            g, "center", bbox_to_anchor=leg_bbox, fontsize=fs, ncols=1, frameon=False, alignment='left',
        )
    else:
        g._legend.remove()
    
    return concat_df
    

## draw lr pair gene expression spatial distribution
def draw_lr_spatial(crop, keep_niche_cell_ids, cts, genes,
                    ct_names = None, ligand_first = True, complex_fun=['min', 'max', 'mean', 'geom_mean'],
                    cmap1 = cmr.get_sub_cmap(mpl.cm.PiYG, 0.5, 1),
                    cmap2 = cmr.get_sub_cmap(mpl.cm.PiYG_r, 0.5, 1),
                    ms_bg = 4, color_bg = '#F0F0F0', lw_bg = 0, ms = 6, ecolor = '#A0A0A0', lw = 0.1,
                    leg_fs = 16, quantiles = [1, 1], figsize = (7,10), dpi = 300):

    fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=dpi)
    cmaps = [cmap1, cmap2]
    leg_cts = cts
    leg_genes = genes
    leg_cmaps = cmaps
    if not ligand_first:
        cts = cts[::-1]
        genes = genes[::-1]
        cmaps = cmaps[::-1]
        quantiles = quantiles[::-1]
    
    # background
    bg_df = crop[~crop.obs.cell_id.isin(keep_niche_cell_ids)].obs.copy()
    ax.scatter(bg_df['x'], bg_df['y'], ms_bg, facecolors=color_bg, edgecolor=color_bg, lw=lw_bg)
    # gene expression
    raw = False
    scores = []
    # first draw zero expression points
    for i, (ct, gene, cmap) in enumerate(zip(cts, genes, cmaps)):
        crop_tmp = crop[(crop.obs.cell_type==ct)&(crop.obs.cell_id.isin(keep_niche_cell_ids)), gene]
        if isinstance(gene, str):
            z = crop_tmp.X.toarray().flatten()
        else:
            z = crop_tmp.X.toarray()
            if complex_fun == 'min':
                z = z.min(1)
            elif complex_fun == 'max':
                z = z.max(1)
            elif complex_fun == 'mean':
                z = z.mean(1)
            elif complex_fun == 'geom_mean':
                z = np.sqrt(z[:,0]*z[:,1])
        if raw:
            z = np.exp(z) - 1
        tmp = pd.DataFrame({'x': crop_tmp.obs['x'].values,
                            'y': crop_tmp.obs['y'].values,
                            'z': z}).sort_values('z')
        tmp_ = tmp.loc[tmp.z==0]
        ecolor = mpl.colors.rgb2hex(cmap.colors[int(len(cmap.colors)*0.9)])
        ax.scatter(tmp_['x'], tmp_['y'], ms, tmp_['z'], cmap=cmap, alpha=1, edgecolor=ecolor, lw=lw)
        scores.append(tmp.loc[tmp.z>0])
    for i in range(len(cts)):
        sca_lims = {'vmin': 0, 'vmax': np.quantile(scores[i].z, quantiles[i])}
        ecolor = mpl.colors.rgb2hex(cmaps[i].colors[int(len(cmaps[i].colors)*0.9)])
        ax.scatter(scores[i]['x'], scores[i]['y'], ms, scores[i]['z'], cmap=cmaps[i], alpha=1, edgecolor=ecolor, lw=lw, **sca_lims)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ['top', 'bottom', 'right', 'left']:
        ax.spines[s].set_visible(False)
    
    legend_elements = []
    for i, (ct_name, gene, cmap) in enumerate(zip(leg_cts, leg_genes, leg_cmaps)):
        if ct_names is not None:
            ct_name = ct_names[i]
        label = f'{ct_name} ${gene}$'
        if isinstance(gene, list):
            label = f'{ct_name} ${gene[0]}$_${gene[1]}$'
        c = mpl.colors.rgb2hex(cmap.colors[int(len(cmap.colors)*0.6)])
        ecolor = mpl.colors.rgb2hex(cmap.colors[int(len(cmap.colors)*0.9)])
        ele = mlines.Line2D([], [], color='w', marker='o', markersize=12, 
                            markeredgecolor=ecolor, markerfacecolor=c, label=label)
        legend_elements.append(ele)
    leg = ax.legend(handles=legend_elements, bbox_to_anchor=(0.5,1.02), loc="center", 
                    borderpad=0, borderaxespad=0, ncol=2,
                    framealpha=0, markerscale=1, fontsize=leg_fs, handletextpad=0.3)
    
    ax.margins(0,0)
    plt.show()

    
