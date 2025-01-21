import os, sys, time, tqdm, importlib
import numpy as np
import pandas as pd
import pathlib
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import style
import seaborn as sns
import scanpy as sc
import spatialdata as sd
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
from legendkit import SizeLegend, Colorbar
from legendkit.layout import vstack, hstack
from rpy2.robjects.packages import importr
import rpy2.robjects as ro
from rpy2.robjects import numpy2ri, pandas2ri


### neighborhood analysis arcplot
def draw_arcplot(nodes, nhood_df, node1='from', node2='to', weights='weights', 
                 figsize=(7,3), dpi=200, bg_color='white', cmap='Oranges', title=None):
    
    # create the diagram
    arcdiag = ArcDiagram(nodes, figsize=figsize, dpi=dpi, title=title)
    
    # connect the nodes
    for connection in nhood_df.iterrows():
        arcdiag.connect(
            connection[1][node1],
            connection[1][node2],
            linewidth=connection[1][weights]
        )
    
    # custom colors
    arcdiag.set_background_color(bg_color)
    arcdiag.set_color_map(cmap)
    
    # plot the diagram
    arcdiag.show_plot()


### niche composition, cov test and cca
def run_method(crop, target_ct,  target_ct_abbr, 
               sigma=20, cutoff=0.05, self=False,
               max_cand_genes=None,
               n_hvg=3000, cov_thres=0.05, cov_test_null=True,
               use_cov_test_genes=True, sort_comp_by_corr=False,
               cca_comp=8, px=0.6, pz=0.4):

    ## rpy2
    r = ro.r
    importr('PMA')
    CCA = r['CCA']
    numpy2ri.activate()
    pandas2ri.activate()
    cov_test = r.source('cov_test.r')
    sparkx_sk = cov_test.rx2('value')

    t0 = time.time()
    ### preliminary
    print('***** Niche matrix *****')
    ## num cell id
    loc = crop.obs[['cell_id', 'x', 'y', 'cell_type']]
    num_cell_ID = pd.get_dummies(loc, columns=['cell_type'], dtype=float)
    group_names = [x.replace('cell_type_', '') for x in num_cell_ID.columns[3:].tolist()]
    print(f'{len(group_names)} cell_types.')
    num_cell_ID = num_cell_ID.iloc[:,3:].values
    print(f'num_cell_ID: {num_cell_ID.shape}.')

    ## calculate niche matrix
    target_ct_idx = group_names.index(target_ct)
    if self:
        use_group_ids = list(range(len(group_names)))
        use_groups = group_names
    else:
        use_group_ids = list(set(range(len(group_names))) - set([target_ct_idx]))
        use_groups = [x for x in group_names if x!= target_ct]
    print(f'Use other {len(use_groups)} cell types.')
    loc_target = crop.obs.loc[crop.obs.cell_type==target_ct, ['x', 'y']].values
    loc_all = crop.obs[['x','y']].values
    K_target = cdist(loc_target, loc_all)
    K_target = np.exp(-K_target**2/sigma**2)
    K_target[K_target < cutoff] = 0
    N_target = K_target @ num_cell_ID
    cell_ids_target = crop.obs.loc[crop.obs.cell_type==target_ct,'cell_id'].values.tolist()
    print(f'{len(loc_target)} {target_ct}, {len(loc_all)} cells in total.\n')

    ### cov test
    use_gene_names = crop.var.loc[crop.var.highly_variable_rank<n_hvg,].index.tolist()
    sparkx_X = crop[crop.obs.cell_type==target_ct,use_gene_names].X.toarray().T
    keep_gene_idx = np.where(sparkx_X.sum(axis=1)!=0)[0]
    keep_gene_names = [use_gene_names[i] for i in keep_gene_idx]
    cov_test_qqplot_data = None
    if use_cov_test_genes:
        print('***** Cov test *****')
        sparkx_X = sparkx_X[keep_gene_idx,:]
        sparkx_N = N_target[:,use_group_ids].astype('float32')
        keep_cell_idx = np.where(sparkx_N.sum(axis=1)!=0)[0]
        sparkx_X = sparkx_X[:,keep_cell_idx]
        sparkx_N = sparkx_N[keep_cell_idx,:]
        print(f'sparkx_X: {sparkx_X.shape}; sparkx_N: {sparkx_N.shape}')
    
        cov_target_ = sparkx_sk(sparkx_X, sparkx_N)
        cov_target = pd.concat([pd.DataFrame({'gene_id': np.array(keep_gene_names)[np.array(cov_target_.rx2('gene_ids'))-1], 
                                    'vec_stat': cov_target_.rx2('stats').flatten(),
                                    'vec_daviesp': cov_target_.rx2('res_stest').flatten()}),
                                ro.conversion.rpy2py(cov_target_.rx2('res_mtest')).reset_index(drop=True)], axis=1)
        print(f'{cov_target.shape[0]} / {len(keep_gene_names)} genes have pvalue.')

        ## cov test null distribution: 5 reps
        if cov_test_null:
            print('Cov test on permuted data...')
            cov_null_reps = []
            for r in range(5):
                np.random.seed(2*r+1)
                # rand_idx_X = np.random.choice(np.arange(sparkx_X.shape[1]), sparkx_X.shape[1], replace=False)
                sparkx_X_null = sparkx_X.copy().flatten()
                np.random.shuffle(sparkx_X_null)
                sparkx_X_null = sparkx_X_null.reshape(sparkx_X.shape)
                np.random.seed(2*r+2)
                rand_idx_N = np.random.choice(np.arange(sparkx_N.shape[0]), sparkx_N.shape[0], replace=False)
                # sparkx_X_null = sparkx_X[:,rand_idx_X]
                sparkx_N_null = sparkx_N[rand_idx_N,:]
                cov_target_null_ = sparkx_sk(sparkx_X_null, sparkx_N_null)
                cov_null_reps.append(cov_target_null_.rx2('res_stest').flatten())
            cov_test_qqplot_data = [cov_target.vec_daviesp.values] + cov_null_reps
        
        ## cand genes
        cov_target_sorted = cov_target.sort_values(by=['adjustedPval']).reset_index(drop=True)
        cand_genes_target = cov_target_sorted.loc[cov_target_sorted.adjustedPval<cov_thres,'gene_id'].values.tolist()
        print(f'{len(cand_genes_target)} genes with p_adj < {cov_thres}.')
        if max_cand_genes is not None:
            cand_genes_target = cand_genes_target[:max_cand_genes]
            print(f'Use top {len(cand_genes_target)} genes with p_adj < {cov_thres}.\n')
        else:
            print(f'Use all {len(cand_genes_target)} genes with p_adj < {cov_thres}.\n')
        cca_genes = cand_genes_target
    else:
        cca_genes = keep_gene_names

    ### CCA
    print('***** CCA *****')
        
    ## input
    cca_X = crop[crop.obs.cell_type==target_ct,cca_genes].X.toarray()
    cca_N = N_target[:,use_group_ids]
    keep_cell_idx = np.where(cca_N.sum(axis=1)!=0)[0]
    cca_X = cca_X[keep_cell_idx,:]
    cca_N = cca_N[keep_cell_idx,:]
    print(f'cca_X: {cca_X.shape}; cca_N: {cca_N.shape}')

    ## nonneg cca
    pmd = CCA(cca_X, cca_N, K=cca_comp,
              typex="standard",typez="standard",standardize=True,
              penaltyx=px,penaltyz=pz,trace=False,upos=True,vpos=True)
    u = pmd.rx2('u')
    v = pmd.rx2('v')
    cors = pmd.rx2('cors')
    if sort_comp_by_corr:
        comp_order = np.argsort(cors)[::-1]
        cors = cors[comp_order]
        u = u[:,comp_order]
        v = v[:,comp_order]
    print(f"Nonneg CCA cors: {np.round(cors,3)}")

    
    ## post proc
    udf = pd.DataFrame(u, index=cca_genes, columns=[f'comp{i}' for i in range(1,cca_comp+1)])
    vdf = pd.DataFrame(v, index=use_groups, columns=[f'comp{i}' for i in range(1,cca_comp+1)])
    print(f'udf: {udf.shape}; vdf: {vdf.shape}.\n')
    t1 = time.time()
    print(f'{target_ct}: Finished in {t1 - t0:.1f}s.')

    ## other variables to return
    meta = {'N_target': N_target,
            'loc_target': loc_target,
            'group_names': group_names,
            'target_ct': target_ct,
            'target_ct_abbr': target_ct_abbr,
            'target_ct_idx': target_ct_idx,
            'use_group_ids': use_group_ids,
            'use_groups': use_groups,
            'cca_genes': cca_genes,
            'cors': cors,
            'cca_comp': cca_comp,
            'px': px,
            'pz': pz,
            'sigma': sigma,
            'cutoff': cutoff,
            'cov_test': use_cov_test_genes,
            'cov_thres': cov_thres,
            'max_cand_genes': max_cand_genes,
            'n_hvg': n_hvg,
            'cov_test_qqplot_data': cov_test_qqplot_data}

    return udf, vdf, meta


### niche score
def compute_score(crop, udf, vdf, target_ct, N_target, loc_target, use_group_ids):

    genes = udf.index.tolist()
    U = udf.values
    V = vdf.values
    ct_df = crop[crop.obs.cell_type==target_ct].obs[['cell_type', 'sub_cell_type']].reset_index(drop=True)
    X0 = crop[crop.obs.cell_type==target_ct,genes].X.toarray()
    X1 = (X0 - np.mean(X0, axis=0)) / np.std(X0, axis=0)
    N0 = N_target[:,use_group_ids]
    N1 = (N0 - np.mean(N0, axis=0)) /  np.std(N0, axis=0)
    ## niche gene score & niche ct score
    Y1 = X1 @ U
    Y2 = N1 @ V
    X1_df = pd.DataFrame(X0, columns=udf.index.values)
    Y1_df = pd.DataFrame(Y1, columns=[f'Xu_{x}' for x in list(udf.columns)])
    Y2_df = pd.DataFrame(Y2, columns=[f'Nv_{x}' for x in list(udf.columns)])
    XY_df = pd.concat([X1_df, Y1_df, Y2_df], axis=1)
    ## concord pos
    concord_pos = {}
    for comp in udf.columns:
        y1 = Y1_df[f'Xu_{comp}'].values
        y2 = Y2_df[f'Nv_{comp}'].values
        y1_pos = np.maximum(y1, 0)
        y2_pos = np.maximum(y2, 0)
        concord_pos[f'concord_{comp}'] = np.log1p(y1_pos * y2_pos)
    concord_pos_df = pd.DataFrame(concord_pos)
    loc_df = pd.DataFrame(loc_target, columns=['x','y'])
    XY_loc_df = pd.concat([XY_df, concord_pos_df, loc_df, ct_df], axis=1)
    print(f'Computed niche-related scores: XY_loc_df {XY_loc_df.shape}.')
    
    return XY_loc_df


### niche cell type circular plot
def draw_v_circular(vdf, comps, v_thres=0.1, cmap=mpl.colormaps['Set2'], figsize=(5,5), dpi=200, title=None):
    
    categories = vdf.index.tolist()
    comps = [f'comp{x}' for x in comps]
    angles = np.linspace(0, 2*np.pi, len(categories), endpoint=False)
    colors = [cmap.colors[i] for i in range(len(comps))]
    
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi, subplot_kw=dict(projection='polar'))
    alpha = 0.6
    width = 0.3
    offset = 2*np.pi/72
    counter = {k:0  for k in categories}
    for i, comp in enumerate(comps):
        angles_ = np.zeros(len(categories))
        angles = np.linspace(0, 2*np.pi, len(categories), endpoint=False)
        keys_ = vdf.loc[vdf[comp]>v_thres].index.tolist()
        index_ = [categories.index(x) for x in keys_]
        for j, k in zip(index_, keys_):
            angles_[j] = angles[j] + counter[k] * offset
            counter[k] += 1
        values_ = vdf[comp].values
        values_[values_ <= v_thres] = 0
        ax.bar(angles_, values_, width=width, align='center', label=i+1, facecolor=list(colors[i])+[alpha], edgecolor=colors[i])
        
    ax.set_xticks(angles)
    ax.set_xticklabels(categories)
    angs = np.linspace(0,2*np.pi,len(ax.get_xticklabels())+1)
    angs[np.cos(angs) < 0] = angs[np.cos(angs) < 0] + np.pi
    angs = np.rad2deg(angs)
    labels = []
    for label, angle in zip(ax.get_xticklabels(), angs):
        x,y = label.get_position()
        lab = ax.text(x,y-.015*len(label.get_text()), label.get_text(), transform=label.get_transform(),
                      ha=label.get_ha(), va=label.get_va(), fontsize=10)
        lab.set_rotation(angle)
        labels.append(lab)
    ax.set_xticklabels([])
    
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), title='Factor', fontsize=10, title_fontsize=12, alignment='left')
    plt.suptitle(title, y=1.1, fontsize=14)
    plt.show()


### niche cell type heatmap
def draw_v_heatmap(vdf, figsize=(3, 6), dpi=200, title=''):
    
    ## v plot
    data = vdf.values
    cbarlabel = ''
    col_labels = list(vdf.columns)
    row_labels = list(vdf.index)
    threshold = 0.2
    valfmt="{x:.2f}"
    # textcolors=("black", "white")
    textcolors=('grey', 'black')
    
    fs = 7
    fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=dpi)
    # Plot the heatmap
    cmap = mpl.cm.Reds
    cmap.set_bad('white',1.)
    im = ax.imshow(data, cmap=cmap, vmin=0, vmax=1, aspect="auto")
    
    # # Create colorbar
    # cbar = ax.figure.colorbar(im, ax=ax, fraction=0.03, anchor=(1,1), aspect=15)
    # cbar.outline.set_linewidth(0)
    # cbar.ax.tick_params(labelsize=fs)
    # cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")
    
    # Show all ticks and label them with the respective list entries.
    ax.set_xticks(np.arange(data.shape[1]), labels=col_labels, fontsize=fs)
    ax.set_yticks(np.arange(data.shape[0]), labels=row_labels, fontsize=fs)
    
    # Let the horizontal axes labeling appear on bottom.
    ax.tick_params(top=False, bottom=True, labeltop=False, labelbottom=True)
    
    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=90, ha="right", rotation_mode="anchor")
    
    # Turn spines off and create white grid.
    ax.spines[:].set_visible(False)
    
    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=1)
    ax.tick_params(which="minor", bottom=False, left=False)
    
    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()
    
    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max())/2.
    
    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center",
             fontsize=fs)
    # kw.update(textkw)
    
    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = mpl.ticker.StrMethodFormatter(valfmt)
    
    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)
    
    plt.title(title, fontsize=fs+1)


### niche sub cell type count and enrichment fold
def sub_ct_enrich(comp, score_df, subcts, score='concord', q1=0.9, q2=0.95):

    col = f'{score}_comp{comp}'
    quant1 = np.quantile(score_df[col], q1)
    quant2 = np.quantile(score_df[col], q2)
    high_df1 = score_df.loc[score_df[col]>quant1]
    high_df2 = score_df.loc[score_df[col]>quant2]
    
    base_count = score_df.groupby('sub_cell_type').size().loc[subcts]
    high_count1 = high_df1.groupby('sub_cell_type').size().loc[subcts]
    high_count2 = high_df2.groupby('sub_cell_type').size().loc[subcts]
    count_df = pd.DataFrame({'base': base_count, 'obs_high_09': high_count1, 'obs_high_095': high_count2})
    count_df['exp_high_09'] = (count_df['base'] * count_df['obs_high_09'].sum() / count_df['base'].sum()).astype(int)
    count_df['exp_high_09'] = np.maximum(count_df['exp_high_09'], 1)
    count_df['exp_high_095'] = (count_df['base'] * count_df['obs_high_095'].sum() / count_df['base'].sum()).astype(int)
    count_df['exp_high_095'] = np.maximum(count_df['exp_high_095'], 1)
    count_df['fold_09'] = count_df['obs_high_09'] / count_df['exp_high_09']
    count_df['fold_095'] = count_df['obs_high_095'] / count_df['exp_high_095']
    count_df = count_df.sort_values('fold_095', ascending=False)
    
    return count_df


### niche sub cell type prop vs base prop stacked barplot
def draw_sub_ct_prop(count_df, alphabet=False, figsize=(8,1), dpi=300, h=0.35, yticks=[0, 0.4], 
                     yticklabels=['Base', 'Factor 1'], colors=mpl.colormaps['Set2'].colors, ylim=(-0.22,0.62)):

    sub_cts = list(count_df.index)
    tmp = count_df.copy()
    if alphabet:
        sub_cts = sorted(sub_cts)
    tmp = tmp.loc[sub_cts]
    prop0 = tmp['base'] / tmp['base'].sum()
    prop1 = tmp['obs_high_095'] / tmp['obs_high_095'].sum()
    props = np.vstack([prop0, prop1])
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    start0, start1 = 0, 0
    for i in range(len(sub_cts)):
        ax.barh(yticks, props[:,i], left=[start0, start1], height=h, label=sub_cts[i], color=colors[i])
        start0 += props[0,i]
        start1 += props[1,i]
    ax.xaxis.set_visible(False)
    ax.set_xlim(0, 1)
    ax.set_ylim(ylim)
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticklabels)
    ax.invert_yaxis()
    for x in ['top','bottom','left','right']:
        ax.spines[x].set_visible(True)
    ax.legend(ncols=len(sub_cts), bbox_to_anchor=(-0.01, 0), loc='upper left', fontsize='small')


### niche sub cell type enrichment fold barplot
def draw_sub_ct_enrich(count_df, figsize=(8,5), dpi=200, fs=14, width=0.4, alpha=1, title=None):

    subcts = list(count_df.index)
    style.use('seaborn-v0_8-white')
    sns.set_context('paper', font_scale=1)
    fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=dpi)
    nct = len(subcts)
    bar1 = ax.bar(np.arange(1,nct+1)-0.5*width, count_df['fold_09'].values, width=width, color='#9A9DE6', alpha=alpha, label='90th quantile')
    bar2 = ax.bar(np.arange(1,nct+1)+0.5*width, count_df['fold_095'].values, width=width, color='#8CC17E', alpha=alpha, label='95th quantile')
    ax.axhline(y=1, linestyle='dashed', linewidth=1, color='darkblue')
    ax.legend(bbox_to_anchor=(1.0, 1.0),fontsize=fs,handletextpad=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_ylabel('Observed / Expected', fontsize=fs+2)
    ax.set_xticks(np.arange(1,nct+1))
    ax.set_xticklabels(count_df.index.tolist(), rotation=45, ha='right')
    ax.tick_params(axis='both', which='major', labelsize=fs)
    ax.set_title(title, fontsize=fs+2)
    plt.show()
    

### niche gene dotmap
def draw_u_dotmap(udf, focus_comp, draw_comps, n_top_gene=10, sizes=(5, 150), xticklabels=None,
                  fs=11, h=None, w=3, title=None, mx=0.2, my=0.03, leg_pos=(1.1,0.95), dpi=200):

    focus_comp = f'comp{focus_comp}'
    draw_comps = [f'comp{k}' for k in draw_comps]
    ucolor_df0 = udf[draw_comps].sort_values(focus_comp, ascending=False).iloc[:n_top_gene,:]
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
        palette="vlag", hue_norm=(-csm, csm), edgecolor=".7",
        height=h, aspect=asp, sizes=sizes, size_norm=(0, csm), legend=True, 
    )
    g.set(xlabel="", ylabel="", xticklabels=xticklabels, title=title)
    
    g.set_xticklabels(size = fs, rotation=90)
    g.set_yticklabels(size = fs, style='italic')
    g.despine(left=True, bottom=True)
    g.ax.margins(x=mx, y=my)
    
    sns.move_legend(
        g, "upper right", bbox_to_anchor=leg_pos, fontsize=fs, ncols=1, frameon=False,
    )
    

### niche gene heatmap
def draw_u_heatmap(score_df, udf, vdf, focus_comp, draw_comps, score=['Xu', 'concord', 'Nv'],
                   v_thres=0.2, u_thres=0.05, q=0.8,
                   cmap=mpl.colormaps['Reds'], sizes=(1,200), 
                   label_font=8, title_font=8, figsize=(2,5)):
    
    niche_cts = vdf.loc[vdf[focus_comp]>v_thres].index.tolist()
    genes = udf.loc[udf[focus_comp]>u_thres].index.tolist()

    ## color
    ucolor_df = udf.loc[udf[focus_comp]>u_thres, draw_comps]
    ucolor = ucolor_df.values
    # ucolor_norm = ucolor/ucolor.max()
    color_mapper = ScalarMappable(cmap=cmap, norm=mpl.colors.Normalize(vmin=0,vmax=ucolor.max()))
    color_ticks = np.round(np.linspace(0,ucolor.max(),5),2)

    ## size
    dfs = []
    for comp in draw_comps:
        col = f'{score}_{comp}'
        quant = np.quantile(score_df[col], q)
        df = pd.DataFrame({comp: score_df.loc[score_df[col]>quant, genes].mean(0)})
        dfs.append(df)
    usize_df = pd.concat(dfs, axis=1)
    usize = usize_df.values
    usize_norm = usize/usize.max()
    print(usize_norm.shape, ucolor.shape)

    ## draw
    u_heatmap(usize_norm, ucolor, cmap, color_mapper, color_ticks,
              marker='o', marker_shape='circle', sizes=sizes,
              yticklabels=genes, xticklabels=draw_comps, 
              xtick_rot=0, ytick_rot=0, label_font=8, title_font=8,
              title=f"{focus_comp}: {','.join(niche_cts)} (u>{u_thres})", figsize=figsize)


### niche score spatial distribution by kernel density estimation
def kde_diff(score_df, comp, score=['Xu', 'Nv', 'concord'], z_quantile=0.99, bw=20):
    
    comp = f'comp{comp}'
    x = score_df.x.values
    y = score_df.y.values
    z = score_df[f'{score}_{comp}'].values
    xy = np.vstack([x, y]).T # coordinates of points
    keep_ids = np.where(z <= np.quantile(z, z_quantile))[0]
    xy = xy[keep_ids,:]
    z = z[keep_ids]
    z = (z - z.min()) / (z.max() - z.min())
    xmin, ymin = xy.min(0)
    xmax, ymax = xy.max(0)
    xgrid, ygrid = np.mgrid[xmin:xmax:200j, ymin:ymax:200j]
    grid_coords = np.vstack([xgrid.ravel(), ygrid.ravel()]).T
    kde1 = neighbors.KernelDensity(kernel='gaussian', bandwidth=bw).fit(xy, sample_weight=z)
    kde0 = neighbors.KernelDensity(kernel='gaussian', bandwidth=bw).fit(xy)
    Z1 = kde1.score_samples(grid_coords)
    Z0 = kde0.score_samples(grid_coords)
    Z1 = np.reshape(np.exp(Z1), xgrid.shape)
    Z0 = np.reshape(np.exp(Z0), xgrid.shape)
    Z = Z1 - Z0
    Z = Z - Z.mean()
    
    return Z, (xmin, xmax, ymin, ymax), keep_ids

    
### draw niche score spatial distribution
def draw_kde_diff(score_df, comp, target_ct, Z, lims, draw_ids, score=['Xu', 'Nv', 'concord'],
                  use_quant=0.99, draw_points=True, cmap=mpl.colormaps['coolwarm'], 
                  ylabel=None, fs=12, title=None, figsize=(6, 6), dpi=200):
    
    ## kde diff
    comp = f'comp{comp}'
    coord_x = score_df.x.values[draw_ids]
    coord_y = score_df.y.values[draw_ids]
    values = score_df[f'{score}_{comp}'].values[draw_ids]

    ## draw
    fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=dpi)
    im_lims = {'vmin': Z.min(), 'vmax': Z.max()}
    sca_lims = {'vmin': 0, 'vmax': max(np.floor(values.max()), 1)}
    ax.imshow(np.rot90(Z), cmap=cmap, extent=lims, **im_lims)
    if draw_points:
        ax.scatter(coord_x, coord_y, 4, values, edgecolor='w', lw=0.1, cmap=cmap, **sca_lims)
    ax.set_xlim([lims[0], lims[1]])
    ax.set_ylim([lims[2], lims[3]])
    ax.invert_yaxis()
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    for s in ['top', 'bottom', 'right', 'left']:
        ax.spines[s].set_visible(False)
    # ax.set_title(f'{target_ct}: {comp} ({score})')
    # ax.set_title(title)
    ax.set_ylabel(ylabel, fontsize=fs)
    # plt.axis('off')
    plt.show()


### go analysis of selected factors
def go_combine(udf, vdf, draw_comps, u_thres=None, u_top=None, pval_thres=0.05, n_top=10, enrichr_cutoff=0.1):

    assert (u_thres is None) + (u_top is None), 'Provide either u threshold or number of top genes!'
    if u_thres is not None:
        assert isinstance(u_thres, list), 'u_thres should be a list!'
    ## combine ora results
    comp_dfs = {}
    for i, k in enumerate(draw_comps):
        comp = f'comp{k}'
        ctlist = vdf.loc[vdf[comp] > 0.2].index.tolist()
        if u_thres is not None:
            glist = udf.loc[udf[comp] > u_thres[i]].index.tolist()
        if u_top is not None:
            glist = udf.sort_values(comp, ascending=False).iloc[:u_top].index.tolist()
        enr_res = gseapy.enrichr(gene_list=glist, organism='Human', gene_sets=['GO_Biological_Process_2018'], cutoff = enrichr_cutoff)
        comp_df = enr_res.res2d.loc[enr_res.res2d['Adjusted P-value']<pval_thres].loc[:(n_top-1),['Term','Overlap','P-value','Adjusted P-value','Genes']]
        comp_df['Comp'] = i + 1
        comp_dfs[comp] = comp_df
        print(k, ctlist, len(glist), comp_df.shape[0])
    df = pd.concat(list(comp_dfs.values()), axis=0)
    ldf = df.copy()
    ldf['-log10_padj'] = -np.log10(ldf['Adjusted P-value'])
    ldf['num_gene'] = ldf.Overlap.apply(lambda x: int(x.split('/')[0]))
    ldf.Term = ldf.Term.apply(lambda x: x.split(' (GO:')[0].capitalize())
    ldf.Term=pd.Categorical(ldf.Term,categories=ldf.Term.unique(),ordered=True)
    ldf = ldf.reset_index(drop=True)

    wdf_padj = ldf.pivot(index='Term', columns='Comp', values='-log10_padj')
    wdf_overlap = ldf.pivot(index='Term', columns='Comp', values='num_gene')
    wdf = ldf.pivot(index='Term', columns='Comp', values=['-log10_padj', 'num_gene']).fillna(0)
    wdf1 = wdf['-log10_padj'].stack().reset_index(name='-log10_padj')
    wdf2 = wdf['num_gene'].stack().reset_index(name='num_gene')
    wdf12 = wdf1.copy()
    wdf12['num_gene'] = wdf2['num_gene']
    wdf12['Comp'] = wdf12['Comp'].astype(str)
    wdf12['-log10_padj'].min(),wdf12['-log10_padj'].max(),wdf12['num_gene'].min(),wdf12['num_gene'].max()

    return wdf12


### draw go result dotmap
def draw_go_combine(wdf12, draw_comps, dpi=200, sizes=(5,300), height=10, hue_norm=None, size_norm=None, fs=12, mx=0.2, my=0.02, leg_pos=(1,1), title=None):

    xticklabels = [f"Factor {x}" for x in range(1,len(draw_comps)+1)]
    plt.rcParams.update({"figure.dpi": dpi})
    sns.set_theme(style="white")
    sns.set_context('paper',font_scale=1.)
    g = sns.relplot(
        data=wdf12, 
        x="Comp", y="Term", hue="-log10_padj", size="num_gene", marker='s',
        palette='light:seagreen', hue_norm=hue_norm, edgecolor=".5", linewidth=0.8,
        height=height, aspect=1, sizes=sizes, size_norm=size_norm
    )
    g.set(xlabel="", ylabel="", xticklabels=xticklabels, aspect="equal")
    g.set_xticklabels(size = fs, rotation=90)
    g.set_yticklabels(size = fs)
    g.despine(left=True, bottom=True)
    g.ax.margins(x=mx, y=my)
    sns.move_legend(
        g, "upper right", fontsize=fs, 
        bbox_to_anchor=leg_pos, title=title, frameon=False,
    )


### QQ plot
def qqplot(data, labels, n_quantiles=200, alpha=0.95, error_type='theoretical', 
           distribution = 'binomial', log10conv=True, 
           color=['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'tab:brown', 'C9', 'tab:olive', 'tab:cyan', 'tab:gray'], 
           shape=['.','+','.','+','.','+'],
           fill_dens=[0.1 for _ in range(10)], type = 'uniform', title='None', ms=5, legloc=2, xlim=None, ylim=None, 
           showXlabel=True, showYlabel=True, showXticks=True, showYticks=True, showLeg=True, ax=None):
    '''
    Function for plotting Quantile Quantile (QQ) plots with confidence interval (CI)
    :param data: NumPy 1D array with data
    :param labels:
    :param type: type of the plot
    :param n_quantiles: number of quntiles to plot
    :param alpha: confidence interval
    :param distribution: beta/normal/binomial -- type of the error estimation. Most common in the literature is 'beta'.
    :param log10conv: conversion to -log10(p) for the figure
    :return: nothing
    '''
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
            ax.plot(-np.log10(q_th[n_quantiles-1:]), -np.log10(q_data[n_quantiles-1:]), '-', color=color[j],alpha=0.7)
            ax.plot(-np.log10(q_th[:n_quantiles]), -np.log10(q_data[:n_quantiles]), '.', color=color[j], marker=shape[j], markersize=ms, label=labels[j],alpha=0.7)
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
        ax.set_xlabel('Theoretical -log10')
    if showYlabel:
        ax.set_ylabel('Experimental -log10')
    ax.plot([0, 100], [0, 100],'--k',linewidth=0.5)
    if xlim is None:
        ax.set_xlim([0, np.ceil(xmax)])
    else:
        ax.set_xlim(xlim)
    if ylim is None:
        ax.set_ylim([0, np.ceil(ymax)])#np.ceil(ymax*1.05)])
    else:
        ax.set_ylim(ylim)
    ax.set_title(title)
    if not showXticks:
        ax.set_xticks([])
    if not showYticks:
        ax.set_yticks([])


### Arc diagram
class ArcDiagram:
    def __init__(self, nodes, figsize, dpi, title=None):

        self.__nodes = nodes
        self.__title = title
        self.__figsize = figsize
        self.__dpi = dpi
        self.__arc_coordinates = []
        self.__linewidths = []
        # self.__colors = plt.cm.viridis(np.linspace(0, 1, len(self.__nodes)))
        self.__colors = None
        self.__background_color = "white"
        self.__label_rotation_degree = 45
        self.__legend_labels = []

    def connect(self, start_node, end_node, linewidth=0.1, arc_position="above"):
        start = self.__nodes.index(start_node)
        end = self.__nodes.index(end_node)

        arc_center = (start + end) / 2
        radius = abs(end - start) / 2

        if arc_position == "below":
            theta = np.linspace(180, 360, 100)
        else:
            theta = np.linspace(0, 180, 100)

        x = arc_center + radius * np.cos(np.radians(theta))
        y = radius * np.sin(np.radians(theta))
        self.__arc_coordinates.append((x, y, start, linewidth))
        self.__linewidths.append(linewidth)

    def help(self):
        function_list = """
        ArcDiagram(node_list, title_string)
        .set_background_color(string)
        .set_color_map(string)
        .set_custom_colors(color_list)
        .set_label_rotation_degree(arc_degree)
        .set_legend_labels(list_of_labels)
        .connect(start, end, linewidth=100, arc_position="below")
        .show_plot()
        .save_plot_as(file_name, resolution="100")
        """
        print(function_list)

    def set_background_color(self, color):
        self.__background_color = color

    def set_color_map(self, color_map_name):
        color_map = colormaps[color_map_name]
        # self.__colors = color_map(np.linspace(0, 1, len(self.__nodes)))
        self.__colors = color_map(np.linspace(0.1, 1, int(max(self.__linewidths))))

    def set_custom_colors(self, color_list):
        self.__colors = ListedColormap(color_list).colors

    def set_label_rotation_degree(self, degree):
        self.__label_rotation_degree = degree

    def set_legend_labels(self, legend_labels):
        self.__legend_labels = legend_labels

    def save_plot_as(self, file_name, resolution="figure"):
        fig, ax = self.__plot()
        plt.savefig(file_name, dpi=resolution, bbox_inches="tight")

    def show_plot(self):
        fig, ax = self.__plot()
        plt.show()

    def __label_color_distribution(self, colors, n):
        if n <= 0:
            return []

        step = (len(colors) - 1) / (n - 1)
        indices = [round(i * step) for i in range(n)]
        return [colors[i] for i in indices]

    def __plot(self):
        fig, ax = plt.subplots(figsize=self.__figsize, dpi=self.__dpi)
        ax.set_facecolor(self.__background_color)

        # plot nodes as points
        node_positions = np.arange(len(self.__nodes))
        ax.scatter(
            node_positions, np.zeros_like(node_positions), color='white', edgecolor='k', s=100, linewidth=2,
        )

        max_value = max(self.__arc_coordinates, key=itemgetter(3))[3]
        # plot connections as arcs
        for x, y, index, raw_linewidth in self.__arc_coordinates:
            if self.__colors is not None:
                ax.plot(
                    x,
                    y,
                    color=self.__colors[max(int(raw_linewidth)-1,0)],
                    zorder=1,
                    linewidth=self._map_to_linewidth(raw_linewidth, max_value),
                )
            else:
                ax.plot(
                    x,
                    y,
                    color='k',
                    zorder=1,
                    linewidth=self._map_to_linewidth(raw_linewidth, max_value),
                )
            # print(index, raw_linewidth)

        plt.xticks(rotation=self.__label_rotation_degree, ha='right')
        ax.set_xticks(node_positions)
        ax.set_xticklabels(self.__nodes)
        ax.set_yticks([])
        ax.set_title(self.__title)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)

        if self.__legend_labels != []:
            legend_labels = self.__legend_labels
            label_colors = self.__label_color_distribution(
                self.__colors, len(legend_labels)
            )
            ax.legend(
                handles=[
                    plt.Line2D(
                        [0],
                        [0],
                        marker="o",
                        color="w",
                        label=label,
                        markerfacecolor=label_colors[i],
                        markersize=10,
                    )
                    for i, label in enumerate(legend_labels)
                ],
                loc="upper right",
            )

        return fig, ax

    def _map_to_linewidth(self, value, max_value):
        if value < 1:
            return 1
        else:
            return (10 * value) / max_value
        


    def create_arc_plot(
        df: pd.DataFrame,
        start_node: str,
        end_node: str,
        weights=None,
        positions=None,
        invert_positions: bool = False,
        bg_color="white",
        cmap="viridis",
        title="My Diagram",
    ):
        """
        Wrapper for the ArcDiagram class, which creates diagrams from a pandas dataframe.
        Args:
            df (pd.DataFrame): The dataframe containing the data.
            start_node (str): The name of the column containing the start node.
            end_node (str): The name of the column containing the end node.
            weights (str, optional): The name of the column containing the weights. Defaults to None.
            positions (str, optional): The name of the column containing the positions. Defaults to None.
            invert_positions (bool, optional): Whether to invert the positions. Defaults to False.
            bg_color (str, optional): The background color. Defaults to 'white'.
            cmap (str, optional): The color map. Defaults to 'viridis'.
            title (str, optional): The title of the diagram. Defaults to 'My Diagram'.
        Raises:
            ValueError: If start_node or end_node are not columns in the dataframe.
            ValueError: If start_node and end_node do not have the same length.
            ValueError: If positions is not a column in the dataframe.
            ValueError: If positions does not have 1 or 2 unique values.
            ValueError: If weights is not a column in the dataframe.
        """
    
        data = df.copy()
    
        if start_node not in data.columns or end_node not in data.columns:
            raise ValueError("start_node and end_node must be columns in the dataframe")
    
        if len(data[start_node]) != len(data[end_node]):
            raise ValueError("start_node and end_node must have the same length")
    
        # get all unique nodes
        nodes = data[start_node].unique().tolist() + data[end_node].unique().tolist()
        nodes = list(set(nodes))
    
        # initialize the diagram
        arcdiag = ArcDiagram(nodes, title)
    
        # get positions
        if positions:
            if positions not in data.columns:
                raise ValueError("positions must be a column in the dataframe")
            else:
                n_positions = data[positions].nunique()
                if n_positions not in [1, 2]:
                    raise ValueError("positions must have 1 or 2 unique values")
                else:
                    if n_positions == 1:
                        position_map = {data[positions].unique()[0]: "above"}
                    else:
                        position_map = {
                            data[positions].unique()[0]: "above",
                            data[positions].unique()[1]: "below",
                        }
                    data[positions] = data[positions].map(position_map)
    
                    if invert_positions:
                        data[positions] = data[positions].map(
                            {"below": "above", "above": "below"}
                        )
        else:
            data[positions] = "above"
    
        # get weights
        if not weights:
            data[weights] = 0.1
        else:
            if weights not in data.columns:
                raise ValueError("weights must be a column in the dataframe")
    
        # connect the nodes
        for connection in data.iterrows():
            arcdiag.connect(
                connection[1][start_node],
                connection[1][end_node],
                linewidth=connection[1][weights],
                arc_position=connection[1][positions],
            )
    
        # custom colors
        arcdiag.set_background_color(bg_color)
        arcdiag.set_color_map(cmap)

        return arcdiag

    def show_arc_plot(
        df: pd.DataFrame,
        start_node: str,
        end_node: str,
        weights=None,
        positions=None,
        invert_positions: bool = False,
        bg_color="white",
        cmap="viridis",
        title="My Diagram",
    ):
        arc_diagram = create_arc_plot(
            df,
            start_node,
            end_node,
            weights,
            positions,
            invert_positions,
            bg_color,
            cmap,
            title,
        )
    
        # plot the diagram
        arc_diagram.show_plot()
    
    def save_arc_plot_as(
        df: pd.DataFrame,
        start_node: str,
        end_node: str,
        file_name: str,
        weights=None,
        positions=None,
        invert_positions: bool = False,
        bg_color="white",
        cmap="viridis",
        title="My Diagram",
        resolution="figure"
    ):
        arc_diagram = create_arc_plot(
            df,
            start_node,
            end_node,
            weights,
            positions,
            invert_positions,
            bg_color,
            cmap,
            title,
        )
    
        arc_diagram.save_plot_as(file_name, resolution)
    
    ### Dot heatmap
    def set_default(arg, default):
        if arg is None:
            return default
        else:
            return arg


