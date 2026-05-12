import os, sys, time, pickle, json, logging
from datetime import datetime
import numpy as np
import pandas as pd
import scanpy as sc
import squidpy as sq
import scipy.stats as ss
import matplotlib.pyplot as plt
import cmasher as cmr
import matplotlib as mpl
from matplotlib import style
from matplotlib.patches import Patch, Wedge, Rectangle, Circle
from matplotlib.lines import Line2D
import seaborn as sns
from scipy.spatial import cKDTree
from scipy.stats import pearsonr, spearmanr
from scipy.spatial.distance import braycurtis, jensenshannon, cosine
from importlib import reload
from scipy.optimize import linear_sum_assignment
from sklearn.metrics.pairwise import cosine_similarity


## data alignment
def calculate_translation(adata_rep1, adata_rep3, target_type='tumor',
                                coord_x_key_rep1='x', coord_y_key_rep1='y',
                                coord_x_key_rep3='x', coord_y_key_rep3='y'):

    df1_all = adata_rep1.obs[[coord_x_key_rep1, coord_y_key_rep1]]
    df3_all = adata_rep3.obs[[coord_x_key_rep3, coord_y_key_rep3]]
    if target_type is not None:
        df1 = adata_rep1.obs[adata_rep1.obs['cell_type'] == target_type][[coord_x_key_rep1, coord_y_key_rep1]]
        df3 = adata_rep3.obs[adata_rep3.obs['cell_type'] == target_type][[coord_x_key_rep3, coord_y_key_rep3]]
    else:
        df1 = df1_all
        df3 = df3_all
    
    centroid1 = (df1_all.max() + df1_all.min()).values / 2
    centroid3 = (df3_all.max() + df3_all.min()).values / 2
    
    delta_x = centroid1[0] - centroid3[0]
    delta_y = centroid1[1] - centroid3[1]

    return delta_x, delta_y


def align_spatial_coordinates(adata, delta_x, delta_y, x_col='x', y_col='y', copy=True):

    if copy:
        adata = adata.copy()

    adata.obs[f'{x_col}_aligned'] = adata.obs[x_col] + delta_x
    adata.obs[f'{y_col}_aligned'] = adata.obs[y_col] + delta_y
    
    if 'spatial' in adata.obsm:
        # Assuming 'spatial' follows [X, Y] convention
        adata.obsm['spatial_aligned'] = adata.obsm['spatial'].copy().astype('float64')
        adata.obsm['spatial_aligned'][:, 0] += delta_x
        adata.obsm['spatial_aligned'][:, 1] += delta_y
    
    return adata


## grid visualization
def plot_spatial_cell_types_with_grid(
    adata, x_col, y_col, type_col, 
    x_min, y_min, x_max, y_max,
    step=400, bin_factor=1, alpha=1,
    ax=None, title="", draw_grid=False,
    palette="tab20", dot_size=0.5,
    title_fs=24, leg_fs=16, leg_labelspacing=0.3,
    ct_name_map=None,
    draw_grid_demo=False,         
    grid_demo_step=None,          
    grid_demo_bin_factor=1,       
    grid_demo_lw=1,               
    grid_demo_alpha=0.8,          
    grid_demo_color='black',      
    grid_demo_ls='--'             
):

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(10, 7), dpi=300)

    coarse_step = step * bin_factor
    
    n_grids_x = int(np.ceil((x_max - x_min) / coarse_step))
    n_grids_y = int(np.ceil((y_max - y_min) / coarse_step))
    
    grid_end_x = x_min + n_grids_x * coarse_step
    grid_end_y = y_min + n_grids_y * coarse_step

    plot_df = adata.obs[[x_col, y_col, type_col]].copy()
    
    if ct_name_map is not None:
        target_order = [ct_name_map[k] for k in ct_name_map.keys() if k in plot_df[type_col].values]
        plot_df[type_col] = plot_df[type_col].astype(str).map(ct_name_map).fillna(plot_df[type_col])
        plot_df[type_col] = pd.Categorical(plot_df[type_col], categories=target_order, ordered=True)
        if isinstance(palette, dict):
            palette = {ct_name_map.get(k, k): v for k, v in palette.items()}

    sns.scatterplot(
        data=plot_df, 
        x=x_col, y=y_col, hue=type_col, 
        ax=ax, s=dot_size, palette=palette, 
        edgecolor=None, alpha=alpha, legend='full'
    )

    if draw_grid:
        line_style = dict(color='black', lw=1, alpha=0.5, linestyle='solid')
        for i in range(n_grids_x + 1):
            x = x_min + i * coarse_step
            ax.plot([x, x], [y_min, grid_end_y], **line_style)
        for j in range(n_grids_y + 1):
            y = y_min + j * coarse_step
            ax.plot([x_min, grid_end_x], [y, y], **line_style)

    if draw_grid_demo:
        demo_step = step if grid_demo_step is None else grid_demo_step
        demo_coarse_step = demo_step * grid_demo_bin_factor

        n_demo_x = int(np.ceil((x_max - x_min) / demo_coarse_step))
        n_demo_y = int(np.ceil((y_max - y_min) / demo_coarse_step))
        demo_end_x = x_min + n_demo_x * demo_coarse_step
        demo_end_y = y_min + n_demo_y * demo_coarse_step

        line_style_demo = dict(
            color=grid_demo_color,
            lw=grid_demo_lw,
            alpha=grid_demo_alpha,
            linestyle=grid_demo_ls
        )

        for i in range(n_demo_x + 1):
            x = x_min + i * demo_coarse_step
            ax.plot([x, x], [y_min, demo_end_y], **line_style_demo)
        for j in range(n_demo_y + 1):
            y = y_min + j * demo_coarse_step
            ax.plot([x_min, demo_end_x], [y, y], **line_style_demo)

    ax.set_aspect('equal')
    ax.set_title(title, fontsize=title_fs, pad=10, fontweight='normal')
    ax.set_xlim(x_min - step, grid_end_x + step)
    ax.set_ylim(y_min - step, grid_end_y + step)
    ax.legend(title="", bbox_to_anchor=(1., 0.5), loc='center left', handletextpad=0.1,
              labelspacing=leg_labelspacing, fontsize=leg_fs, markerscale=6, frameon=False)
    
    ax.margins(0.01, 0.01)
    plt.axis('off')
    plt.show()


## calculate grid mcn proportions
def get_grid_mcn_proportions(score_df, mcn_cols, grid_centers, step, bin_factor=1, threshold=0,
                            coord_x_key='x', coord_y_key='y'):

    scores = score_df[mcn_cols].values
    s_min = scores.min(axis=0)
    s_max = scores.max(axis=0)
    scores_norm = (scores - s_min) / (s_max - s_min + 1e-9)
    max_norm_scores = np.max(scores_norm, axis=1)
    max_idx = np.argmax(scores_norm, axis=1)
    
    labels = max_idx.copy()
    labels[max_norm_scores <= threshold] = len(mcn_cols)

    coarse_step = step * bin_factor
    x = score_df[coord_x_key].values
    y = score_df[coord_y_key].values

    ## full
    x_min, y_min = grid_centers.min(axis=0) - step/2
    grid_x = ((x - x_min) // coarse_step).astype(int)
    grid_y = ((y - y_min) // coarse_step).astype(int)
    
    score_df['grid_x_idx'] = grid_x
    score_df['grid_y_idx'] = grid_y
    score_df['mcn_hard_label'] = labels

    counts = score_df.groupby(['grid_x_idx', 'grid_y_idx', 'mcn_hard_label']).size().unstack(fill_value=0)
    proportions = counts.div(counts.sum(axis=1), axis=0)

    ## exclude hole
    hole_center = np.array([14300, 167800])
    hole_radius = 2600

    grid_indices = proportions.index.to_frame()
    grid_centers_x = x_min + grid_indices['grid_x_idx'] * coarse_step + coarse_step / 2
    grid_centers_y = y_min + grid_indices['grid_y_idx'] * coarse_step + coarse_step / 2
    distances = np.sqrt((grid_centers_x - hole_center[0])**2 + (grid_centers_y - hole_center[1])**2)
    
    mask_keep = distances > hole_radius
    proportions_exclude_hole = proportions.loc[mask_keep.values]
    
    return proportions, proportions_exclude_hole


def get_grid_domain_proportions(cluster_df, domain_col, grid_centers, step, bin_factor=1,
                                coord_x_key='x', coord_y_key='y'):
    
    coarse_step = step * bin_factor
    x = cluster_df[coord_x_key].values
    y = cluster_df[coord_y_key].values

    ## all cells
    x_min, y_min = grid_centers.min(axis=0) - step/2
    grid_x = ((x - x_min) // coarse_step).astype(int)
    grid_y = ((y - y_min) // coarse_step).astype(int)
    cluster_df['grid_x_idx'] = grid_x
    cluster_df['grid_y_idx'] = grid_y

    counts = cluster_df.groupby(['grid_x_idx', 'grid_y_idx', domain_col]).size().unstack(fill_value=0)
    proportions = counts.div(counts.sum(axis=1), axis=0)

    ## exclude hole
    hole_center = np.array([14300, 167800])
    hole_radius = 2600

    grid_indices = proportions.index.to_frame()
    grid_centers_x = x_min + grid_indices['grid_x_idx'] * coarse_step + coarse_step / 2
    grid_centers_y = y_min + grid_indices['grid_y_idx'] * coarse_step + coarse_step / 2
    distances = np.sqrt((grid_centers_x - hole_center[0])**2 + (grid_centers_y - hole_center[1])**2)
    
    mask_keep = distances > hole_radius
    proportions_exclude_hole = proportions.loc[mask_keep.values]
    
    return proportions, proportions_exclude_hole
    
    
def calculate_compositional_similarity(prop_df1, prop_df3, method='pearson', pseudo_count=1e-6):

    common_indices = prop_df1.index.intersection(prop_df3.index)
    p1_mat = prop_df1.loc[common_indices].values
    p3_mat = prop_df3.loc[common_indices].values
    
    results = []
    
    for i in range(len(p1_mat)):
        u = p1_mat[i]
        v = p3_mat[i]
        
        if np.sum(u) == 0 or np.sum(v) == 0:
            continue
            
        try:
            if method == 'pearson':
                val = pearsonr(u, v)[0]
            elif method == 'spearman':
                val = spearmanr(u, v)[0]
            elif method == 'cosine':
                val = 1 - cosine(u, v)
            elif method == 'jsd':
                val = 1 - jensenshannon(u + pseudo_count, v + pseudo_count)
            elif method == 'braycurtis':
                val = 1 - braycurtis(u, v)
            elif method == 'aitchison':
                u_log = np.log(u + pseudo_count)
                v_log = np.log(v + pseudo_count)
                u_clr = u_log - np.mean(u_log)
                v_clr = v_log - np.mean(v_log)
                dist = np.linalg.norm(u_clr - v_clr)
                val = 1 / (1 + dist) # Transform distance to a similarity score
            else:
                raise ValueError(f"Unknown method: {method}")
            
            if not np.isnan(val):
                results.append(val)
        except:
            continue
            
    return results


def align_clusters_robust(prop_df1, prop_df3, sim_threshold=0.3, size_threshold=0.01):
    
    def get_valid_clusters(df, threshold):
        total_signal = df.sum(axis=0)
        relative_size = total_signal / total_signal.sum()
        return relative_size[relative_size >= threshold].index.tolist()

    common_indices = prop_df1.index.intersection(prop_df3.index)
    prop_df1 = prop_df1.loc[common_indices]
    prop_df3 = prop_df3.loc[common_indices]

    valid_c1 = get_valid_clusters(prop_df1, size_threshold)
    valid_c3 = get_valid_clusters(prop_df3, size_threshold)
    # print('valid_c1:', valid_c1)
    # print('valid_c3:', valid_c3)
    
    sub_df1 = prop_df1[valid_c1]
    sub_df3 = prop_df3[valid_c3]
    
    sim_matrix = cosine_similarity(sub_df1.T, sub_df3.T)
    
    row_ind, col_ind = linear_sum_assignment(sim_matrix, maximize=True)
    
    matched_pairs = []   

    for r, c in zip(row_ind, col_ind):
        score = sim_matrix[r, c]
        c1_label = valid_c1[r]
        c3_label = valid_c3[c]
        
        if score >= sim_threshold:
            matched_pairs.append((c1_label, c3_label, score))

    c1_total_signal = prop_df1.sum(axis=0)
    matched_pairs = sorted(
        matched_pairs,
        key=lambda x: c1_total_signal[x[0]],
        reverse=True
    )

    matched_c1 = [x[0] for x in matched_pairs]
    matched_c3 = [x[1] for x in matched_pairs]

    c1_to_domain = {c1_label: f"Domain_{i+1}" for i, c1_label in enumerate(matched_c1)}
    final_mapping_3to1 = {c3_label: c1_to_domain[c1_label] for c1_label, c3_label, _ in matched_pairs}
    
    aligned_df1 = pd.DataFrame(index=prop_df1.index)
    bg_cols_1 = [c for c in prop_df1.columns if c not in matched_c1]
    # print('bg_cols_1:', len(bg_cols_1))
    
    for c1 in matched_c1:
        aligned_df1[c1_to_domain[c1]] = prop_df1[c1]
    if len(bg_cols_1) > 0:
        aligned_df1['Background'] = prop_df1[bg_cols_1].sum(axis=1)

    aligned_df3 = pd.DataFrame(index=prop_df3.index)
    bg_cols_3 = [c for c in prop_df3.columns if c not in matched_c3]
    # print('bg_cols_3:', len(bg_cols_3))
    
    for c3, new_label in final_mapping_3to1.items():
        aligned_df3[new_label] = prop_df3[c3]
    if len(bg_cols_3) > 0:
        aligned_df3['Background'] = prop_df3[bg_cols_3].sum(axis=1)

    all_columns = [f"Domain_{i+1}" for i in range(len(matched_c1))]
    if 'Background' in aligned_df1.columns:
        all_columns = all_columns + ['Background']
    
    # Add missing columns with zeros if any (due to asymmetric mapping)
    for col in all_columns:
        if col not in aligned_df1.columns:
            aligned_df1[col] = 0.0
        if col not in aligned_df3.columns:
            aligned_df3[col] = 0.0
        
    return aligned_df1[all_columns], aligned_df3[all_columns], matched_c1, matched_c3, final_mapping_3to1, sim_matrix


def plot_grid_scatter_pie(
    prop_df, x_min, y_min, step, bin_factor, niche_cmap_dict, 
    colors=None, title="", leg_labels=None, 
    title_fs=20, leg_fs=16,
    window=None, window_lw=1, window_ls='--', window_lc='k'):

    fig, ax = plt.subplots(1, 1, figsize=(10, 7), dpi=300)

    coarse_step = step * bin_factor
    radius = coarse_step * 0.5

    if colors is None:
        colors = [niche_cmap_dict['Niche 1'], niche_cmap_dict['Niche 2'], niche_cmap_dict['Niche 3'], niche_cmap_dict['Niche 4'], '#e0e0e0']
        colors = [niche_cmap_dict['Niche 1'], niche_cmap_dict['Niche 2'], niche_cmap_dict['Niche 4'], niche_cmap_dict['Niche 3'], '#e0e0e0']
        
    for (gx, gy), row in prop_df.iterrows():
        center_x = x_min + gx * coarse_step + coarse_step / 2
        center_y = y_min + gy * coarse_step + coarse_step / 2
        
        proportions = row.values
        if np.sum(proportions) > 0:
            start_angle = 90
            for i, p in enumerate(proportions):
                if p > 0:
                    wedge_angle = 360 * p
                    wedge = Wedge(
                        center=(center_x, center_y), r=radius,
                        theta1=start_angle, theta2=start_angle + wedge_angle,
                        facecolor=colors[i], edgecolor='white', lw=0.1, alpha=0.9
                    )
                    ax.add_patch(wedge)
                    start_angle += wedge_angle

    ax.set_aspect('equal')
    ax.set_title(title, fontsize=title_fs, fontweight='normal', pad=10)
    n_grids_x = prop_df.index.get_level_values(0).max() + 1
    n_grids_y = prop_df.index.get_level_values(1).max() + 1
    ax.set_xlim(x_min, x_min + n_grids_x * coarse_step)
    ax.set_ylim(y_min, y_min + n_grids_y * coarse_step)
    plt.axis('off')

    if leg_labels is None:
        leg_labels = [f"MCN {i+1}" for i in range(len(proportions)-1)] + ["Background"]
    legend_elements = [Line2D([0], [0], marker='o', color='w', label=l,
                              markerfacecolor=c, markersize=10) for l, c in zip(leg_labels, colors)]
    ax.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5,-0.02), 
              ncol=len(leg_labels), frameon=False, fontsize=leg_fs)

    if window is not None:
        x0, y0, xl, yl = window
        rect = Rectangle((x0, y0), xl, yl, linewidth=window_lw, linestyle=window_ls, edgecolor=window_lc, facecolor='none')
        ax.add_patch(rect)

    