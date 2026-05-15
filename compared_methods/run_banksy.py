import warnings
warnings.filterwarnings("ignore") 
import os, time, gc, argparse
import pandas as pd
import numpy as np
import anndata
import scanpy as sc
import random
import scipy.sparse as sparse
from scipy.sparse import csr_matrix, issparse
from typing import List, Union, Tuple

from banksy.initialize_banksy import initialize_banksy
from banksy.run_banksy import run_banksy_multiparam
from banksy_utils.color_lists import spagcn_color

start = time.perf_counter_ns()
random_seed = 1234
cluster_algorithm = 'leiden'
np.random.seed(random_seed)
random.seed(random_seed)

from banksy.embed_banksy import generate_banksy_matrix
from banksy.labels import Label
from banksy.main import concatenate_all
from banksy.cluster_methods import run_Leiden_partition, run_mclust_partition
from banksy_utils.umap_pca import pca_umap


def run_banksy_multiparam_(adata: anndata.AnnData,
                          banksy_dict: dict,
                          lambda_list: List[int],
                          resolutions: List[int],
                          color_list: Union[List, str],
                          max_m: int,
                          filepath: str,
                          key: Tuple[str],
                          match_labels: bool = False,
                          pca_dims: List[int] = [20, ],
                          annotation_key: str = "cluster_name",
                          max_labels: int = None,
                          variance_balance: bool = False,
                          cluster_algorithm: str = 'leiden',
                          partition_seed: int = 1234,
                          add_nonspatial: bool = True,
                           cluster_cells: str = 'clusterall',
                           target_ct: str = None,
                          **kwargs):
    options = {
        'save_all_h5ad': True,
        'save_name': 'slideseq_mousecerebellum_',
        'no_annotation': True,
        's': 50,
        'a': 1.0
    }

    options.update(kwargs)
    if not os.path.exists(filepath):
        os.makedirs(filepath)

    banksy_dict, banksy_matrix = generate_banksy_matrix(adata,
                                                        banksy_dict,
                                                        lambda_list,
                                                        max_m,
                                                        variance_balance=variance_balance)

    # Add nonspatial banksy matrix
    if add_nonspatial:
        banksy_dict["nonspatial"] = {0.0: {"adata": concatenate_all([adata.X], 0, adata=adata), }}

    pca_umap(banksy_dict,
             pca_dims=pca_dims,
             plt_remaining_var=False)

    if annotation_key:
        annotations = Label(adata.obs[annotation_key].cat.codes.tolist())
    else:
        annotations = None

    # Clustering algorithm
    if cluster_cells == 'clustertarget':
        nbr_weight_decay = list(banksy_dict.keys())[0]
        adata_temp = banksy_dict[nbr_weight_decay][lambda_list[0]]['adata']
        adata_temp = adata_temp[adata_temp.obs['cell_type']==target_ct]
        banksy_dict[nbr_weight_decay][lambda_list[0]]['adata'] = adata_temp
    if cluster_algorithm == 'leiden':
        print(f'Conducting clustering with Leiden Parition')
        results_df, max_num_labels = run_Leiden_partition(
            banksy_dict=banksy_dict,
            resolutions=resolutions,
            num_nn=50,
            num_iterations=-1,
            partition_seed=partition_seed,
            match_labels=match_labels,
            annotations=annotations,
            max_labels=max_labels
        )

    elif cluster_algorithm == 'mclust':
        print(f'Conducting clustering with mcluster algorithm')
        try:
            import rpy2
        except ModuleNotFoundError:
            print(f'Package rpy2 not installed, try pip install')

        match_labels = False
        results_df, max_num_labels = run_mclust_partition(
            banksy_dict=banksy_dict,
            partition_seed=partition_seed,
            match_labels=match_labels,
            annotations=annotations,
            num_labels=max_labels,
        )

    for params_name in results_df.index:
        gc.collect()

        adata_temp = results_df.loc[params_name, 'adata']
        raw_labels = results_df.loc[params_name, 'labels']
        num_clusters = results_df.loc[params_name, 'num_labels']
        lambda_p = results_df.loc[params_name, "lambda_param"]

        if annotation_key:
            ari_temp = results_df.loc[params_name, "ari"]
            ari_label = f'\nari = {round(ari_temp, 2)}'
        else:
            ari_label = ""

        raw_clusters = []

        if isinstance(raw_labels, Label):
            raw_labels = raw_labels.dense
        for i in raw_labels:
            raw_clusters.append(color_list[i])

        print(f'Anndata {adata_temp.obsm}')

    return results_df, raw_labels

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, help='directory of simulated data', required=True)
    parser.add_argument('--data-name', type=str, help='name of simulated data', required=True)
    parser.add_argument('--out-dir', type=str, help='directory for saving results', required=True)
    parser.add_argument('--leiden-res', type=float, default=0.2, help='leiden clustering resolution', required=True)
    parser.add_argument('--cluster-cells', type=str, choices=['clusterall', 'clustertarget'], help='cluster all cells or target cells', required=True)
    args = parser.parse_args()

    adata = sc.read(f'{args.data_dir}/{args.data_name}.h5ad')

    target_ct = 'A'
    coord_keys = ('x', 'y', 'spatial')  # Adjust based on your data
    resolutions = [args.leiden_res] # clustering resolution for Leiden clustering
    pca_dims = [20] # number of dimensions to keep after PCA
    lambda_list = [.8] # lambda
    k_geom = 10 #spatial neighbours
    max_m = 1 # use AGF
    nbr_weight_decay = "scaled_gaussian" # can also be "reciprocal", "uniform" or "ranked"
    num_clusters = None
    os.makedirs(args.out_dir, exist_ok=True)

    # Initialize BANKSY
    banksy_dict = initialize_banksy(
        adata,
        coord_keys,
        num_neighbours=k_geom,
        nbr_weight_decay=nbr_weight_decay,
        max_m=max_m,
        plt_edge_hist=False,
        plt_nbr_weights=False,
        plt_agf_angles=False,
        plt_theta=False,
    )

    results_df, raw_labels = run_banksy_multiparam_(
        adata,
        banksy_dict,
        lambda_list,
        resolutions,
        color_list = spagcn_color,
        max_m = max_m,
        filepath = args.out_dir,
        key = coord_keys,
        pca_dims = pca_dims,
        annotation_key = None,
        max_labels = num_clusters,
        cluster_algorithm = 'leiden',
        match_labels = False,
        add_nonspatial = False,
        variance_balance = False,
        cluster_cells = args.cluster_cells,
        target_ct = target_ct
    )

    if args.cluster_cells == 'clusterall':
        cluster_df = adata.obs[['cell_id', 'x', 'y', 'cell_type', 'region']].copy()
    elif args.cluster_cells == 'clustertarget':
        cluster_df = adata.obs[adata.obs['cell_type']==target_ct][['cell_id', 'x', 'y', 'cell_type', 'region']].copy()
    cluster_df['domain'] = raw_labels
    if len(np.unique(cluster_df['domain'])) > 6:
        domain_map = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4}
        for d_ in cluster_df['domain'].unique():
            if int(d_) not in [0, 1, 2, 3, 4]:
                domain_map[d_] = 5
        cluster_df['domain'] = cluster_df['domain'].map(domain_map)
    cluster_df.to_csv(f'{args.out_dir}/{args.data_name}_{args.cluster_cells}_res{args.leiden_res}.csv', sep='\t', index=False)
