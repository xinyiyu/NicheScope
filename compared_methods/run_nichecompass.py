import os
import random
import argparse
import warnings
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scanpy as sc
import seaborn as sns
import squidpy as sq
from matplotlib import gridspec
from sklearn.preprocessing import MinMaxScaler

from nichecompass.models import NicheCompass
from nichecompass.utils import add_gps_from_gp_dict_to_adata, create_new_color_dict
import torch
from scipy.sparse import csr_matrix

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, help='directory of simulated data', required=True)
    parser.add_argument('--data-name', type=str, help='name of simulated data', required=True)
    parser.add_argument('--out-dir', type=str, help='directory for saving results', required=True)
    parser.add_argument('--leiden-res', type=float, default=0.1, help='leiden clustering resolution', required=True)
    parser.add_argument('--cluster-cells', type=str, choices=['clusterall', 'clustertarget'], help='cluster all cells or target cells', required=True)
    parser.add_argument('--load-model', action='store_true', help='load trained model')
    args = parser.parse_args()

    datapath = f'{args.data_dir}/{args.data_name}.h5ad'

    ### Dataset ###
    spatial_key = "spatial"
    n_neighbors = 20

    ### Model ###
    # AnnData Keys
    counts_key = "counts"
    adj_key = "spatial_connectivities"
    gp_names_key = "nichecompass_gp_names"
    active_gp_names_key = "nichecompass_active_gp_names"
    gp_targets_mask_key = "nichecompass_gp_targets"
    gp_targets_categories_mask_key = "nichecompass_gp_targets_categories"
    gp_sources_mask_key = "nichecompass_gp_sources"
    gp_sources_categories_mask_key = "nichecompass_gp_sources_categories"
    latent_key = "nichecompass_latent"

    # Architecture
    conv_layer_encoder = "gcnconv" # change to "gatv2conv" if enough compute and memory
    active_gp_thresh_ratio = 0.01

    # Trainer
    n_epochs = 100
    n_epochs_all_gps = 25
    lr = 0.001
    lambda_edge_recon = 500000.
    lambda_gene_expr_recon = 300.
    lambda_l1_masked = 0. # prior GP  regularization
    lambda_l1_addon = 30. # de novo GP regularization
    edge_batch_size = 1024 # increase if more memory available or decrease to save memory
    n_sampled_neighbors = 20
    use_cuda_if_available = True

    ### Analysis ###
    cell_type_key = "cell_type"
    latent_leiden_resolution = args.leiden_res
    latent_cluster_key = 'domain'
    sample_key = "region"
    spot_size = 10
    differential_gp_test_results_key = "nichecompass_differential_gp_test_results"

    # Define paths
    model_folder_path = f'{args.out_dir}/model'
    figure_folder_path = f'{args.out_dir}/figures'
    os.makedirs(model_folder_path, exist_ok=True)
    os.makedirs(figure_folder_path, exist_ok=True)

    # Check if trained model exists
    if args.load_model and os.path.exists(f'{model_folder_path}/{args.data_name}.h5ad'):
        # Load trained model
        model = NicheCompass.load(dir_path=model_folder_path,
                                adata=None,
                                adata_file_name=f"{args.data_name}.h5ad",
                                gp_names_key=gp_names_key)
    else:
        adata = sc.read_h5ad(datapath)
        adata.X = csr_matrix(adata.X)
        adata.layers['counts'] = csr_matrix(adata.layers['counts'])
        # Compute spatial neighborhood
        sq.gr.spatial_neighbors(adata,
                                coord_type="generic",
                                spatial_key=spatial_key,
                                n_neighs=n_neighbors)

        # Make adjacency matrix symmetric
        adata.obsp[adj_key] = (adata.obsp[adj_key].maximum(adata.obsp[adj_key].T))

        # Pseudo GP
        genes = adata.var_names.to_list()
        n_genes = len(genes)
        print(f'{n_genes} genes.')

        n_prior_gp = 20              
        n_sources_per_gp = 3
        n_targets_per_gp = 5

        combined_gp_dict = {}

        for i in range(n_prior_gp):
            gp_name = f"SimGP_{i}"

            source_genes = np.random.choice(
                genes, size=n_sources_per_gp, replace=False
            ).tolist()

            target_genes = np.random.choice(
                list(set(genes) - set(source_genes)),
                size=n_targets_per_gp,
                replace=False
            ).tolist()

            combined_gp_dict[gp_name] = {
                "sources": source_genes,
                "targets": target_genes,
                "sources_categories": ["ligand"] * n_sources_per_gp,
                "targets_categories": ["receptor"] * n_targets_per_gp,
            }

        # Add the GP dictionary as binary masks to the adata
        add_gps_from_gp_dict_to_adata(
            gp_dict=combined_gp_dict,
            adata=adata,
            gp_targets_mask_key=gp_targets_mask_key,
            gp_targets_categories_mask_key=gp_targets_categories_mask_key,
            gp_sources_mask_key=gp_sources_mask_key,
            gp_sources_categories_mask_key=gp_sources_categories_mask_key,
            gp_names_key=gp_names_key,
            min_genes_per_gp=2,
            min_source_genes_per_gp=1,
            min_target_genes_per_gp=1,
            max_genes_per_gp=None,
            max_source_genes_per_gp=None,
            max_target_genes_per_gp=None)

        # Initialize model
        model = NicheCompass(adata,
                            counts_key=counts_key,
                            adj_key=adj_key,
                            gp_names_key=gp_names_key,
                            active_gp_names_key=active_gp_names_key,
                            gp_targets_mask_key=gp_targets_mask_key,
                            gp_targets_categories_mask_key=gp_targets_categories_mask_key,
                            gp_sources_mask_key=gp_sources_mask_key,
                            gp_sources_categories_mask_key=gp_sources_categories_mask_key,
                            latent_key=latent_key,
                            conv_layer_encoder=conv_layer_encoder,
                            active_gp_thresh_ratio=active_gp_thresh_ratio)

        # Train model
        model.train(n_epochs=n_epochs,
                    n_epochs_all_gps=n_epochs_all_gps,
                    lr=lr,
                    lambda_edge_recon=lambda_edge_recon,
                    lambda_gene_expr_recon=lambda_gene_expr_recon,
                    lambda_l1_masked=lambda_l1_masked,
                    edge_batch_size=edge_batch_size,
                    n_sampled_neighbors=n_sampled_neighbors,
                    use_cuda_if_available=use_cuda_if_available,
                    verbose=False)    


        # Compute latent neighbor graph
        sc.pp.neighbors(model.adata,
                        use_rep=latent_key,
                        key_added=latent_key)
        # Compute UMAP embedding
        sc.tl.umap(model.adata, neighbors_key=latent_key)

        # Save trained model
        model.save(dir_path=model_folder_path,
                overwrite=True,
                save_adata=True,
                adata_file_name=f"{args.data_name}.h5ad")
        
    # Compute latent Leiden clustering
    if args.cluster_cells == 'clusterall':
        adata_cluster = model.adata.copy()
        sc.tl.leiden(adata=adata_cluster,
                    resolution=latent_leiden_resolution,
                    key_added=latent_cluster_key,
                    neighbors_key=latent_key)
        latent_cluster_colors = create_new_color_dict(
            adata=adata_cluster,
            cat_key=latent_cluster_key)
    
        cluster_df = adata_cluster.obs[['cell_id', 'x', 'y', 'cell_type', 'region', latent_cluster_key]].copy()
        cluster_df.to_csv(f'{args.out_dir}/{args.data_name}_{args.cluster_cells}_res{latent_leiden_resolution}.csv', sep='\t', index=False)
    elif args.cluster_cells == 'clustertarget':
        adata_cluster = model.adata[model.adata.obs['cell_type']=='A'].copy()
        sc.tl.leiden(adata=adata_cluster,
                    resolution=latent_leiden_resolution,
                    key_added=latent_cluster_key,
                    neighbors_key=latent_key)
        latent_cluster_colors = create_new_color_dict(
            adata=adata_cluster,
            cat_key=latent_cluster_key)

        tmp = adata_cluster.obs.copy()
        n_domains = len(np.unique(tmp['domain']))
        if n_domains > 6:
            domain_map = {'0': '0', '1': '1', '2': '2', '3': '3', '4': '4'}
            for d_ in adata_cluster.obs['domain'].unique():
                if int(d_) not in [0, 1, 2, 3, 4]:
                    domain_map[d_] = '5'
            tmp['domain'] = tmp['domain'].map(domain_map)
            n_domains = len(np.unique(tmp['domain']))
            tmp['domain'].value_counts()
            tmp.to_csv(f'{args.out_dir}/{args.data_name}_{args.cluster_cells}_res{latent_leiden_resolution}.csv', sep='\t', index=False)
            