# in CytoCommunity_gpu conda env
import os, sys, time, tqdm, importlib, pickle, json, argparse
import numpy as np
import pandas as pd
import pathlib
import seaborn as sns
import scanpy as sc
import torch
import STAGATE_pyG
import torch.nn.functional as F
import warnings
warnings.filterwarnings('ignore')

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, help='directory of simulated data', required=True)
    parser.add_argument('--data-name', type=str, help='name of simulated data', required=True)
    parser.add_argument('--out-dir', type=str, help='directory for saving results', required=True)
    parser.add_argument('--louvain-res', type=float, default=0.3, help='louvain clustering resolution', required=True)
    parser.add_argument('--cluster-cells', type=str, choices=['clusterall', 'clustertarget'], help='cluster all cells or target cells', required=True)
    args = parser.parse_args()

    if args.cluster_cells == 'clusterall':
        adata = sc.read(f'{args.data_dir}/{args.data_name}.h5ad')
    
        ## construct spatial network
        STAGATE_pyG.Cal_Spatial_Net(adata, rad_cutoff=20)
        STAGATE_pyG.Stats_Spatial_Net(adata)
    
        ## train model
        adata = STAGATE_pyG.train_STAGATE(adata, device=torch.device('cuda:0'))
        # sc.write(f'{args.out_dir}/{args.data_name}_train.h5ad', adata)
    
        ## clustering
        adata_cluster = adata.copy()
        sc.pp.neighbors(adata_cluster, use_rep='STAGATE')
        sc.tl.umap(adata_cluster)
        sc.tl.louvain(adata_cluster, resolution=args.louvain_res, key_added='domain')
    
        cluster_df = adata_cluster.obs[['cell_id', 'x', 'y', 'cell_type', 'region', 'domain']].copy()
        cluster_df.to_csv(f'{args.out_dir}/{args.data_name}_{args.cluster_cells}_res{args.louvain_res}.csv', sep='\t', index=False)
    elif args.cluster_cells == 'clustertarget':
        # ## load trained
        # adata = sc.read(f'{args.out_dir}/{args.data_name}_train.h5ad')
        adata = sc.read(f'{args.data_dir}/{args.data_name}.h5ad')
    
        ## construct spatial network
        STAGATE_pyG.Cal_Spatial_Net(adata, rad_cutoff=20)
        STAGATE_pyG.Stats_Spatial_Net(adata)
    
        ## train model
        adata = STAGATE_pyG.train_STAGATE(adata, device=torch.device('cuda:0'))
        
        ## clustering
        adata_cluster = adata[adata.obs['cell_type']=='A'].copy()
        sc.pp.neighbors(adata_cluster, use_rep='STAGATE')
        sc.tl.umap(adata_cluster)
        sc.tl.louvain(adata_cluster, resolution=args.louvain_res, key_added='domain')
    
        cluster_df = adata_cluster.obs[['cell_id', 'x', 'y', 'cell_type', 'region', 'domain']].copy()
        cluster_df.to_csv(f'{args.out_dir}/{args.data_name}_{args.cluster_cells}_res{args.louvain_res}.csv', sep='\t', index=False)