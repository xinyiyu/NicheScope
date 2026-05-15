import os, argparse
import anndata as ad
import squidpy as sq
import cellcharter as cc
import pandas as pd
import scanpy as sc
import scvi
import numpy as np
from lightning.pytorch import seed_everything

seed_everything(12345)
scvi.settings.seed = 12345

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, help='directory of simulated data', required=True)
    parser.add_argument('--data-name', type=str, help='name of simulated data', required=True)
    parser.add_argument('--out-dir', type=str, help='directory for saving results', required=True)
    parser.add_argument('--num-cluster', type=int, default=6, help='number of clusters', required=True)
    parser.add_argument('--cluster-cells', type=str, choices=['clusterall', 'clustertarget'], help='cluster all cells or target cells', required=True)
    args = parser.parse_args()

    adata = sc.read(f'{args.data_dir}/{args.data_name}.h5ad')
    
    ## dimension reduction
    scvi.model.SCVI.setup_anndata(adata, layer="counts")
    model = scvi.model.SCVI(adata)
    model.train(early_stopping=True, enable_progress_bar=True)
    adata.obsm['X_scVI'] = model.get_latent_representation(adata).astype(np.float32)
    model.save(f'{args.out_dir}/{args.data_name}')
    
    ## clustering
    if args.cluster_cells == 'clustertarget':
        adata_cluster = adata[adata.obs['cell_type']=='A'].copy()
    else:
        adata_cluster = adata.copy()
    sq.gr.spatial_neighbors(adata_cluster, 
                            coord_type='generic', 
                            delaunay=True, 
                            percentile=99,
                            key_added='cellcharter')
    cc.gr.aggregate_neighbors(adata_cluster, 
                            n_layers=3, 
                            connectivity_key='cellcharter',
                            use_rep='X_scVI', 
                            out_key='X_cellcharter')

    gmm = cc.tl.Cluster(
        n_clusters=args.num_cluster, 
        random_state=12345,
        trainer_params=dict(accelerator='gpu', devices=1)
    )
    gmm.fit(adata_cluster, use_rep='X_cellcharter')
    adata_cluster.obs['domain'] = gmm.predict(adata_cluster, use_rep='X_cellcharter')
    save_columns = ['cell_id', 'x', 'y', 'cell_type', 'region', 'domain']
    cluster_df = adata_cluster.obs[save_columns].copy()
    cluster_df.to_csv(f'{args.out_dir}/{args.data_name}_{args.cluster_cells}_{args.num_cluster}domain.csv', sep='\t', index=False)
    