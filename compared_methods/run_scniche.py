import os, sys, time, argparse
import numpy as np
import scniche as sn
import scanpy as sc
from sklearn.metrics import adjusted_rand_score
import warnings
warnings.filterwarnings('ignore')

# set seed
sn.pp.set_seed()

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, help='directory of simulated data', required=True)
    parser.add_argument('--data-name', type=str, help='name of simulated data', required=True)
    parser.add_argument('--out-dir', type=str, help='directory for saving results', required=True)
    parser.add_argument('--num-cluster', type=int, default=7, help='number of clusters', required=True)
    parser.add_argument('--cluster-cells', type=str, choices=['clusterall', 'clustertarget'], help='cluster all cells or target cells', required=True)
    args = parser.parse_args()

    k_cutoff = 30
    batch_num = 30
    lr = 0.01
    epochs = 100

    if args.cluster_cells == 'clusterall':
        adata = sc.read(f'{args.data_dir}/{args.data_name}.h5ad')
    
        # prepare
        sc.pp.scale(adata)
        adata = sn.pp.cal_spatial_neighbors(adata=adata, celltype_key='cell_type', mode='KNN', k_cutoff=k_cutoff, verbose=False)
        adata = sn.pp.cal_spatial_exp(adata=adata, mode='KNN', k_cutoff=k_cutoff, is_pca=True, verbose=False)
        adata = sn.pp.prepare_data_batch(adata=adata, verbose=False, batch_num=batch_num)
    
        # adata_save = adata.copy()
        # del adata_save.uns['dataloader']
        # sc.write(f'{args.out_dir}/{args.data_name}_scniche_prepro.h5ad', adata_save)
    
        # train model
        model = sn.tr.Runner_batch(adata=adata, device='cuda:0', verbose=False)
        adata = model.fit(lr=lr, epochs=epochs)
    
        # adata_save = adata.copy()
        # del adata_save.uns['dataloader']
        # sc.write(f'{args.out_dir}/{args.data_name}_train.h5ad', adata_save)
    
        # clustering
        adata_cluster = adata.copy()
        adata_cluster = sn.tr.clustering(adata=adata_cluster, target_k=args.num_cluster, add_key='domain')
        obs = adata_cluster.obs[['cell_id', 'x', 'y', 'cell_type', 'region', 'domain']].copy()
        obs['domain'] = obs['domain'].apply(lambda x: int(x.replace('Niche', '')))
        obs.to_csv(f'{args.out_dir}/{args.data_name}_{args.cluster_cells}_{args.num_cluster}domain.csv', sep='\t', index=False)
    elif args.cluster_cells == 'clustertarget':
        # # load trained
        # adata = sc.read(f'{args.out_dir}/{args.data_name}_train.h5ad')
        adata = sc.read(f'{args.data_dir}/{args.data_name}.h5ad')
    
        # prepare
        sc.pp.scale(adata)
        adata = sn.pp.cal_spatial_neighbors(adata=adata, celltype_key='cell_type', mode='KNN', k_cutoff=k_cutoff, verbose=False)
        adata = sn.pp.cal_spatial_exp(adata=adata, mode='KNN', k_cutoff=k_cutoff, is_pca=True, verbose=False)
        adata = sn.pp.prepare_data_batch(adata=adata, verbose=False, batch_num=batch_num)
    
        # train model
        model = sn.tr.Runner_batch(adata=adata, device='cuda:0', verbose=False)
        adata = model.fit(lr=lr, epochs=epochs)
        
        adata_cluster = adata[adata.obs['cell_type']=='A'].copy()
        adata_cluster = sn.tr.clustering(adata=adata_cluster, target_k=args.num_cluster, add_key='domain')
        obs = adata_cluster.obs[['cell_id', 'x', 'y', 'cell_type', 'region', 'domain']].copy()
        obs['domain'] = obs['domain'].apply(lambda x: int(x.replace('Niche', '')))
        obs.to_csv(f'{args.out_dir}/{args.data_name}_{args.cluster_cells}_{args.num_cluster}domain.csv', sep='\t', index=False)
