from SPACEL.setting import set_environ_seed
from SPACEL import Splane
set_environ_seed(42)

import os, argparse
import scanpy as sc
import numpy as np
import pandas as pd
import torch

def identify_spatial_domain_target(model,target_ids=None,key='domain'):

    model.model_g.load_state_dict(torch.load(model.best_path))
    model.model_g.eval()
    encoded, decoded = model.model_g(model.graph[0], model.graph[1:])
    print(encoded.shape)
    if target_ids is not None:
        encoded = encoded[target_ids,:]
        print(f'Use target cells for clustering: {encoded.shape}')
    clusters = model.Cluster.fit_predict(encoded.cpu().detach().numpy())
    
    return clusters

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, help='directory of simulated data', required=True)
    parser.add_argument('--data-name', type=str, help='name of simulated data', required=True)
    parser.add_argument('--out-dir', type=str, help='directory for saving results', required=True)
    parser.add_argument('--num-cluster', type=int, help='number of clusters', required=True)
    parser.add_argument('--cluster-cells', type=str, choices=['clusterall', 'clustertarget'], help='cluster all cells or target cells', required=True)
    args = parser.parse_args()

    adata = sc.read(f'{args.data_dir}/{args.data_name}.h5ad')

    # train model
    Splane.utils.add_cell_type_composition(adata, celltype_anno=adata.obs['cell_type'])
    adata_list = [adata]
    splane_model = Splane.init_model(adata_list, n_clusters=args.num_cluster, use_gpu=True, n_neighbors=20, gnn_dropout=0.5)
    splane_model.train(d_l=0.)

    # cluster
    if args.cluster_cells == 'clusterall':
        splane_model.identify_spatial_domain(key='domain')
        adata.obs['domain'].value_counts()
        # adata.write(f'{args.out_dir}/{args.data_name}_{args.num_cluster}domain.h5ad')
        cluster_df = adata.obs[['cell_id', 'x', 'y', 'cell_type', 'region', 'domain']].copy()
    elif args.cluster_cells == 'clustertarget':
        target_ids = np.where(adata.obs.cell_type=='A')[0]
        clusters_ = identify_spatial_domain_target(splane_model, target_ids=target_ids)
        cluster_df = adata.obs.loc[adata.obs['cell_type']=='A'][['cell_id', 'x', 'y', 'cell_type', 'region']].copy()
        cluster_df['domain'] = clusters_
    cluster_df.to_csv(f'{args.out_dir}/{args.data_name}_{args.cluster_cells}_{args.num_cluster}domain.csv', sep='\t', index=False)

