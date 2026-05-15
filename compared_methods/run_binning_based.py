import argparse
import numpy as np
import pandas as pd
import squidpy as sq
import scanpy as sc
from tqdm import tqdm


def format_data_neighs_colapse(adata,sname,condit,neighs=10):
    """ Redefine the expression of cells in adata by collapsing the expression of its neighbors into each cell (a.k.a pseudobining)
     
    Parameters:
    adata (AnnData): AnnData object with the cells of the experiment
    sname(str): column in adata.obs where sample is stored
    condit(str): column in adata.obs where the sample each cell belongs to is stored
    neighs(int): number of neighbors to consider when collapsing the expression of neighboring cells
    
    Returns:
    adata1 (AnnData): AnnData object with expression of cells collapsed from neighboring cells
    """
    adata.obsm["spatial"]=np.array([adata.obs['x'],adata.obs['y']]).transpose().astype('float64')
    adata_copy_int=adata
    sq.gr.spatial_neighbors(adata_copy_int,n_neighs=neighs)
    result=np.zeros([adata.shape[0],adata.shape[1]])
    n=0
    tr=adata_copy_int.obsp['spatial_distances'].transpose()
    tr2=tr>0
    exp=adata_copy_int.to_df()
    
    #tdd=tr2.todense()
    for i in tqdm(range(0,adata_copy_int.to_df().shape[0])):
        result[i,:]=np.sum(exp[tr2[i,:].todense().transpose()],axis=0)
    adata1=sc.AnnData(result,obs=adata.obs,var=adata.var)
    return adata1

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, help='directory of simulated data', required=True)
    parser.add_argument('--data-name', type=str, help='name of simulated data', required=True)
    parser.add_argument('--out-dir', type=str, help='directory for saving results', required=True)
    parser.add_argument('--res', type=float, default=0.2, help='leiden clustering resolution', required=True)
    parser.add_argument('--cluster-cells', type=str, choices=['clusterall', 'clustertarget'], help='cluster all cells or target cells', required=True)
    args = parser.parse_args()

    adata = sc.read(f'{args.data_dir}/{args.data_name}.h5ad')

    if args.cluster_cells == 'clusterall':
        # workflow for clusterall (select resolution)
        # define hyperparameters
        neighs=[20]
        ss='sample'
        annot='Class'
        target_ct = 'A'
        for ng in neighs:
            an_concat=[]
            asub=adata
            adataneigh=format_data_neighs_colapse(asub,annot,ss,neighs=ng)
            an_concat.append(adataneigh)
            adataneigh=sc.concat(an_concat)
            adataneigh.obs['total_counts'] = adataneigh.X.sum(1)
            adataneigh.obsm["spatial"]=np.array([adataneigh.obs['x'],adataneigh.obs['y']]).transpose().astype('float64')
            adataneigh.X=np.nan_to_num(adataneigh.X)
            adataneigh=adataneigh[adataneigh.obs['total_counts']>3]
            adataneigh.raw=adataneigh
    
            sc.pp.neighbors(adataneigh, n_neighbors=20,n_pcs=0)
            sc.tl.umap(adataneigh,min_dist=0.1)

            sc.tl.leiden(adataneigh, resolution=args.res, key_added='domain')
            n_domains = len(np.unique(adataneigh.obs['domain']))
            tmp = adataneigh.obs.copy()
            if n_domains > 6:
                domain_map = {'0': '0', '1': '1', '2': '2', '3': '3', '4': '4'}
                for d_ in adataneigh.obs['domain'].unique():
                    if int(d_) not in [0, 1, 2, 3, 4]:
                        domain_map[d_] = '5'
                tmp['domain'] = tmp['domain'].map(domain_map)
                n_domains = len(np.unique(tmp['domain']))
                tmp['domain'].value_counts()
            fname = f'{args.out_dir}/{args.data_name}_{args.cluster_cells}_neigh{ng}_res{args.res:.4g}_{n_domains}domain.csv'
            tmp[['cell_id', 'x', 'y', 'cell_type', 'region', 'domain']].to_csv(fname, sep='\t', index=False)

    elif args.cluster_cells == 'clustertarget':
        # workflow for clustertarget (fix resolution)
        # define hyperparameters
        neighs=[20]
        ss='sample'
        annot='Class'
        target_ct = 'A'
        for ng in neighs:
            an_concat=[]
            asub=adata
            adataneigh=format_data_neighs_colapse(asub,annot,ss,neighs=ng)
            an_concat.append(adataneigh)
            adataneigh=sc.concat(an_concat)
            adataneigh.obs['total_counts'] = adataneigh.X.sum(1)
            adataneigh.obsm["spatial"]=np.array([adataneigh.obs['x'],adataneigh.obs['y']]).transpose().astype('float64')
            adataneigh.X=np.nan_to_num(adataneigh.X)
            adataneigh=adataneigh[adataneigh.obs['total_counts']>3]
            adataneigh.raw=adataneigh
    
            sc.pp.neighbors(adataneigh, n_neighbors=20,n_pcs=0)
            sc.tl.umap(adataneigh,min_dist=0.1)

        adataneigh_target = adataneigh[adataneigh.obs['cell_type']==target_ct].copy()
        sc.tl.leiden(adataneigh_target, resolution=0.15, key_added='domain')
        n_domains = len(np.unique(adataneigh_target.obs['domain']))

        tmp = adataneigh_target.obs.copy()
        if n_domains > 6:
            domain_map = {'0': '0', '1': '1', '2': '2', '3': '3', '4': '4'}
            for d_ in adataneigh_target.obs['domain'].unique():
                if int(d_) not in [0, 1, 2, 3, 4]:
                    domain_map[d_] = '5'
            tmp['domain'] = tmp['domain'].map(domain_map)
            n_domains = len(np.unique(tmp['domain']))
            tmp['domain'].value_counts()
        fname = f'{args.out_dir}/{args.data_name}_{args.cluster_cells}_neigh{ng}_res{args.res:.4g}_{n_domains}domain.csv'
        tmp[['cell_id', 'x', 'y', 'cell_type', 'region', 'domain']].to_csv(fname, sep='\t', index=False)

