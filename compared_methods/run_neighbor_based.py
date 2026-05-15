import argparse
import numpy as np
import pandas as pd
import squidpy as sq
import scanpy as sc
from tqdm import tqdm


def format_data_neighs(adata,sname,condit,neighs=10):
    """ Redefine the expression of cells in adata by counting the neighnoring cell types of each cell
    
    Parameters:
    adata (AnnData): AnnData object with the cells of the experiment
    sname(str): column in adata.obs where the cluster assigned to each cells are stored
    neighs(int): number of neighbors to consider when computing neighboring cells
    
    Returns:
    adata1 (AnnData): AnnData object with neighboring cell types included in a cell-by-celltype matrix
    """
    try:
        adata.obsm['spatial']
    except:
        adata.obsm["spatial"]=np.array([adata.obs['x'],adata.obs['y']]).transpose().astype('float64')
    adata_copy_int=adata
    sq.gr.spatial_neighbors(adata_copy_int,n_neighs=neighs)
    result=np.zeros([adata.shape[0],len(adata_copy_int.obs[sname].unique())])
    n=0
    tr=adata_copy_int.obsp['spatial_distances'].transpose()
    tr2=tr>0

    for g in tqdm(adata_copy_int.obs[sname].unique()):
        epv=adata_copy_int.obs[sname]==g*1
        opv=list(epv*1)
        result[:,n]=tr2.dot(opv)
        n=n+1
    expmat=pd.DataFrame(result,columns=adata_copy_int.obs[sname].unique())
    adata1=sc.AnnData(expmat,obs=adata.obs)
    #adata1.obs['sample']=condit
    adata1.obs['condition']=condit
    return adata1

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, help='directory of simulated data', required=True)
    parser.add_argument('--data-name', type=str, help='name of simulated data', required=True)
    parser.add_argument('--out-dir', type=str, help='directory for saving results', required=True)
    parser.add_argument('--res', type=float, default=0.02, help='leiden clustering resolution', required=True)
    parser.add_argument('--cluster-cells', type=str, choices=['clusterall', 'clustertarget'], help='cluster all cells or target cells', required=True)
    args = parser.parse_args()

    adata = sc.read(f'{args.data_dir}/{args.data_name}.h5ad')

    if args.cluster_cells == 'clusterall':
        # workflow for clusterall (select resolution)
        # define hyperparameters
        neighs=[20]
        ss='sample'
        annot='cell_type'
    
        for ng in neighs:
    
            asub=adata
            adataneigh=format_data_neighs(asub,annot,ss,neighs=ng)
            adataneigh.obs['total_counts'] = adataneigh.X.sum(1)
            adataneigh.obsm["spatial"]=np.array([adataneigh.obs['x'],adataneigh.obs['y']]).transpose().astype('float64')
            adataneigh.X=np.nan_to_num(adataneigh.X)
            adataneigh=adataneigh[adataneigh.obs['total_counts']>3]
            adataneigh.raw=adataneigh
            #sc.pp.normalize_total(adataneigh, target_sum=None)
            #sc.pp.log1p(adataneigh)
            sc.pp.neighbors(adataneigh, n_neighbors=20,n_pcs=0)
            sc.tl.umap(adataneigh,min_dist=0.1)
    
            sc.tl.leiden(adataneigh, resolution=args.res, key_added='domain')
            tmp = adataneigh.obs.copy()
            n_domains = len(np.unique(adataneigh.obs['domain']))
            if n_domains > 6:
                domain_map = {'0': '0', '1': '1', '2': '2', '3': '3', '4': '4'}
                for d_ in adataneigh.obs['domain'].unique():
                    if int(d_) not in [0, 1, 2, 3, 4]:
                        domain_map[d_] = '5'
                tmp['domain'] = tmp['domain'].map(domain_map)
                n_domains = len(np.unique(tmp['domain']))
                fname = f'{args.out_dir}/{args.data_name}_{args.cluster_cells}_neigh{ng}_res{args.res:.4g}_{n_domains}domain.csv'
                tmp[['cell_id', 'x', 'y', 'cell_type', 'region', 'domain']].to_csv(fname, sep='\t', index=False)

    elif args.cluster_cells == 'clustertarget':
        # workflow for clusterall (select resolution)
        # define hyperparameters
        neighs=[20]
        ss='sample'
        annot='cell_type'
    
        for ng in neighs:
    
            asub=adata
            adataneigh=format_data_neighs(asub,annot,ss,neighs=ng)
            adataneigh.obs['total_counts'] = adataneigh.X.sum(1)
            adataneigh.obsm["spatial"]=np.array([adataneigh.obs['x'],adataneigh.obs['y']]).transpose().astype('float64')
            adataneigh.X=np.nan_to_num(adataneigh.X)
            adataneigh=adataneigh[adataneigh.obs['total_counts']>3]
            adataneigh.raw=adataneigh
            #sc.pp.normalize_total(adataneigh, target_sum=None)
            #sc.pp.log1p(adataneigh)
            sc.pp.neighbors(adataneigh, n_neighbors=20,n_pcs=0)
            sc.tl.umap(adataneigh,min_dist=0.1)
    
            res=args.res
            target_ct = 'A'
            adataneigh_target = adataneigh[adataneigh.obs['cell_type']==target_ct].copy()
            sc.tl.leiden(adataneigh_target, resolution=res, key_added='domain')
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

            fname = f'{args.out_dir}/{args.data_name}_{args.cluster_cells}_neigh{ng}_res{res:.4g}_{n_domains}domain.csv'
            tmp[['cell_id', 'x', 'y', 'cell_type', 'region', 'domain']].to_csv(fname, sep='\t', index=False)
                    