import os, sys, time, tqdm, logging
from datetime import datetime
import numpy as np
import pandas as pd
import scanpy as sc
from scipy.spatial.distance import cdist

from rpy2.robjects.packages import importr
import rpy2.robjects as ro
from rpy2.robjects import numpy2ri, pandas2ri
from covariance_test import cov_test


def setup_logger(name, level=logging.INFO):
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if logger.hasHandlers():
        logger.handlers.clear()

    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s', '%Y-%m-%d %H:%M:%S')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    return logger


logger = setup_logger("nichescope", level=logging.INFO)
logger_sub = setup_logger("nichescope.sub", level=logging.WARNING)

    
def matrix_standard(X):
    X_std = np.std(X, axis=0)
    X_std[X_std==0] = 1
    return (X - np.mean(X, axis=0)) / X_std


### niche composition
def compute_N(
    adata, 
    target_ct, 
    *,
    sigma=20, 
    cutoff=0.05, 
    cell_id_key='cell_id',
    cell_type_key='cell_type',
    max_chunk_target=10000,
    **kwargs
):

    """
    Compute neighborhood cell type composition matirx N of the target cell type.

    Parameters
    ----------
    adata : anndata.AnnData
        Input ST AnnData object.

    target_ct : str
        Name of target cell type.

    sigma : int or float, default=20
        Gaussian kernel bandwidth.

    cutoff : float, default=0.05
        Setting kernel value lower than cutoff to 0.

    cell_id_key : str, default='cell_id'
        Cell ID key in adata.obs.

    cell_type_key : str, default='cell_type'
        Cell type key in adata.obs.

    max_chunk_target : int, default=10000
        Compute matrix N by chunk if number of cells exceeding max_chunk_target.

    Returns
    -------
    N_target_df : pandas.DataFrame
        Neighborhood cell type composition dataframe.
    
    """
    
    ## num cell id
    loc = adata.obs[[cell_id_key, 'x', 'y', cell_type_key]]
    ct_dummy = pd.get_dummies(loc, columns=[cell_type_key], dtype=float)
    group_names = [x.replace(f'{cell_type_key}_', '') for x in ct_dummy.columns[3:].tolist()]
    ct_dummy = ct_dummy.iloc[:,3:].values
    logger_sub.debug(f'ct_dummy: {ct_dummy.shape}.')

    ## calculate niche matrix
    loc_target = adata.obs.loc[adata.obs[cell_type_key]==target_ct, ['x', 'y']].values
    loc_all = adata.obs[['x','y']].values
    logger_sub.debug(f'{len(loc_target)} {target_ct}, {len(loc_all)} cells in total.')

    if loc_target.shape[0] > max_chunk_target:
        n_chunk = np.ceil(loc_target.shape[0]/max_chunk_target).astype(int)
        loc_target_chunks = np.array_split(loc_target, n_chunk, axis=0)
        N_target_chunks = []
        for i, loc_target_chunk in enumerate(loc_target_chunks):
            K_target_chunk = cdist(loc_target_chunk, loc_all)
            K_target_chunk = np.exp(-K_target_chunk**2/sigma**2)
            K_target_chunk[K_target_chunk < cutoff] = 0
            N_target_chunk = K_target_chunk @ ct_dummy
            logger_sub.debug(f'N_target_chunk {i} {N_target_chunk.shape}.')
            N_target_chunks.append(N_target_chunk)
        N_target = np.vstack(N_target_chunks)
    else:
        K_target = cdist(loc_target, loc_all)
        K_target = np.exp(-K_target**2/sigma**2)
        K_target[K_target < cutoff] = 0
        N_target = K_target @ ct_dummy

    N_target_df = pd.DataFrame(N_target, columns=group_names)

    return N_target_df


def select_candidate_genes(
    adata,
    target_ct,
    N_target_df,
    *,
    adata2=None,
    N_target_df2=None,
    use_cov_test_genes=True, 
    n_hvg=3000, 
    filter_genes=None,
    cov_thres=0.05, 
    max_cand_genes=500,
    cov_test_null=False, 
    standardize=False,
    cell_type_key='cell_type',
    **kwargs
):

    """
    Select candidate genes via covariance tests.

    Parameters
    ----------
    adata : anndata.AnnData
        Input ST AnnData object.

    target_ct : str
        Name of target cell type.

    N_target_df : pandas.DataFrame
        Neighborhood cell type composition dataframe.

    adata2 : anndata.AnnData, default=None
        ST AnnData object for a second dataset (optional).

    N_target_df2 : pandas.DataFrame, default=None
        Neighborhood cell type composition dataframe of the second dataset (optional).

    use_cov_test_genes : bool, default=True
        Whether to use candidate genes selected by covariance tests.

    n_hvg : int, default=3000
        Use top n_hvg highly variable genes.

    filter_genes : list, default=None
        Restrict the analysis to genes in this list.

    cov_thres : float, default=0.05
        P-value threshold for selecting candidate genes.

    max_cand_genes : int, default=500
        Maximum number of candidate genes used for CCA.

    cov_test_null : bool, default=False
        Whether to perform covariance tests on permuted data to generate null distribution of p-values.

    standardize: bool, default=False
        Whether to standardize the gene expression and neighborhood composition matrix.

    cell_type_key : str, default='cell_type'
        Cell type key in adata.obs.

    Returns
    -------
    cov_target : pandas.DataFrame
        Gene-level statistics and p-values of covariance tests.

    cca_genes : list
        Candidate genes significant in covariance tests.

    cov_real_null : list
        Gene-level p-values of covariance tests on real and permuted data.
    
    """
    
    if adata2 is not None:
        assert N_target_df2 is not None, 'N_target_df2 not provided for combined covariance test!'
        use_gene_names = list(set(adata.var.loc[adata.var.highly_variable_rank<n_hvg,].index.tolist()) & set(adata2.var.loc[adata2.var.highly_variable_rank<n_hvg,].index.tolist()))
        if filter_genes is not None:
            use_gene_names = list(set(use_gene_names) & set(filter_genes))
            logger_sub.debug(f'{len(use_gene_names)} use genes.')
        cov_X1 = adata[adata.obs[cell_type_key]==target_ct, use_gene_names].X.toarray()
        cov_X2 = adata2[adata2.obs[cell_type_key]==target_ct, use_gene_names].X.toarray()
        cov_X1 = matrix_standard(cov_X1)
        cov_X2 = matrix_standard(cov_X2)
        cov_X = np.concatenate((cov_X1, cov_X2),axis=0).T
        keep_gene_idx = np.where(cov_X.sum(axis=1)!=0)[0]
        cov_X = cov_X[keep_gene_idx,:]
    else:
        use_gene_names = adata.var.loc[adata.var.highly_variable_rank<n_hvg,].index.tolist()
        if filter_genes is not None:
            use_gene_names = list(set(use_gene_names) & set(filter_genes))
            logger_sub.debug(f'{len(use_gene_names)} use genes.')
        cov_X = adata[adata.obs[cell_type_key]==target_ct, use_gene_names].X.toarray().T
        keep_gene_idx = np.where(cov_X.sum(axis=1)!=0)[0]
        cov_X = cov_X[keep_gene_idx,:]
        if standardize:
            cov_X = matrix_standard(cov_X)
    keep_gene_names = [use_gene_names[i] for i in keep_gene_idx]
    cov_real_null = None
    cov_target = None
    cca_genes = []

    if use_cov_test_genes:
    
        if adata2 is not None:
            cov_N1 = N_target_df.values
            cov_N2 = N_target_df2.values
            cov_N1 = matrix_standard(cov_N1)
            cov_N2 = matrix_standard(cov_N2)
            cov_N = np.concatenate((cov_N1, cov_N2),axis=0)
        else:
            cov_N = N_target_df.values
            if standardize:
                cov_N = matrix_standard(cov_N)
        keep_cell_idx = np.where(cov_N.sum(axis=1)!=0)[0]
        cov_X = cov_X[:,keep_cell_idx]
        cov_N = cov_N[keep_cell_idx,:]
        logger_sub.debug(f'cov_X: {cov_X.shape}; cov_N: {cov_N.shape}')
    
        if (np.linalg.eigvals(cov_N.T@cov_N)<=0).sum()>0:
            logger_sub.error('Neighbour matrix N is Singular Matrix! Increase Gaussian kernel bandwith sigma or remove cell type columns with all zeros.')
            return cov_target, cca_genes, cov_real_null
    
        cov_target_ = cov_test(cov_X, cov_N)
        cov_target = pd.concat(
            [pd.DataFrame({
                'gene_id': np.array(keep_gene_names)[np.array(cov_target_['gene_ids'])], 
                'vec_stat': cov_target_['stats'].flatten(),
                'vec_pval': cov_target_['res_stest'].flatten()}),
            cov_target_['res_mtest'].reset_index(drop=True)], axis=1
        )
        logger_sub.debug(f'{cov_target.shape[0]} / {len(keep_gene_names)} genes have pvalue.')
        logger_sub.debug(f'No pvalue genes: {list(set(keep_gene_names)-set(cov_target.gene_id))}.')
    
        ## cov test null distribution: 5 reps
        if cov_test_null:
            logger_sub.debug('Cov test on permuted data...')
            cov_null_reps = []
            for r in range(5):
                np.random.seed(2*r+1)
                cov_X_null = cov_X.T.copy()
                np.random.shuffle(cov_X_null)
                cov_X_null = cov_X_null.T
                np.random.shuffle(cov_X_null)
                
                np.random.seed(2*r+2)
                cov_N_null = cov_N.copy()
                np.random.shuffle(cov_N_null)
                
                cov_target_null_ = cov_test(cov_X_null, cov_N_null)
                cov_null_reps.append(cov_target_null_['res_stest'].flatten())
            cov_real_null = [cov_target['vec_pval'].values] + cov_null_reps

        ## cand genes
        cov_target_sorted = cov_target.sort_values(by=['adjustedPval']).reset_index(drop=True)
        cand_genes_index = cov_target_sorted.loc[cov_target_sorted.adjustedPval<cov_thres,'gene_id'].values.tolist()
        cca_genes = cand_genes_index
        if use_cov_test_genes and max_cand_genes is not None:
            n_cand_genes0 = len(cca_genes)
            cca_genes = cca_genes[:max_cand_genes]
    else:
        cca_genes = keep_gene_names

    return cov_target, cca_genes, cov_real_null
    

def cca(
    adata,
    target_ct,
    N_target_df,
    cca_genes,
    *,
    cca_comp=8,
    px=0.6,
    pz=0.5,
    sort_comp_by_corr=False,
    cell_type_key='cell_type',
    **kwargs
):

    """
    Sparse nonnegative canonical correlation analysis.

    Parameters
    ----------
    adata : anndata.AnnData
        Input ST AnnData object.

    target_ct : str
        Name of target cell type.

    N_target_df : pandas.DataFrame
        Neighborhood cell type composition dataframe.

    cca_genes : list
        Candidate genes significant in covariance tests.

    cca_comp : int, default=8
        Number of components in CCA.

    px : int, default=0.6
        Parameter in [0, 1] controlling sparsity of gene coefficients u. A smaller px leads to less nonzero values in gene coefficients.
    
    pz : int, default=0.5
        Parameter in [0, 1] controlling sparsity of gene coefficients v. A smaller pz leads to less nonzero values in cell type coefficients.

    sort_comp_by_corr : bool, default=False
        Whether to sort CCA components by the descending order of CCA correlations.

    cell_type_key : str, default='cell_type'
        Cell type key in adata.obs.

    Returns
    -------
    udf : pandas.DataFrame
        Gene coefficients of all the CCA components.

    vdf : pandas.DataFrame
        Cell type coefficients of all the CCA components.

    cors : numpy.array
        Correlations between Xu and Nv of all the CCA components.

    ds : numpy.array
        Scaling factors of all the CCA components, used in adjusting the previous components.
    
    """

    ## rpy2 
    r = ro.r
    importr('PMA')
    CCA = r['CCA']
    numpy2ri.activate()
    pandas2ri.activate()
    
    ## input
    cca_X = adata[adata.obs[cell_type_key]==target_ct, cca_genes].X.toarray()
    cca_N = N_target_df.values
    keep_cell_idx = np.where(cca_N.sum(axis=1)!=0)[0]
    cca_X = cca_X[keep_cell_idx,:]
    cca_N = cca_N[keep_cell_idx,:]
    logger_sub.debug(f'cca_X: {cca_X.shape}; cca_N: {cca_N.shape}')

    ## nonneg cca
    pmd = CCA(cca_X, cca_N, K=cca_comp, penaltyx=px, penaltyz=pz, typex="standard", typez="standard", standardize=True, upos=True, vpos=True, trace=False)

    u = pmd.rx2('u')
    v = pmd.rx2('v')
    cors = pmd.rx2('cors')
    ds = pmd.rx2('d')
    if sort_comp_by_corr:
        comp_order = np.argsort(cors)[::-1]
        comp_order = comp_order[cors[comp_order]>0]
        cors = cors[comp_order]
        u = u[:,comp_order]
        v = v[:,comp_order]
        cca_comp = len(comp_order)
        ds = ds[comp_order]   

    ## post proc
    udf = pd.DataFrame(u, index=cca_genes, columns=[f'comp{i}' for i in range(1,cca_comp+1)])
    vdf = pd.DataFrame(v, index=N_target_df.columns, columns=[f'comp{i}' for i in range(1,cca_comp+1)])

    return udf, vdf, cors, ds

### niche score
def compute_niche_score(
    adata, 
    target_ct,
    N_target_df,
    udf, 
    vdf, 
    *, 
    cell_id_key='cell_id', 
    cell_type_key='cell_type', 
    sub_cell_type_key='sub_cell_type',
    **kwargs
):
    
    """
    Compute niche scores.

    Parameters
    ----------
    adata : anndata.AnnData
        Input ST AnnData object.

    target_ct : str
        Name of target cell type.

    N_target_df : pandas.DataFrame
        Neighborhood cell type composition dataframe.

    udf : pandas.DataFrame
        Gene coefficients of all the CCA components.

    vdf : pandas.DataFrame
        Cell type coefficients of all the CCA components.

    cell_id_key : str, default='cell_id'
        Cell ID key in adata.obs.
        
    cell_type_key : str, default='cell_type'
        Cell type key in adata.obs.

    sub_cell_type_key : str, default='sub_cell_type'
        Sub cell type key in adata.obs (optional).

    Returns
    -------
    score_df : pandas.DataFrame
        Niche gene scores, niche composition scores, and niche scores. Columns of form "Xu_comp1", "Nv_comp1", and "S_comp1" are niche gene scores, niche composition scores, and niche scores, respectively.
    
    """

    genes = udf.index.tolist()
    U = udf.values
    V = vdf.values
    ct_df = adata[adata.obs[cell_type_key]==target_ct].obs[[cell_type_key, cell_id_key]].reset_index(drop=True)
    if sub_cell_type_key in adata.obs.columns:
        ct_df = adata[adata.obs[cell_type_key]==target_ct].obs[[cell_type_key, sub_cell_type_key, cell_id_key]].reset_index(drop=True)
    X0 = adata[adata.obs[cell_type_key]==target_ct,genes].X.toarray()
    X1 = matrix_standard(X0)
    N0 = N_target_df.values
    N1 = matrix_standard(N0)
    
    ## niche gene score & niche composition score
    Y1 = X1 @ U
    Y2 = N1 @ V
    X1_df = pd.DataFrame(X0, columns=udf.index.values)
    Y1_df = pd.DataFrame(Y1, columns=[f'Xu_{x}' for x in list(udf.columns)])
    Y2_df = pd.DataFrame(Y2, columns=[f'Nv_{x}' for x in list(udf.columns)])
    XY_df = pd.concat([X1_df, Y1_df, Y2_df], axis=1)
    
    ## niche score
    S_pos = {}
    for comp in udf.columns:
        y1 = Y1_df[f'Xu_{comp}'].values
        y2 = Y2_df[f'Nv_{comp}'].values
        y1_pos = np.maximum(y1, 0)
        y2_pos = np.maximum(y2, 0)
        S_pos[f'S_{comp}'] = np.log1p(y1_pos * y2_pos)
    S_pos_df = pd.DataFrame(S_pos)
    loc_target = adata.obs.loc[adata.obs[cell_type_key]==target_ct, ['x', 'y']].values
    loc_df = pd.DataFrame(loc_target, columns=['x','y'])
    score_df = pd.concat([XY_df, S_pos_df, loc_df, ct_df], axis=1)
    
    return score_df
    

### neighborhood composition, cov test and cca
def nichescope(
    adata, 
    target_ct,
    *,
    sigma=20, 
    cutoff=0.05, 
    n_hvg=3000, 
    filter_genes=None,
    max_cand_genes=500, 
    cov_thres=0.05, 
    cov_test_null=True,
    use_cov_test_genes=True,
    cca_comp=8, 
    px=0.6, 
    pz=0.5,
    sort_comp_by_corr=False,
    cell_id_key='cell_id',
    cell_type_key='cell_type',
    sub_cell_type_key='sub_cell_type',
    **kwargs
):

    """
    Run NicheScope for cell niche detection under single condition.

    Parameters
    ----------
    adata : anndata.AnnData
        Input ST AnnData object.

    target_ct : str
        Name of target cell type.

    sigma : int or float, default=20
        Gaussian kernel bandwidth.

    cutoff : float, default=0.05
        Setting kernel value lower than cutoff to 0.

    n_hvg : int, default=3000
        Use top n_hvg highly variable genes.

    filter_genes : list, default=None
        Restrict the analysis to genes in this list.

    cov_thres : float, default=0.05
        P-value threshold for selecting candidate genes.
        
    max_cand_genes : int, default=500
        Maximum number of candidate genes used for CCA.

    cov_test_null : bool, default=False
        Whether to perform covariance tests on permuted data to generate null distribution of p-values.

    use_cov_test_genes : bool, default=True
        Whether to use candidate genes selected by covariance tests.

    cca_comp : int, default=8
        Number of components in CCA.

    px : int, default=0.6
        Parameter in [0, 1] controlling sparsity of gene coefficients u. A smaller px leads to less nonzero values in gene coefficients.

    pz : int, default=0.5
        Parameter in [0, 1] controlling sparsity of gene coefficients v. A smaller pz leads to less nonzero values in cell type coefficients.

    sort_comp_by_corr : bool, default=False
        Whether to sort CCA components by the descending order of CCA correlations.

    cell_id_key : str, default='cell_id'
        Cell ID key in adata.obs.

    cell_type_key : str, default='cell_type'
        Cell type key in adata.obs.

    sub_cell_type_key : str, default='sub_cell_type'
        Sub cell type key in adata.obs (optional).

    Returns
    -------
    meta : dict
        Dictionary containing parameters and niche detection results.
    
    """

    ## parameters
    params = {
        'sigma': sigma,
        'cutoff': cutoff,
        'n_hvg': n_hvg,
        'max_cand_genes': max_cand_genes,
        'cov_test_null': cov_test_null,
        'cov_thres': cov_thres,
        'use_cov_test_genes': use_cov_test_genes,
        'cca_comp': cca_comp,
        'px': px,
        'pz': pz,
        'sort_comp_by_corr': sort_comp_by_corr,
        'cell_id_key': cell_id_key,
        'cell_type_key': cell_type_key,
        'sub_cell_type_key': sub_cell_type_key
    }
    kwargs = params.copy()
    kwargs['filter_genes'] = filter_genes
    
    ## summarize dataset
    nobs, ngene = adata.shape
    cts = adata.obs[cell_type_key].cat.categories.tolist()
    logger.info(f'Dataset summary: {nobs} cells, {ngene} genes.')
    logger.info(f'{len(cts)} cell types: {cts}.\n')

    t0 = time.time()
    ### preliminary
    N_target_df = compute_N(adata, target_ct, **kwargs)
    logger.info(f'Computed neighborhood composition matrix N of shape {N_target_df.shape}.')

    ### gene selection
    cov_target, cca_genes, cov_real_null = select_candidate_genes(adata, target_ct, N_target_df, standardize=False, **kwargs)
    if len(cca_genes) == 0:
        logger.error(f'No candidate genes provided for CCA! NicheScope stopped.')
        return
    
    ### CCA
    udf, vdf, cors, ds = cca(adata, target_ct, N_target_df, cca_genes, **kwargs)
    logger.info(f'Nonneg CCA cors: {np.round(cors,3)}\n')
    
    ### compute niche score
    score_df = compute_niche_score(adata, target_ct, N_target_df, udf, vdf, **kwargs)

    t1 = time.time()
    logger.info(f'NicheScope on {target_ct}: Finished in {t1 - t0:.1f}s.')

    ## return
    meta = {
        'target_ct': target_ct,
        'params': params,
        'N_target_df': N_target_df,
        'cov_target': cov_target,
        'cov_real_null': cov_real_null,
        'cca_genes': cca_genes,
        'udf': udf,
        'vdf': vdf,
        'cors': cors,
        'ds': ds,
        'score_df': score_df
    }

    return meta

    
