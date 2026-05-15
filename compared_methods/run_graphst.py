import os, argparse
import torch
import numpy as np
import pandas as pd
import scanpy as sc
from sklearn import metrics
from GraphST import GraphST
from GraphST.utils import clustering
from GraphST import utils as st_utils
import rpy2.robjects as robjects

def patched_mclust_R(adata, num_cluster, modelNames='EEE', used_obsm='emb_pca', random_seed=2020):
    # 1. 获取数据并做严格的安全检查
    data = np.array(adata.obsm[used_obsm], dtype=np.float64)
    if len(data.shape) != 2:
        raise ValueError(f"Expected 2D array, got shape {data.shape}")
    nr, nc = data.shape
    
    if nr < num_cluster:
        raise ValueError(f"Cell number ({nr}) is smaller than cluster number ({num_cluster}). Mclust cannot proceed.")

    # 2. 将数据打平为一维，只传递最基础的数据类型，骗过 rpy2 的自动转换器
    robjects.globalenv['tmp_data'] = robjects.FloatVector(data.flatten(order='C'))
    robjects.globalenv['tmp_nr'] = int(nr)
    robjects.globalenv['tmp_nc'] = int(nc)
    robjects.globalenv['tmp_G'] = int(num_cluster)
    robjects.globalenv['tmp_modelNames'] = str(modelNames)
    robjects.globalenv['tmp_seed'] = int(random_seed)
    
    # 3. 【绝对防御】：将核心逻辑封装为纯 R 代码字符串！
    r_script = """
    suppressPackageStartupMessages(library(mclust))
    
    # 在纯 R 环境中重组矩阵，绝对不会丢失 dim 属性
    mat <- matrix(tmp_data, nrow=tmp_nr, ncol=tmp_nc, byrow=TRUE)
    
    # 显式赋予行列名，提前满足 Mclust 的内部校验条件，彻底封杀 dimnames 报错
    rownames(mat) <- paste0("Cell_", 1:tmp_nr)
    colnames(mat) <- paste0("PC_", 1:tmp_nc)
    
    set.seed(tmp_seed)
    
    # 执行聚类
    res <- Mclust(mat, G=tmp_G, modelNames=tmp_modelNames)
    
    # 安全提取结果 (应对极端情况下的收敛失败)
    if (is.null(res)) {
        classification <- rep(NA, tmp_nr)
    } else {
        classification <- res$classification
    }
    
    # 清理环境变量，释放内存
    rm(tmp_data, tmp_nr, tmp_nc, tmp_G, tmp_modelNames, tmp_seed, mat)
    
    classification
    """
    
    # 4. 运行这段纯 R 脚本
    classification = robjects.r(r_script)
    
    # 5. 存回 adata
    adata.obs['mclust'] = np.array(classification).astype(str)
    
    return adata

# 3. 【核心步骤】：在 GraphST.utils 模块的作用域中，把原有的 mclust_R 替换为我们的修复版
st_utils.mclust_R = patched_mclust_R

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, help='directory of simulated data', required=True)
    parser.add_argument('--data-name', type=str, help='name of simulated data', required=True)
    parser.add_argument('--out-dir', type=str, help='directory for saving results', required=True)
    parser.add_argument('--num-cluster', type=int, default=6, help='number of clusters', required=True)
    parser.add_argument('--cluster-cells', type=str, choices=['clusterall', 'clustertarget'], help='cluster all cells or target cells', required=True)
    args = parser.parse_args()

    adata = sc.read(f'{args.data_dir}/{args.data_name}.h5ad')
    
    # define model
    model = GraphST.GraphST(adata, device=torch.device('cuda:0'))

    # train model
    adata = model.train()

    # set radius to specify the number of neighbors considered during refinement
    radius = 20
    if args.cluster_cells == 'clusterall':
        adata_cluster = adata.copy()
    elif args.cluster_cells == 'clustertarget':
        adata_cluster = adata[adata.obs['cell_type']=='A'].copy()
    clustering(adata_cluster, args.num_cluster, radius=radius, method='mclust', refinement=True) # For DLPFC dataset, we use optional refinement step.
    cluster_df = adata_cluster.obs[['cell_id', 'x', 'y', 'cell_type', 'region', 'domain']].copy()
    cluster_df.to_csv(f'{args.out_dir}/{args.data_name}_{args.cluster_cells}_{args.num_cluster}domain.csv', sep='\t', index=False)
