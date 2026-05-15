import sys
import time
import os
import logging
import argparse
from datetime import datetime
import math

import scanpy
import numpy as np
import scipy.sparse as sp
from scipy.spatial import Delaunay

import torch
from torch import optim

if '/data2/xiaojiashun/niche/STACI-master' not in sys.path:
    sys.path.append('/data2/xiaojiashun/niche/STACI-master')
import gae.gae.optimizer as optimizer
import gae.gae.model
import gae.gae.preprocessing as preprocessing

import pickle
import seaborn as sns
import umap
import pandas as pd
from sklearn.preprocessing import scale
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import euclidean_distances

# 修改建图函数，使其接收局部的 chunk_data
def getA_delaunay(chunk_data, savepath=None):
    sobj_coord_np=chunk_data.obs[['x','y']].to_numpy()
    tri = Delaunay(sobj_coord_np)
    
    a_size=sobj_coord_np.shape[0]
    a=sp.lil_matrix((a_size,a_size))
    for tri_i in range(tri.simplices.shape[0]):
        tri_i_idx=tri.simplices[tri_i,:]
        a[tri_i_idx[0],tri_i_idx[1]]=1
        a[tri_i_idx[1],tri_i_idx[0]]=1
        a[tri_i_idx[0],tri_i_idx[2]]=1
        a[tri_i_idx[2],tri_i_idx[0]]=1
        a[tri_i_idx[1],tri_i_idx[2]]=1
        a[tri_i_idx[2],tri_i_idx[1]]=1
    
    a=a.tocsr()
    if savepath !=None:
        sp.save_npz(savepath,a)
    return a

def getA_knn(chunk_data, k, a_mode, savepath=None):
    sobj_coord_np=chunk_data.obs[['x','y']].to_numpy()
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='ball_tree').fit(sobj_coord_np)
    a=nbrs.kneighbors_graph(sobj_coord_np,mode=a_mode)
    if a_mode=='connectivity':
        a=a-sp.identity(sobj_coord_np.shape[0],format='csr')
    if a_mode=='distance':
        a[a!=0]=1/a[a!=0]
    if savepath !=None:
        sp.save_npz(savepath,a)
    return a

def getA_physicalDist(chunk_data, distThresh, a_mode, savepath=None):
    sobj_coord_np=chunk_data.obs[['x','y']].to_numpy()
    allDist=euclidean_distances(sobj_coord_np,sobj_coord_np)
    minDist=np.min(allDist[allDist != 0])
    thresh=distThresh*minDist
    a=np.zeros_like(allDist)
    edgeIdx=np.logical_and(allDist<thresh,allDist!=0)
    if a_mode=='connectivity':
        a[edgeIdx]=1
    elif a_mode=='distance':
        a[edgeIdx]=1/allDist[edgeIdx]
    else:
        print('a mode not supported')
    a=sp.csr_matrix(a)
    if savepath !=None:
        sp.save_npz(savepath,a)
    return a

def getA_physicalDist_gradient(chunk_data, distThresh, decay, savepath=None):
    sobj_coord_np=chunk_data.obs[['x','y']].to_numpy()
    allDist=euclidean_distances(sobj_coord_np,sobj_coord_np)
    minDist=np.min(allDist[allDist != 0])
    a=np.zeros_like(allDist)
    
    prethresh=0
    for i in range(len(distThresh)):
        thresh=distThresh[i]*minDist
        edgeIdx=np.logical_and(np.logical_and(allDist>=prethresh,allDist<thresh),allDist!=0)
        a[edgeIdx]+=decay[i]
        prethresh=thresh
    a=sp.csr_matrix(a)
    if savepath !=None:
        sp.save_npz(savepath,a)
    return a


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, help='directory of simulated data', required=True)
    parser.add_argument('--data-name', type=str, help='name of simulated data', required=True)
    parser.add_argument('--out-dir', type=str, help='directory for saving results', required=True)
    # 增加 chunk_size 参数防止显存溢出
    parser.add_argument('--chunk-size', type=int, default=20000, help='Max cells per batch to prevent CUDA OOM')
    args = parser.parse_args()

    ## Settings
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    use_cuda=True 
    fastmode=False 
    seed=3 
    useSavedMaskedEdges=False 
    maskedgeName='knn20_connectivity'
    epochs=5001 
    saveFreq=100 
    lr=0.001 
    lr_adv=0.001 
    weight_decay=0 

    hidden1=6000 
    hidden2=6000 
    fc_dim1=6000 
    adv_hidden=128 

    dropout=0.01 
    testNodes=0.1 
    valNodes=0.05 
    XreconWeight=20  
    advWeight=2 
    model_str='gcn_vae_xa_e2_d1_dca' 
    adv=None  
    ridgeL=0.01 
    shareGenePi=True 

    dataname = args.data_name
    training_sample_X='logminmax' 
    switchFreq=10 
    standardizeX=False 
    name='newModel' 
    useA=True

    logsavepath=f'{args.out_dir}/log/train_gae/'+dataname
    modelsavepath=f'{args.out_dir}/models/train_gae/'+dataname
    plotsavepath=f'{args.out_dir}/plots/train_gae/'+dataname
    datadir=args.data_dir
    savedir=args.out_dir 
    adj_dir=os.path.join(savedir,'a')
    asavepath=f'{args.out_dir}/a' 
    os.makedirs(asavepath, exist_ok=True)

    # 1. 载入全量数据并进行分块 (Chunking) 以解决 OOM
    print(f"Loading full data from {datadir}/{dataname}...")
    if training_sample_X in ['corrected','scaled']:
        full_data = scanpy.read_h5ad(datadir+f'/{dataname}-scaled.h5ad')
    else:
        full_data = scanpy.read_h5ad(datadir+f'/{dataname}.h5ad')
        
    if 'dca' in model_str:
        full_raw_data = scanpy.read_h5ad(datadir+f'/{dataname}.h5ad')
    
    num_features=full_data.shape[1] 
    total_cells = full_data.shape[0]
    chunk_size = args.chunk_size
    num_chunks = math.ceil(total_cells / chunk_size)
    print(f"Total cells: {total_cells}. Splitting into {num_chunks} chunks of max size {chunk_size}.")

    # 训练样本列表更新为所有的分块名称
    training_samples = [f"{dataname}_chunk{i}" for i in range(num_chunks)]
    
    featureslist={}
    features_raw_list={}
    adj_list={}

    for i, s in enumerate(training_samples):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, total_cells)
        chunk_data = full_data[start_idx:end_idx].copy()
        
        # 为当前块生成 KNN 图并保存
        dist_path = os.path.join(asavepath,f'{s}_knn20_distance.npz')
        conn_path = os.path.join(asavepath,f'{s}_{maskedgeName}.npz')
        getA_knn(chunk_data, 20, 'distance', dist_path)
        getA_knn(chunk_data, 20, 'connectivity', conn_path)
        
        # 准备特征张量
        if training_sample_X in ['corrected','scaled']:
            featureslist[s+'X_'+training_sample_X] = torch.tensor(chunk_data.layers[training_sample_X])
        else:
            try:
                scaleddata_train = chunk_data.layers['counts'].A
            except:
                scaleddata_train = chunk_data.layers['counts']
            if training_sample_X=='logminmax':
                featurelog_train=np.log2(scaleddata_train+1/2) # 修改为直接处理 csr_matrix/array
                scaler = MinMaxScaler()
                featurelog_train_minmax=np.transpose(scaler.fit_transform(np.transpose(featurelog_train)))
                featureslist[s+'X_'+training_sample_X]=torch.tensor(featurelog_train_minmax)
            elif training_sample_X=='logminmax10':
                featurelog_train=np.log2(scaleddata_train+1/2)
                scaler = MinMaxScaler(feature_range=(0,10))
                featurelog_train_minmax=np.transpose(scaler.fit_transform(np.transpose(featurelog_train)))
                featureslist[s+'X_'+training_sample_X]=torch.tensor(featurelog_train_minmax)
                
        # 记录邻接矩阵
        adj_list[s]=sp.load_npz(conn_path)

        # 准备原数据特征
        if 'dca' in model_str:
            raw_chunk = full_raw_data[start_idx:end_idx].copy()
            features_raw_list[s+'X_'+'raw'] = torch.tensor(raw_chunk.layers['counts'])

    # 预处理邻接矩阵列表
    adjnormlist={}
    pos_weightlist={}
    normlist={}
    for ai in adj_list.keys():
        adjnormlist[ai]=preprocessing.preprocess_graph(adj_list[ai])
        pos_weightlist[ai] = torch.tensor(float(adj_list[ai].shape[0] * adj_list[ai].shape[0] - adj_list[ai].sum()) / adj_list[ai].sum())
        normlist[ai] = adj_list[ai].shape[0] * adj_list[ai].shape[0] / float((adj_list[ai].shape[0] * adj_list[ai].shape[0] - adj_list[ai].sum()) * 2)
        
        adj_label=adj_list[ai] + sp.eye(adj_list[ai].shape[0])
        adj_list[ai]=torch.tensor(adj_label.todense())

    # Set cuda and seed
    np.random.seed(seed)
    if use_cuda and (not torch.cuda.is_available()):
        print('cuda not available')
        use_cuda=False
    torch.manual_seed(seed)
    if use_cuda:
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.enabled = True

    os.makedirs(logsavepath, exist_ok=True)
    os.makedirs(modelsavepath, exist_ok=True)
    os.makedirs(plotsavepath, exist_ok=True)
        
    mse=torch.nn.MSELoss()
    # Create model
    model = gae.gae.model.GCNModelVAE_XA_e2_d1_DCA(num_features, hidden1,hidden2,fc_dim1, dropout)
    loss_kl=optimizer.optimizer_kl
    loss_x=optimizer.optimizer_zinb
    loss_a=optimizer.optimizer_CE

    if 'NB' in name:
        print('using NB loss for X')
        loss_x=optimizer.optimizer_nb
        
    if use_cuda:
        model.cuda()

    optimizerVAEXA = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    log_file = os.path.join(logsavepath, f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()]
    )
    logger = logging.getLogger()

    def train(epoch):
        t = time.time()
        model.train()
        optimizerVAEXA.zero_grad()
        
        adj_recon,mu,logvar,z,features_recon = model(features, adj_norm)

        loss_kl_train=loss_kl(mu, logvar, train_nodes_idx)
        
        if 'dca' in model_str:
            if 'NB' in name:
                loss_x_train=loss_x(features_recon, features,train_nodes_idx,XreconWeight)
            else:
                loss_x_train=loss_x(features_recon, features,train_nodes_idx,XreconWeight,ridgeL,features_raw)
        else:
            loss_x_train=loss_x(features_recon, features,train_nodes_idx,XreconWeight,mse)
        
        loss_a_train=loss_a(adj_recon, adj_label, pos_weight, norm,train_nodes_idx)
        
        loss=loss_kl_train+loss_x_train
        if useA:
            loss=loss+loss_a_train

        loss.backward()
        optimizerVAEXA.step()

        if not fastmode:
            model.eval()
            adj_recon,mu,logvar,z, features_recon = model(features, adj_norm)
        
        if 'dca' in model_str:
            if 'NB' in name:
                loss_x_val=loss_x(features_recon, features,val_nodes_idx,XreconWeight)
            else:
                loss_x_val=loss_x(features_recon, features,val_nodes_idx,XreconWeight,ridgeL,features_raw)
        else:
            loss_x_val=loss_x(features_recon, features,val_nodes_idx,XreconWeight,mse)
        
        loss_a_val=loss_a(adj_recon, adj_label, pos_weight, norm,val_nodes_idx)
        
        loss_val=loss_x_val
        if useA:
            loss_val=loss_val+loss_a_val

        train_loss_ep = loss.item()
        train_loss_kl_ep = loss_kl_train.item()
        train_loss_x_ep = loss_x_train.item()
        train_loss_a_ep = loss_a_train.item()
        val_loss_ep = loss_val.item()
        val_loss_x_ep = loss_x_val.item()
        val_loss_a_ep = loss_a_val.item()

        logger.info(
            training_samples_t + ' Epoch: {:04d} '.format(epoch) +
            'loss_train: {:.4f} '.format(loss.item()) +
            'loss_kl_train: {:.4f} '.format(loss_kl_train.item()) +
            'loss_x_train: {:.4f} '.format(loss_x_train.item()) +
            'loss_a_train: {:.4f} '.format(loss_a_train.item()) +
            'loss_val: {:.4f} '.format(loss_val.item()) +
            'loss_x_val: {:.4f} '.format(loss_x_val.item()) +
            'loss_a_val: {:.4f} '.format(loss_a_val.item()) +
            'time: {:.4f}s'.format(time.time() - t)
        )
        
        return train_loss_ep,train_loss_kl_ep,train_loss_x_ep,train_loss_a_ep,val_loss_ep,val_loss_x_ep,val_loss_a_ep
        
    train_loss_eps=[None]*epochs
    train_loss_kl_eps=[None]*epochs
    train_loss_x_eps=[None]*epochs
    train_loss_a_eps=[None]*epochs
    train_loss_adv_eps=[None]*epochs
    train_loss_advD_eps=[None]*epochs
    val_loss_eps=[None]*epochs
    val_loss_x_eps=[None]*epochs
    val_loss_a_eps=[None]*epochs
    val_loss_adv_eps=[None]*epochs
    val_loss_advD_eps=[None]*epochs
    t_ep=time.time()

    os.makedirs(f'{savedir}/trainMask', exist_ok=True)

    # 保存上一轮处理的 chunk 标识用于判定是否需要释放显存
    last_samples_t = None

    for ep in range(epochs):
        # 依赖代码内置样本切换机制，每隔 switchFreq 的 Epochs 将轮转到下一个 chunk 以控制并行占用率。
        t_idx=int(ep/switchFreq)%len(training_samples)
        training_samples_t=training_samples[t_idx]

        # 换块时，为了显存彻底回收，清理释放
        if last_samples_t is not None and last_samples_t != training_samples_t:
            if use_cuda:
                torch.cuda.empty_cache()

        adj_norm=adjnormlist[training_samples_t].cuda().float() if use_cuda else adjnormlist[training_samples_t].float()
        adj_label=adj_list[training_samples_t].cuda().float() if use_cuda else adj_list[training_samples_t].float()
        features=featureslist[training_samples_t+'X_'+training_sample_X].cuda().float() if use_cuda else featureslist[training_samples_t+'X_'+training_sample_X].float()
        pos_weight=pos_weightlist[training_samples_t]
        norm=normlist[training_samples_t]
        
        if 'dca' in model_str:
            features_raw=features_raw_list[training_samples_t+'X_raw'].cuda() if use_cuda else features_raw_list[training_samples_t+'X_raw']
        
        num_nodes,_ = features.shape
        
        maskpath=os.path.join(savedir,'trainMask',training_samples_t+'_'+maskedgeName+'_seed'+str(seed)+'.pkl')
        if useSavedMaskedEdges and os.path.exists(maskpath):
            with open(maskpath, 'rb') as input:
                maskedgeres = pickle.load(input)
        else:
            maskedgeres= preprocessing.mask_nodes_edges(features.shape[0],testNodeSize=testNodes,valNodeSize=valNodes,seed=seed)
            with open(maskpath, 'wb') as output:
                pickle.dump(maskedgeres, output, pickle.HIGHEST_PROTOCOL)
                
        train_nodes_idx,val_nodes_idx,test_nodes_idx = maskedgeres
        if use_cuda:
            train_nodes_idx=train_nodes_idx.cuda()
            val_nodes_idx=val_nodes_idx.cuda()
            test_nodes_idx=test_nodes_idx.cuda()
        
        train_loss_eps[ep],train_loss_kl_eps[ep],train_loss_x_eps[ep],train_loss_a_eps[ep],val_loss_eps[ep],val_loss_x_eps[ep],val_loss_a_eps[ep]=train(ep)

        last_samples_t = training_samples_t

        if ep%saveFreq == 0:
            torch.save(model.cpu().state_dict(), os.path.join(modelsavepath,str(ep)+'.pt'))
            if use_cuda:
                model.cuda()
                
    if use_cuda:
        torch.cuda.empty_cache()
        
    print(' total time: {:.4f}s'.format(time.time() - t_ep))

    with open(os.path.join(logsavepath,'train_loss'), 'wb') as output:
        pickle.dump(train_loss_eps, output, pickle.HIGHEST_PROTOCOL)
    with open(os.path.join(logsavepath,'train_loss_kl'), 'wb') as output:
        pickle.dump(train_loss_kl_eps, output, pickle.HIGHEST_PROTOCOL)
    with open(os.path.join(logsavepath,'train_loss_x'), 'wb') as output:
        pickle.dump(train_loss_x_eps, output, pickle.HIGHEST_PROTOCOL)
    with open(os.path.join(logsavepath,'train_loss_a'), 'wb') as output:
        pickle.dump(train_loss_a_eps, output, pickle.HIGHEST_PROTOCOL)
    with open(os.path.join(logsavepath,'val_loss'), 'wb') as output:
        pickle.dump(val_loss_eps, output, pickle.HIGHEST_PROTOCOL)
    with open(os.path.join(logsavepath,'val_loss_x'), 'wb') as output:
        pickle.dump(val_loss_x_eps, output, pickle.HIGHEST_PROTOCOL)
    with open(os.path.join(logsavepath,'val_loss_a'), 'wb') as output:
        pickle.dump(val_loss_a_eps, output, pickle.HIGHEST_PROTOCOL)