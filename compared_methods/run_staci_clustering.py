import sys
import time
import os

import scanpy
import anndata as ad
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
import umap
import pandas as pd
from sklearn.preprocessing import scale
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN,MiniBatchKMeans,AgglomerativeClustering 

seed = 3
np.random.seed(seed)
def inverseLeakyRelu(v,slope=0.01):
    vnegidx=(v<0)
    v[vnegidx]=1/slope*v[vnegidx]
    return v

def clusterLeiden_single(inArray,n_neighbors,n_pcs,min_dist,resolution,randseed=seed):
    n_pcs=np.min([inArray.shape[0]-1,inArray.shape[1]-1,n_pcs])
    adata=ad.AnnData(inArray)
    scanpy.tl.pca(adata, svd_solver='arpack')
    scanpy.pp.neighbors(adata, n_neighbors=n_neighbors, n_pcs=n_pcs)
    scanpy.tl.umap(adata,min_dist=min_dist,random_state=randseed)
    scanpy.tl.leiden(adata,resolution=resolution,random_state=randseed,key_added='domain')
    return adata.obs['domain'].to_numpy()


if __name__ == '__main__':
    
    os.environ["CUDA_VISIBLE_DEVICES"] = "0" 
    ifplot=False #if umap plots and clustering plots should be generated
    ifcluster=True #if clustering should be performed (This is to provide the option of adjusting plotting parameters without re-run clustering.)
    
    inverseAct='leakyRelu' #This specifies that the latent should be plotted before the leakyRelu activation function. Use None to plot the latent after leakyRelu.
    # inverseAct=None
    plottype='umap' #umap or pca
    pca=PCA()
    npc=50 #number of principal components to plot in the variance ratio plot
    npc_plot=10 #for pairwise pc plots
    
    #plotting and clustering parameters as defined in the respective functions
    minCells=15 #min number of cells for analysis
    clustermethod=['leiden'] #specify which clustering methods to use
    #umap/leiden clustering parameters
    n_neighbors=10
    min_dist=0.25
    n_pcs=40 #for clustering
    resolution=[0.05,0.1,0.2,0.3]
    plotepoch=4000
    savenameAdd=''
    #DBscan
    epslist= [6,8,10]
    min_sampleslist=[15,30,45] 
    #agglomerative
    nclusterlist=[6]
    aggMetric=['euclidean']
    
    #In the plots and clusters of each cell type, it's also possible to plot/cluster multiple related cell types together.
    combineCelltype={'glia':['Astro','Micro', 'OPC', 'Oligo'],'CA':['CA1', 'CA2', 'CA3']}
    
    #network parameters; should be kept the same as in the training script
    use_cuda=True
    fastmode=False #Validate during training pass
    seed=3
    useSavedMaskedEdges=False
    maskedgeName='knn20_connectivity'
    hidden1=6000 #Number of units in hidden layer 1
    hidden2=6000 #Number of units in hidden layer 2
    # hidden3=16
    fc_dim1=6000
    # fc_dim2=2112
    # fc_dim3=2112
    # fc_dim4=2112
    
    for rep in [1]:

        try:
            dropout=0.01
            model_str='gcn_vae_xa_e2_d1_dca'
            adj_decodeName=None #gala or None
            dataname = f'data_4mcn_design_20260124_rep{rep}_spa0_down0'
            plot_samples = {dataname: dataname}
            plot_sample_X=['logminmax']
            plotRecon='' #'meanRecon'
            standardizeX=False
            name='newModel'
            modelsavepath='/data2/xiaojiashun/niche/revision1/simu_code/sensitivity/gene_sparsity_res/sensitivity_data/staci/models/train_gae/'+dataname
            plotsavepath='/data2/xiaojiashun/niche/revision1/simu_code/sensitivity/gene_sparsity_res/sensitivity_data/staci/plots/train_gae/'+dataname
            datadir='/data2/xiaojiashun/niche/revision1/simu_code/sensitivity/gene_sparsity_data/sensitivity_data'
            
            # Set cuda and seed
            np.random.seed(seed)
            if use_cuda and (not torch.cuda.is_available()):
                print('cuda not available')
                use_cuda=False
            torch.manual_seed(seed)
            if use_cuda:
                torch.cuda.manual_seed(seed)
                torch.backends.cudnn.enabled = True
            
            #Load data
            sampleidx={dataname:dataname} #this is formated as {name of the sample as used in 'training_samples':name of the sample as stored in the metadata}
            savedir='/data2/xiaojiashun/niche/revision1/simu_code/sensitivity/gene_sparsity_res/sensitivity_data/staci'
            adj_dir=os.path.join(savedir,'a')
            
            #normalize the gene expression or load the normalized gene expression from Hu et al.
            #batch information should be stored in the metadata as 'sample'
            featureslist={}
            if plot_sample_X[0] in ['corrected','scaled']:
                scaleddata=scanpy.read_h5ad(datadir+f'/{dataname}-scaled.h5ad') #change to the h5ad file name of the input data
                
                for s in plot_sample_X.keys():
                    featureslist[s+'X_'+'corrected']=torch.tensor(scaleddata.layers['corrected'][scaleddata.obs['sample']==plot_samples[s]])
                    featureslist[s+'X_'+'scaled']=torch.tensor(scaleddata.layers['scaled'][scaleddata.obs['sample']==plot_samples[s]])
            
            else:
                scaleddata=scanpy.read_h5ad(datadir+f'/{dataname}.h5ad') #change to the h5ad file name of the input data
                
                for s in plot_samples.keys():
                    # scaleddata_train=scaleddata.X[scaleddata.obs['sample']==sampleidx[s]]
                    scaleddata_train=scaleddata.layers['counts']
            
                    if plot_sample_X[0]=='logminmax':
                        featurelog_train=np.log2(scaleddata_train+1/2)
                        scaler = MinMaxScaler()
                        featurelog_train_minmax=np.transpose(scaler.fit_transform(np.transpose(featurelog_train)))
                        featureslist[s+'X_'+plot_sample_X[0]]=torch.tensor(featurelog_train_minmax)
                    elif plot_sample_X[0]=='logminmax10':
                        featurelog_train=np.log2(scaleddata_train+1/2)
                        scaler = MinMaxScaler(feature_range=(0,10))
                        featurelog_train_minmax=np.transpose(scaler.fit_transform(np.transpose(featurelog_train)))
                        featureslist[s+'X_'+plot_sample_X[0]]=torch.tensor(featurelog_train_minmax)
            
            #load pre-computed adjacency matrices; adjust the file name as needed
            adj_list={}
            adj_list[dataname]=sp.load_npz(os.path.join(adj_dir,f'{dataname}_chunk0_{maskedgeName}.npz'))
            
            # load model
            num_nodes,num_features = list(featureslist.values())[0].shape
            model = gae.gae.model.GCNModelVAE_XA_e2_d1_DCA(num_features, hidden1,hidden2,fc_dim1, dropout)
            model.load_state_dict(torch.load(os.path.join(modelsavepath,str(plotepoch)+'.pt')))
            
            #compute embeddings
            mulist={}
            for s in plot_samples.keys():
                adj=adj_list[s]
                adj_norm = preprocessing.preprocess_graph(adj)
                adj_decode=None
                if adj_decodeName == 'gala':
                    adj_decode=preprocessing.preprocess_graph_sharp(adj)
                for xcorr in plot_sample_X:
                    samplename=s+'X_'+xcorr
                    features=featureslist[samplename]
                    if standardizeX:
                        features=torch.tensor(scale(features,axis=0, with_mean=True, with_std=True, copy=True))
                    if use_cuda:
                        model.cuda()
                        features = features.cuda().float()
                        adj_norm=adj_norm.cuda()
                        if adj_decodeName:
                            adj_decode=adj_decode.cuda()
                    
                    model.eval()
                    if adj_decodeName==None:
                        adj_recon,mu,logvar,z, features_recon = model(features, adj_norm)
                    else:
                        adj_recon,mu,logvar,z, features_recon = model(features, adj_norm,adj_decode)
                    if inverseAct=='leakyRelu':
                        muplot=inverseLeakyRelu(mu.cpu().detach().numpy())
                    else:
                        muplot=mu.cpu().detach().numpy()
                    if plotRecon:
                        if plotRecon=='meanRecon':
                            mulist[samplename]=features_recon[3].cpu().detach().numpy()
                    else:
                        mulist[samplename]=muplot
            
            xcorr = plot_sample_X[0]
            s = list(plot_samples.keys())[0]
            latents=None
            celltype_broad=None
            celltype_sub=None
            region=None
            samplenameList=None
            sobj_coord_np=None
            sampleidx=plot_samples[s]        
            samplename=s+'X_'+xcorr
            muplot=np.copy(mulist[samplename])
            
            latents=muplot
            sobj_coord_np=scaleddata.obs[['x','y']].to_numpy()
            samplenameList=np.repeat(s,muplot.shape[0])
            
            npc_plot=2
            reducer = umap.UMAP(n_neighbors=n_neighbors,min_dist=min_dist,random_state=seed)
            embedding = reducer.fit_transform(latents)
            savenameAdd='_nn'+str(n_neighbors)+'mdist0'+str(int(min_dist*100))+'epoch'+str(plotepoch)
    
            ## clustertarget
            res = 0.15
            target_ids = np.where(scaleddata.obs['cell_type']=='A')[0]
            latents_target = latents[target_ids,:]
            clusterRes_target=clusterLeiden_single(latents_target,n_neighbors,n_pcs,min_dist,res,randseed=seed)
            cluster_df = scaleddata.obs.loc[scaleddata.obs['cell_type']=='A'].copy()
            cluster_df['domain'] = clusterRes_target
            n_domains = len(np.unique(cluster_df['domain']))
        
            if n_domains > 6:
                domain_map = {'0': '0', '1': '1', '2': '2', '3': '3', '4': '4'}
                for d_ in cluster_df['domain'].unique():
                    if int(d_) not in [0, 1, 2, 3, 4]:
                        domain_map[d_] = '5'
                cluster_df['domain'] = cluster_df['domain'].map(domain_map)
            print(f"rep{rep}: {n_domains} -> {len(np.unique(cluster_df['domain']))} domain.")
            
            cluster_df.to_csv(f'{savedir}/{dataname}_clustertarget_epoch{plotepoch}_res{res}.csv', sep='\t')
    
            # ## clusterall
            # res = 0.1
            # clusterRes=clusterLeiden_single(latents,n_neighbors,n_pcs,min_dist,res,randseed=seed)
            # cluster_df = scaleddata.obs.copy()
            # cluster_df['domain'] = clusterRes
            # cluster_df['domain'].value_counts()
            # n_domains = len(np.unique(cluster_df['domain']))
        
            # if n_domains > 6:
            #     domain_map = {'0': '0', '1': '1', '2': '2', '3': '3', '4': '4'}
            #     for d_ in cluster_df['domain'].unique():
            #         if int(d_) not in [0, 1, 2, 3, 4]:
            #             domain_map[d_] = '5'
            #     cluster_df['domain'] = cluster_df['domain'].map(domain_map)
            # print(f"rep{rep}: {n_domains} -> {len(np.unique(cluster_df['domain']))} domain.")
            
            # cluster_df.to_csv(f'{savedir}/{dataname}_epoch{plotepoch}_res{res}.csv', sep='\t')
        except Exception as e:
            print(f'Error: {e}')