import sys
import time
import os
import logging
import argparse
from datetime import datetime

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

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, help='directory of simulated data', required=True)
    parser.add_argument('--data-name', type=str, help='name of simulated data', required=True)
    parser.add_argument('--out-dir', type=str, help='directory for saving results', required=True)
    args = parser.parse_args()

    ## Settings
    os.environ["CUDA_VISIBLE_DEVICES"] = "0" #this should be set to the GPU device you would like to use on your machine
    use_cuda=True #set to true if GPU is used 
    fastmode=False #Perform validation during training pass
    seed=3 #random seed
    useSavedMaskedEdges=False #some edges of the adjacency matrices are held-out for validation; set to True to save and use saved version of the edge masks
    maskedgeName='knn20_connectivity'
    epochs=4001 #number of training epochs
    saveFreq=100 #the model parameters will be saved during training at a frequency defined by this parameter
    lr=0.001 #initial learning rate
    lr_adv=0.001 #this is ignored if not using an adversarial loss in the latent space (i.e. it is ignored for the default setup of STACI. If a discriminator is trained to use the adversarial loss, this is the learning rate of the discriminator.)
    weight_decay=0 #regularization term

    hidden1=6000 #Number of units in hidden layer 1
    hidden2=6000 #Number of units in hidden layer 2
    # hidden3=2048 # dimensions of additional hidden layers in the encoder, if more layers are specified
    # hidden4=2048
    # hidden5=128
    fc_dim1=6000 #Number of units in the fully connected layer of the decoder
    # fc_dim2=128 # dimensions of additional hidden layers in the decoder, if more layers are specified
    # fc_dim3=128
    # fc_dim4=128
    adv_hidden=128 #ignored if not using an adversarial loss in the latent space. This is the hidden units of the discriminator.

    dropout=0.01 #neural network dropout term
    testNodes=0.1 #fraction of total cells used for testing
    valNodes=0.05 #fraction of total cells used for validation
    XreconWeight=20  #reconstruction weight of the gene expression
    advWeight=2 # weight of the adversarial loss, if used
    model_str='gcn_vae_xa_e2_d1_dca' #specify which model to use (see definition below): 'gcn_vae_xa_e2_d1_dca' is the default full STACI model, 'fc1_dca' is the version without using cell location
    adv=None  # different choices of the adversarial loss, if used (as defined below): 'clf_fc1_eq', 'clf_fc1_control_eq', 'clf_fc1_control', 'clf_fc1'
    ridgeL=0.01 #regularization weight of the gene dropout parameter
    shareGenePi=True #ignored in the default model; This is a parameter to specify how if the gene dropout term is shared for some variants of the ZINB distribution modeling as discussed in the original deep count autoencoder paper.

    num_features=500 #number of input genes
    dataname = args.data_name
    training_samples=[dataname] #names of the input samples used for training
    targetBatch=None #if adversarial loss is used, one possibility is to make all batches look like one target batch. None, if not using this option.
    training_sample_X='logminmax' #specify the normalization method for the gene expression input. 'logminmax' is the default that log transforms and min-max scales the expression. 'corrected' uses the z-score normalized and ComBat corrected data from Hu et al. 'scaled' uses the same normalization as 'corrected'.
    switchFreq=10 #the number of epochs spent on training the model using one sample, before switching to the next sample
    standardizeX=False #if perform additional z-score normalization of genes. Default is False.
    name='newModel' #name of the model
    useA=True #set to True to include adjacency loss as in the full STACI model

    #provide the paths to save the training log, trained models, and plots, and the path to the directory where the data is stored
    logsavepath=f'{args.out_dir}/log/train_gae/'+dataname
    modelsavepath=f'{args.out_dir}/models/train_gae/'+dataname
    plotsavepath=f'{args.out_dir}/plots/train_gae/'+dataname
    datadir=args.data_dir

    #Load data
    sampleidx={dataname:dataname} #this is formated as {name of the sample as used in 'training_samples':name of the sample as stored in the metadata}
    savedir=args.out_dir #where pre-computed adjacency matrices are stored
    adj_dir=os.path.join(savedir,'a')

    #normalize the gene expression or load the normalized gene expression from Hu et al.
    #batch information should be stored in the metadata as 'sample'
    featureslist={}
    if training_sample_X in ['corrected','scaled']:
        scaleddata=scanpy.read_h5ad(datadir+f'/{dataname}-scaled.h5ad') #change to the h5ad file name of the input data
        
        for s in sampleidx.keys():
            featureslist[s+'X_'+'corrected']=torch.tensor(scaleddata.layers['corrected'][scaleddata.obs['sample']==sampleidx[s]])
            featureslist[s+'X_'+'scaled']=torch.tensor(scaleddata.layers['scaled'][scaleddata.obs['sample']==sampleidx[s]])

    else:
        scaleddata=scanpy.read_h5ad(datadir+f'/{dataname}.h5ad') #change to the h5ad file name of the input data
        
        for s in sampleidx.keys():
            # scaleddata_train=scaleddata.X[scaleddata.obs['sample']==sampleidx[s]]
            scaleddata_train=scaleddata.layers['counts']

            if training_sample_X=='logminmax':
                featurelog_train=np.log2(scaleddata_train+1/2)
                scaler = MinMaxScaler()
                featurelog_train_minmax=np.transpose(scaler.fit_transform(np.transpose(featurelog_train)))
                featureslist[s+'X_'+training_sample_X]=torch.tensor(featurelog_train_minmax)
            elif training_sample_X=='logminmax10':
                featurelog_train=np.log2(scaleddata_train+1/2)
                scaler = MinMaxScaler(feature_range=(0,10))
                featurelog_train_minmax=np.transpose(scaler.fit_transform(np.transpose(featurelog_train)))
                featureslist[s+'X_'+training_sample_X]=torch.tensor(featurelog_train_minmax)


    #load pre-computed adjacency matrices; adjust the file name as needed
    adj_list={}
    adj_list[dataname]=sp.load_npz(os.path.join(adj_dir,f'{dataname}_{maskedgeName}.npz'))

    adjnormlist={}
    pos_weightlist={}
    normlist={}
    for ai in adj_list.keys():
        adjnormlist[ai]=preprocessing.preprocess_graph(adj_list[ai])
        
        pos_weightlist[ai] = torch.tensor(float(adj_list[ai].shape[0] * adj_list[ai].shape[0] - adj_list[ai].sum()) / adj_list[ai].sum()) #using full unmasked adj
        normlist[ai] = adj_list[ai].shape[0] * adj_list[ai].shape[0] / float((adj_list[ai].shape[0] * adj_list[ai].shape[0] - adj_list[ai].sum()) * 2)
        
        adj_label=adj_list[ai] + sp.eye(adj_list[ai].shape[0])
        adj_list[ai]=torch.tensor(adj_label.todense())
        

    if 'dca' in model_str:
        rawdata=scanpy.read_h5ad(datadir+f'/{dataname}.h5ad')
        features_raw_list={}
        for s in sampleidx.keys():
            features_raw_list[s+'X_'+'raw']=torch.tensor(rawdata.layers['counts'])

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

    # loop over all train/validation sets
    np.random.seed(seed)
    torch.manual_seed(seed)
    if use_cuda:
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.enabled = True
        
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

    log_file = os.path.join(
        logsavepath, f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    )
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()  # 同时打印到终端
        ]
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
        
        
        loss=loss_kl_train+loss_x_train #for lossXreconOnly_wKL only
        if useA:
            loss=loss+loss_a_train
    #     loss = loss_kl_train+loss_a_train #for lossAreconOnly_wKL only

        loss.backward()
        optimizerVAEXA.step()

        if not fastmode:
            # Evaluate validation set performance separately,
            # deactivates dropout during validation run & no variation in z.
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
    #     loss_val=loss_a_val

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
        
    # print('cross-validation ',seti)
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

    for ep in range(epochs):
        t=int(ep/switchFreq)%len(training_samples)
        training_samples_t=training_samples[t]

        adj_norm=adjnormlist[training_samples_t].cuda().float()
        adj_label=adj_list[training_samples_t].cuda().float()
        features=featureslist[training_samples_t+'X_'+training_sample_X].cuda().float()
        pos_weight=pos_weightlist[training_samples_t]
        norm=normlist[training_samples_t]
        
        if 'dca' in model_str:
            features_raw=features_raw_list[training_samples_t+'X_raw'].cuda()
        num_nodes,_ = features.shape
        
        maskpath=os.path.join(savedir,'trainMask',training_samples_t+'_'+maskedgeName+'_seed'+str(seed)+'.pkl')
        if useSavedMaskedEdges and os.path.exists(maskpath):
            with open(maskpath, 'rb') as input:
                maskedgeres = pickle.load(input)
        else:
            # construct training, validation, and test sets
            maskedgeres= preprocessing.mask_nodes_edges(features.shape[0],testNodeSize=testNodes,valNodeSize=valNodes,seed=seed)
            with open(maskpath, 'wb') as output:
                pickle.dump(maskedgeres, output, pickle.HIGHEST_PROTOCOL)
        train_nodes_idx,val_nodes_idx,test_nodes_idx = maskedgeres
        if use_cuda:
            train_nodes_idx=train_nodes_idx.cuda()
            val_nodes_idx=val_nodes_idx.cuda()
            test_nodes_idx=test_nodes_idx.cuda()
        
        train_loss_eps[ep],train_loss_kl_eps[ep],train_loss_x_eps[ep],train_loss_a_eps[ep],val_loss_eps[ep],val_loss_x_eps[ep],val_loss_a_eps[ep]=train(ep)

            
        if ep%saveFreq == 0:
            torch.save(model.cpu().state_dict(), os.path.join(modelsavepath,str(ep)+'.pt'))
        if use_cuda:
            model.cuda()
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
