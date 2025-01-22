# -*- coding: utf-8 -*-
"""
Created on Thu Feb  1 15:13:03 2024

@author: MElizPhilip
"""

import argparse
from datetime import datetime
import os
import torch
import torch.nn as nn
import numpy as np
from timeit import default_timer as timer
import pandas as pd
from torch.optim import lr_scheduler
from monai import transforms

from Loc3D_v3dL_data_generator import CustomImageDataset
from Loc3D_v3dL_network import inHouse_3D_LOC


def get_scheduler(optimizer, opt):
    if opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1,verbose =False)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5,verbose =False)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler

if __name__ == '__main__':
    # CUDA for PyTorch
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    torch.backends.cudnn.benchmark = True
    start_time = datetime.now()
    #.................Argument Parser
    Arg_parser = argparse.ArgumentParser()
    Arg_parser.add_argument('-b','--batch',help='batch size',type = int,default = 1)
    Arg_parser.add_argument('-s','--data_size',help='Size of data',type = int,default = 192)
    Arg_parser.add_argument('-k','--kSz',help='Kernel size of Convolution filters.',type = int,default = 3)
    Arg_parser.add_argument('-f','--folds',help='Cross Validation number of folds.',type = int,default = 6)
    Arg_parser.add_argument('-p','--preWeight',help='Use pre-trained weights or NOT',type = bool,default = False)
    Arg_parser.add_argument('-r','--preWeightName',help='Which pre-trained weight to use',default = 'weights/save_49.pt')
    Arg_parser.add_argument('-d','--data_path',help='Path to data (location of data)',default = '/path/to/the/DATA')
    Arg_parser.add_argument('-a','--swap_axis',help='Swap 3d vol axes to make coronal axis first ',action='store_true')
    Arg_parser.add_argument('--lr_policy', type=str, default='step', help='learning rate policy: lambda|step|plateau')
    Arg_parser.add_argument('--lr_decay_iters', type=int, default=10, help='multiply by a gamma every lr_decay_iters iterations')
    Arg_parser.add_argument('--lr', type=float, default=0.0001, help='initial learning rate for adam')
    Arg_parser.add_argument('--epoch', type=int, default=50, help='how many epochs to run.')
    Arg_parser.add_argument('codeName', type=str, help='Specific name to save weight.')
    
    args = Arg_parser.parse_args()
    
    init_type = 'normal'
    learning_rate = args.lr
    
    loss = nn.MSELoss()
    
    train_transform = transforms.Compose([
        transforms.ScaleIntensityRange(0, 255,0, 1), # ToTensor : [0, 255] -> [0, 1]
                             ])
    train = True
    if not os.path.isdir("weights"):
      os.mkdir("weights")
    
    pre_weight_path = os.path.join("weights",args.preWeightName)
    ckpt_dir = os.path.join("weights",args.codeName)
    
    for fold in range(1,args.folds+1):
        min_train_loss = np.inf # for model check pointing
        csv_META_file = 'folds_csv/loc_F'+str(fold)+'_train_AUG.csv'
        last_epoch = 0

        df = pd.read_csv(csv_META_file)
        print('.............................FOLD ',fold)
        print('csv name: ',csv_META_file)
        print('Size of Training set: ',len(df))
        print('Swap axes:', args.swap_axis)
        
        #............Data generator initialization
        train_generator = CustomImageDataset(df, args.data_path,args.data_size, train_transform,args.swap_axis,train)
        
        #............Data Loader Initialization
        train_dLoader = torch.utils.data.DataLoader(train_generator, batch_size=args.batch, shuffle=True, num_workers=8) 
    
        #......initialize the model and weights
        inHouse_3D_model = inHouse_3D_LOC(args.data_size, args.kSz)
        
        if torch.cuda.is_available():
            inHouse_3D_model.cuda()
        
        if args.preWeight == True:
            inHouse_3D_model.load_state_dict(torch.load(args.preWeightName))
            print('model loaded!!!!!........................... ')
        
        #.........Optimizer and Lr Scheduler............
        optimizer = torch.optim.Adam(inHouse_3D_model.parameters(), lr=learning_rate, foreach=False)
        scheduler = get_scheduler(optimizer, args)
        print('Training started...................\n')
        for epoch in range(args.epoch):
            
            start_time = timer()
            # Reset metrics
            epoch_iter = 0
            train_loss = 0.0
            val_loss = 0.0
            inHouse_3D_model.train()
            for vol, GT_st, GT_end, XGT_st, XGT_end, YGT_st, YGT_end in train_dLoader:
                # use GPU if available
                inHouse_3D_model.zero_grad()
                torch.cuda.empty_cache()
                
                vol = vol.to(device)
                GT_st = GT_st.to(device)
                GT_end = GT_end.to(device)
                XGT_st = XGT_st.to(device)
                XGT_end = XGT_end.to(device)
                YGT_st = YGT_st.to(device)
                YGT_end = YGT_end.to(device)
                
                # Training steps
                pred_ST, pred_END,Xpred_ST, Xpred_END,Ypred_ST, Ypred_END = inHouse_3D_model(vol)
                lossST = loss(pred_ST,GT_st)
                lossEND = loss(pred_END,GT_end)
                XlossST = loss(Xpred_ST,XGT_st)
                XlossEND = loss(Xpred_END,XGT_end)
                YlossST = loss(Ypred_ST,YGT_st)
                YlossEND = loss(Ypred_END,YGT_end)
                lossFn = lossST + lossEND + XlossST + XlossEND + YlossST + YlossEND
                
                lossFn.backward() # backward pass: compute gradient of the loss wrt model parameters
                optimizer.step() # update parameters
                train_loss += lossFn.item() # update training loss 
            
            scheduler.step()
            print(f'Epoch: {epoch} \tTraining Loss: {train_loss:.{5}} ')
            # checkpoint if improved
            if min_train_loss>train_loss:
                state_dict = inHouse_3D_model.state_dict()
                torch.save(state_dict, ckpt_dir+'_fold'+str(fold)+'.pt')
                last_epoch = epoch
                min_train_loss = train_loss
                print('Model saved..!')
        print('Last epoch saved : ',last_epoch,ckpt_dir+'_fold'+str(fold)+'.pt')