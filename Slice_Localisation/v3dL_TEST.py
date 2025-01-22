# -*- coding: utf-8 -*-
"""
Created on Tue Feb  6 10:21:54 2024

@author: MElizPhilip
"""

import argparse
from datetime import datetime
import os
import torch
import numpy as np
import pandas as pd
from monai import transforms
from sklearn.metrics import average_precision_score,balanced_accuracy_score,jaccard_score
import sklearn

from v3dL_data_generator import CustomImageDataset
from v3dL_network import Slice_LOC

def precision_recall_curve(y_true, pred_scores, thresholds):
    precisions = []
    recalls = []

    for threshold in thresholds:
        y_pred = [1 if score >= threshold else 0 for score in pred_scores]

        precision = sklearn.metrics.precision_score(y_true=y_true, y_pred=y_pred, pos_label=1)
        recall = sklearn.metrics.recall_score(y_true=y_true, y_pred=y_pred, pos_label=1)

        precisions.append(precision)
        recalls.append(recall)

    return precisions, recalls

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
    Arg_parser.add_argument('-d','--data_path',help='Path to data (location of data)',default = '/path/to/the/DATA')
    Arg_parser.add_argument('-a','--swap_axis',help='Swap 3d vol axes to make coronal axis first ',action='store_true')
    Arg_parser.add_argument('-f','--NFold',help='Number of Folds for cross validation.',type = int,default = 6)
    Arg_parser.add_argument('-k','--kSz',help='Kernel size of Convolution filters.',type = int,default = 3)
    Arg_parser.add_argument('codeName', type=str, help='Specific name that was used to save the weight during training.')
    args = Arg_parser.parse_args()
    
    test_transform = transforms.Compose([
        transforms.ScaleIntensityRange(0, 255,0, 1), # ToTensor : [0, 255] -> [0, 1]
        ])
    
    train = False
    data_size = args.data_size
    corr = 1/data_size
    
    #.............thresholds for computingthe average precision..........
    thresholds = np.arange(start=0.5, stop=0.9, step=0.05)
    thresholds2 = np.arange(start=0.5, stop=0.6, step=0.1)
    
    
    for fold in range(1,args.NFold+1):
        csv_META_file = 'folds_csv/loc_F'+str(fold)+'_test_AUG.csv'
        weight_path = os.path.join("weights",args.codeName+'_fold'+str(fold)+'.pt')
        
        print('.............................FOLD ',fold)
        print(csv_META_file,'\n',weight_path)
        df = pd.read_csv(csv_META_file)
        test_generator = CustomImageDataset(df, args.data_path,args.data_size, test_transform,args.swap_axis,train)
        
        test_dLoader = torch.utils.data.DataLoader(test_generator, batch_size=args.batch, shuffle=False, num_workers=8) 
        
        slice_model = Slice_LOC(args.data_size,args.kSz)
        
        if torch.cuda.is_available():
            slice_model.cuda()
        slice_model.load_state_dict(torch.load(weight_path))
        print('model loaded!!!!!........................... ')
        
        #...metrics initialisation................
        bal_Acc = 0 # Balanced Accuracy
        avp = 0     # Average Precision
        avJ = 0     # Average Jaccard Index
        ctoc_dist = np.zeros((len(df))) # centroid to centroid distance
        #...............
        ap_pred = np.zeros(len(test_dLoader))
        ap_gt = np.ones(len(test_dLoader))
        
        cnt_vol = 0 # counter
        
        # dataframe to save the results to....................
        df_save = pd.DataFrame(columns=['GT_st','GT_end','PRED_st','PRED_end'])
        
        slice_model.eval()
        for VAL_vol, VAL_GT_st, VAL_GT_end in test_dLoader:
            pred = np.zeros((data_size))
            gt = np.zeros((data_size))
            VAL_vol = VAL_vol.to(device)
            VAL_GT_st = VAL_GT_st.to(device)
            VAL_GT_end = VAL_GT_end.to(device)
            
            with torch.no_grad():
                torch.cuda.empty_cache()
                VAL_pred_st, VAL_pred_end = slice_model(VAL_vol)
            pred_st = int((VAL_pred_st.item()+corr) *data_size)
            pred_end = int((VAL_pred_end.item()+corr)*data_size)
            pred_mid = int((pred_st+pred_end)/2)
            gt_st = int((VAL_GT_st.item()+corr)*data_size)
            gt_end = int((VAL_GT_end.item()+corr)*data_size)
            gt_mid = int((gt_st+gt_end)/2)
            
            df_save.loc[cnt_vol] = [gt_st,gt_end,pred_st,pred_end]
            
            pred[pred_st:pred_end] = 1
            gt[gt_st:gt_end] =1
            bal_Acc+=balanced_accuracy_score(gt, pred)
            avp += average_precision_score(gt,pred)
            avJ += jaccard_score(gt,pred)
            ap_pred[cnt_vol] = jaccard_score(gt,pred)
            
            #centroid to centroid distance
            ctoc_dist[cnt_vol] = np.abs(pred_mid-gt_mid) * 0.5 # 0.5 is MElizPhilipvoxel spacing
            cnt_vol += 1
        print('AVERAGE Balanced Accuracy:',bal_Acc/len(test_dLoader))
        print('AVERAGE Precision:',avp/len(test_dLoader))
        print('AVERAGE Jaccard index:',avJ/len(test_dLoader))
        # mean distance
        print('HEART -- Mean centroid to centroid distance:',np.sum(ctoc_dist)/len(df) )
        print('IoUs:',ap_pred)
        
        #......compute Average Precision @ range 0.5 - 0.9
        precisions, recalls = precision_recall_curve(y_true=ap_gt, 
                                              pred_scores=ap_pred, 
                                              thresholds=thresholds)
        precisions.append(1)
        recalls.append(0)
        precisions = np.array(precisions)
        recalls = np.array(recalls)
        AP = np.sum((recalls[:-1] - recalls[1:]) * precisions[:-1])
        print('AVERAGE Precision:',AP)
        
        #......compute Average Precision @ 0.5
        precisions, recalls = precision_recall_curve(y_true=ap_gt, 
                                              pred_scores=ap_pred, 
                                              thresholds=thresholds2)
        precisions.append(1)
        recalls.append(0)
        precisions = np.array(precisions)
        recalls = np.array(recalls)
        AP5 = np.sum((recalls[:-1] - recalls[1:]) * precisions[:-1])
        print('AVERAGE Precision @ 0.5:',AP5)
        
        #.........save the dataframe
        df_save.to_csv(os.path.join("weights","Result",args.codeName+'_fold'+str(fold)+'.csv'), encoding='utf-8', index=False)