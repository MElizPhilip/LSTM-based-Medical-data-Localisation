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

from BraTS_v3dL_data_generator import CustomImageDataset
from BraTS_v3dL_network import BraTS_3D_LOC

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
def compute_ap(recall, precision,method):
    #https://github.com/JDSobek/MedYOLO/blob/main/utils/metrics.py
    """ Compute the average precision, given the recall and precision curves
    # Arguments
        recall:    The recall curve (list)
        precision: The precision curve (list)
    # Returns
        Average precision, precision curve, recall curve
    """

    # Append sentinel values to beginning and end
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))

    # Compute the precision envelope
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))

    # Integrate area under curve
    # methods: 'continuous', 'interp'
    if method == 'interp':
        x = np.linspace(0, 1, 101)  # 101-point interp (COCO)
        ap = np.trapz(np.interp(x, mrec, mpre), x)  # integrate
    else:  # 'continuous'
        i = np.where(mrec[1:] != mrec[:-1])[0]  # points where x axis (recall) changes
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])  # area under curve

    return ap

def compute_iou3D(gt_box,pred_box):
    #https://www.v7labs.com/blog/intersection-over-union-guide#:~:text=7.,to%20determine%20the%20IoU%20score.
    inter_w = min(gt_box["xmax"],pred_box["xmax"]) - max(gt_box["xmin"],pred_box["xmin"])
    inter_h = min(gt_box["ymax"],pred_box["ymax"]) - max(gt_box["ymin"],pred_box["ymin"])
    inter_d = min(gt_box["cmax"],pred_box["cmax"]) - max(gt_box["cmin"],pred_box["cmin"])
    
    if inter_d<=0 or inter_h <=0 or inter_w <=0:
        return 0
    inter_area = inter_d * inter_h * inter_w
    
    #union area
    gt_area = (gt_box["xmax"] - gt_box["xmin"]) * (gt_box["ymax"] - gt_box["ymin"]) * (gt_box["cmax"] - gt_box["cmin"])
    pred_area = (pred_box["xmax"] - pred_box["xmin"]) * (pred_box["ymax"] - pred_box["ymin"]) * (pred_box["cmax"] - pred_box["cmin"])
    
    union_area = gt_area + pred_area - inter_area
    
    IoU = inter_area/union_area
    return IoU

if __name__ == '__main__':
    # CUDA for PyTorch
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    torch.backends.cudnn.benchmark = True
    start_time = datetime.now()
    #.................Argument Parser
    Arg_parser = argparse.ArgumentParser()
    Arg_parser.add_argument('-b','--batch',help='batch size',type = int,default = 1)
    Arg_parser.add_argument('-d','--data_path',help='Path to data (location of data)',default = '/path/to/the/DATA/BraTS')
    Arg_parser.add_argument('-f','--NFold',help='Number of Folds for cross validation.',type = int,default = 6)
    Arg_parser.add_argument('-k','--kSz',help='Kernel size of Convolution filters.',type = int,default = 3)
    Arg_parser.add_argument('codeName', type=str, help='Specific name that was used to save the weight during training.')
    
    args = Arg_parser.parse_args()
    
    test_transform = transforms.Compose([
        transforms.ScaleIntensityRange(0, 255,0, 1), # ToTensor : [0, 255] -> [0, 1]
        ])
    train = False
    data_sz = [160, 224, 224]
    
    #.............thresholds for computing the average precision..........
    thresholds = np.arange(start=0.5, stop=0.9, step=0.05)
    thresholds2 = np.arange(start=0.5, stop=0.6, step=0.1)
    
    for fold in range(1,args.NFold+1):
        csv_META_file = 'folds_csv/brats_f'+str(fold)+'_test.csv'
        weight_path = os.path.join("weights",args.codeName+'_fold'+str(fold)+'.pt')
        
        print('.............................FOLD ',fold)
        print(csv_META_file,'\n',weight_path)
        df = pd.read_csv(csv_META_file)
        test_generator = CustomImageDataset(df, args.data_path,train)
        
        test_dLoader = torch.utils.data.DataLoader(test_generator, batch_size=args.batch, shuffle=False, num_workers=8) 
            
        BraTS_3D_model = BraTS_3D_LOC(args.kSz)
       
        if torch.cuda.is_available():
            BraTS_3D_model.cuda()
        BraTS_3D_model.load_state_dict(torch.load(weight_path))
        print('model loaded!!!!!........................... ')
        
        avJ = 0 # Average Jaccard Index
        ctoc_dist = np.zeros((len(df))) # centroid to centroid distance
        #...............
        ap_pred = np.zeros(len(test_dLoader))
        ap_gt = np.ones(len(test_dLoader))
        
        cnt_vol = 0 #counter
        
        # dataframe to save the results to....................
        df_save = pd.DataFrame(columns=['GT_st','GT_end','GT_Xst','GT_Xend','GT_Yst','GT_Yend','PRED_st','PRED_end','PRED_Xst','PRED_Xend','PRED_Yst','PRED_Yend','iou'])
        
        BraTS_3D_model.eval()
        for VAL_vol, VAL_GT_st, VAL_GT_end, VAL_GT_Xst, VAL_GT_Xend, VAL_GT_Yst, VAL_GT_Yend in test_dLoader:
            VAL_vol = VAL_vol.to(device)
            VAL_GT_st = VAL_GT_st.to(device)
            VAL_GT_end = VAL_GT_end.to(device)
            VAL_GT_Xst = VAL_GT_Xst.to(device)
            VAL_GT_Xend = VAL_GT_Xend.to(device)
            VAL_GT_Yst = VAL_GT_Yst.to(device)
            VAL_GT_Yend = VAL_GT_Yend.to(device)
            
            with torch.no_grad():
                torch.cuda.empty_cache()
                VAL_pred_st, VAL_pred_end, VAL_pred_Xst, VAL_pred_Xend, VAL_pred_Yst, VAL_pred_Yend = BraTS_3D_model(VAL_vol)
            pred_st = int((VAL_pred_st.item()) * data_sz[0])
            pred_end = int((VAL_pred_end.item())* data_sz[0])
            pred_Xst = int((VAL_pred_Xst.item()) * data_sz[1])
            pred_Xend = int((VAL_pred_Xend.item())* data_sz[1])
            pred_Yst = int((VAL_pred_Yst.item()) * data_sz[2])
            pred_Yend = int((VAL_pred_Yend.item())* data_sz[2])
            
            pred_mid = int((pred_st+pred_end)/2)
            pred_Xmid = int((pred_Xst+pred_Xend)/2)
            pred_Ymid = int((pred_Yst+pred_Yend)/2)
            
            gt_st = int((VAL_GT_st.item())* data_sz[0])
            gt_end = int((VAL_GT_end.item())* data_sz[0])
            gt_Xst = int((VAL_GT_Xst.item())* data_sz[1])
            gt_Xend = int((VAL_GT_Xend.item())* data_sz[1])
            gt_Yst = int((VAL_GT_Yst.item())* data_sz[2])
            gt_Yend = int((VAL_GT_Yend.item())* data_sz[2])
            
            gt_mid = int((gt_st+gt_end)/2)
            gt_Xmid = int((gt_Xst+gt_Xend)/2)
            gt_Ymid = int((gt_Yst+gt_Yend)/2)
            
            
            gt =dict(xmin=gt_Xst,ymin=gt_Yst,cmin = gt_st,xmax=gt_Xend,ymax=gt_Yend,cmax = gt_end)
            pred = dict(xmin=pred_Xst,ymin=pred_Yst,cmin=pred_st,xmax=pred_Xend,ymax=pred_Yend,cmax=pred_end)
            
            ap_pred[cnt_vol] = compute_iou3D(gt,pred)
            df_save.loc[cnt_vol] = [gt_st,gt_end,gt_Xst,gt_Xend,gt_Yst,gt_Yend,pred_st,pred_end,pred_Xst,pred_Xend,pred_Yst,pred_Yend,ap_pred[cnt_vol]]

            avJ += ap_pred[cnt_vol]
            #centroid to centroid distance
            Cent_gt = np.array([gt_Xmid,gt_mid,gt_Ymid])
            Cent_pred = np.array([pred_Xmid,pred_mid,pred_Ymid])
            ctoc_dist[cnt_vol] = np.linalg.norm(Cent_gt - Cent_pred) 
            cnt_vol += 1
            
        print('AVERAGE Jaccard index:',avJ/len(test_dLoader))
        # mean distance
        print('HEART -- Mean centroid to centroid distance:',np.sum(ctoc_dist)/len(df) )
        print('IoUs:',ap_pred)
        
        #......compute Average Precision @ range 0.5 - 0.9
        precisions, recalls = precision_recall_curve(y_true=ap_gt, 
                                              pred_scores=ap_pred, 
                                              thresholds=thresholds)
        print('medyolo_interp:',compute_ap(recalls, precisions,'interp'))
        print('medyolo_cont:',compute_ap(recalls, precisions,'continuous'))
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
        print('medyolo_interp5:',compute_ap(recalls, precisions,'interp'))
        print('medyolo_cont5:',compute_ap(recalls, precisions,'continuous'))
        precisions.append(1)
        recalls.append(0)
        precisions = np.array(precisions)
        recalls = np.array(recalls)
        AP5 = np.sum((recalls[:-1] - recalls[1:]) * precisions[:-1])
        print('AVERAGE Precision @ 0.5:',AP5)
        
        
        df_save.to_csv(os.path.join("weights","Result",args.codeName+'_fold'+str(fold)+'.csv'), encoding='utf-8', index=False)
        