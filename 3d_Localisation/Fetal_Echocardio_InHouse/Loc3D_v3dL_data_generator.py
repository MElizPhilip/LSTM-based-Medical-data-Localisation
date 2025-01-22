# -*- coding: utf-8 -*-
"""
Created on Sat Feb  3 12:49:49 2024

@author: MElizPhilip
"""


import os
from torch.utils.data import Dataset
import torch
import numpy as np
import SimpleITK as sitk
from monai import transforms
import random

class CustomImageDataset(Dataset):
    def __init__(self, df, data_path,data_Sz, transform, swapTF,train): 
        self.df = df
        self.data_path = data_path
        self.data_Sz = data_Sz
        self.vol_list = self.df['vol_name'].tolist()
        self.HS_list  = self.df['Heart_St5'].tolist()
        self.HE_list = self.df['Heart_End5'].tolist()
        self.Xmin_list  = self.df['xmin'].tolist()
        self.Xmax_list  = self.df['xmax'].tolist()
        self.Ymin_list  = self.df['ymin'].tolist()
        self.Ymax_list  = self.df['ymax'].tolist()
        self.transform = transform
        self.swapTF = swapTF
        self.train = train

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = os.path.join(self.data_path, self.vol_list[idx])
        Hst = self.HS_list[idx]-1 # -1 to compensate for index starting from 0
        Hend = self.HE_list[idx]-1
        Xmin = self.Xmin_list[idx]-1
        Xmax = self.Xmax_list[idx]-1
        Ymin = self.Ymin_list[idx]-1
        Ymax = self.Ymax_list[idx]-1
        
        
        #volume
        vol_sitk = sitk.ReadImage(img_path, sitk.sitkUInt8)
        if self.swapTF:
            vol = sitk.GetArrayFromImage(vol_sitk).swapaxes(0, 1) # to make first axes the coronal(4cv) slices
        else:
            vol = sitk.GetArrayFromImage(vol_sitk)
        
        #augmentation
        if self.train:
            no = random.randint(0,9)
        else:
            no=0
        
        if no==0: 
            vol_aug = vol
        elif no==1:
            aug_trans = transforms.AdjustContrast(0.5)
            vol_aug_t = aug_trans(vol)
            vol_aug = vol_aug_t.numpy()
        elif no==2:
            aug_trans = transforms.AdjustContrast(0.6)
            vol_aug_t = aug_trans(vol)
            vol_aug = vol_aug_t.numpy()
        elif no==3:
            aug_trans = transforms.AdjustContrast(0.75)
            vol_aug_t = aug_trans(vol)
            vol_aug = vol_aug_t.numpy()
        elif no==4:
            aug_trans = transforms.ShiftIntensity(2, safe=True)
            vol_aug_t = aug_trans(vol)
            vol_aug = vol_aug_t.numpy()
        elif no==5:
            aug_trans = transforms.ShiftIntensity(4, safe=True)
            vol_aug_t = aug_trans(vol)
            vol_aug = vol_aug_t.numpy()
        elif no==6:
            aug_trans = transforms.ShiftIntensity(2, safe=True)
            vol_aug_t = aug_trans(vol)
            vol_aug = vol_aug_t.numpy()
        elif no==7:
            aug_trans = transforms.RandGaussianNoise(prob=1, mean=0.0, std=2)
            vol_aug_t = aug_trans(vol)
            vol_aug = vol_aug_t.numpy()
        elif no==8:
            aug_trans = transforms.RandGaussianNoise(prob=1, mean=0.0, std=4)
            vol_aug_t = aug_trans(vol)
            vol_aug = vol_aug_t.numpy()
        elif no==9:
            aug_trans = transforms.RandGaussianNoise(prob=1, mean=0.0, std=5)
            vol_aug_t = aug_trans(vol)
            vol_aug = vol_aug_t.numpy()
            # pad zeros
        n,X,Y = vol_aug.shape
        nor_vol_pad = np.zeros((self.data_Sz,self.data_Sz,self.data_Sz),dtype=np.uint8)
        nor_vol_pad[:n,:X,:Y] = vol_aug
        
        if self.transform:
            nor_vol_pad = self.transform(nor_vol_pad)
        nor_vol_pad = torch.unsqueeze(nor_vol_pad, 0)
        
        #label
        lab_calST = Hst/192
        label_ST = torch.tensor(lab_calST)
        
        lab_calEND = Hend/192
        label_END = torch.tensor(lab_calEND)
        
        Xlab_calST = Xmin/192
        Xlabel_ST = torch.tensor(Xlab_calST)
        
        Xlab_calEND = Xmax/192
        Xlabel_END = torch.tensor(Xlab_calEND)
        
        Ylab_calST = Ymin/192
        Ylabel_ST = torch.tensor(Ylab_calST)
        
        Ylab_calEND = Ymax/192
        Ylabel_END = torch.tensor(Ylab_calEND)
        
        return nor_vol_pad, label_ST, label_END, Xlabel_ST, Xlabel_END, Ylabel_ST, Ylabel_END

