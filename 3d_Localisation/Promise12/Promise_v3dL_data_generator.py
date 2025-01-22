# -*- coding: utf-8 -*-
"""
Created on Sat Feb  3 12:49:49 2024

@author: Manna
"""


import os
from torch.utils.data import Dataset
import torch
import numpy as np
import SimpleITK as sitk
from monai import transforms
import random

class CustomImageDataset(Dataset):
    def __init__(self, df, data_path,train): 
        self.df = df
        self.data_path = data_path
        self.vol_list = self.df['vol_name'].tolist()
        self.Rmin_list  = self.df['rmin'].tolist()
        self.Rmax_list = self.df['rmax'].tolist()
        self.Cmin_list  = self.df['cmin'].tolist()
        self.Cmax_list  = self.df['cmax'].tolist()
        self.Zmin_list  = self.df['zmin'].tolist()
        self.Zmax_list  = self.df['zmax'].tolist()
        self.train = train

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = os.path.join(self.data_path, self.vol_list[idx]+'_R.mhd')
        Rmin = self.Rmin_list[idx] 
        Rmax = self.Rmax_list[idx]
        Cmin = self.Cmin_list[idx]
        Cmax = self.Cmax_list[idx]
        Zmin = self.Zmin_list[idx]
        Zmax = self.Zmax_list[idx]
        
        
        #volume
        vol_sitk = sitk.ReadImage(img_path)
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
            aug_trans = transforms.AdjustContrast(0.2)
            vol_aug_t = aug_trans(vol)
            vol_aug = vol_aug_t.numpy()
        elif no==5:
            aug_trans = transforms.AdjustContrast(0.3)
            vol_aug_t = aug_trans(vol)
            vol_aug = vol_aug_t.numpy()
        elif no==6:
            aug_trans = transforms.AdjustContrast(0.4)
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
        vol_norm = (vol_aug-np.min(vol_aug))/(np.max(vol_aug) - np.min(vol_aug))
        nor_vol_pad = np.zeros((64,256,256))
        nor_vol_pad[:n,:,:] = vol_norm
        nor_vol_pad = nor_vol_pad.astype(np.float32)

        nor_vol_pad_t = torch.from_numpy(nor_vol_pad)        
        nor_vol_pad_t1 = torch.unsqueeze(nor_vol_pad_t, 0)

        #label
        lab_Rmin = Rmin/64
        label_Rmin = torch.tensor(lab_Rmin)
        
        lab_Rmax = Rmax/64
        label_Rmax = torch.tensor(lab_Rmax)
        
        lab_Cmin = Cmin/256
        label_Cmin = torch.tensor(lab_Cmin)
        
        lab_Cmax = Cmax/256
        label_Cmax = torch.tensor(lab_Cmax)
        
        lab_Zmin = Zmin/256
        label_Zmin = torch.tensor(lab_Zmin)
        
        lab_Zmax = Zmax/256
        label_Zmax = torch.tensor(lab_Zmax)
        
        return nor_vol_pad_t1, label_Rmin, label_Rmax, label_Cmin, label_Cmax, label_Zmin, label_Zmax

