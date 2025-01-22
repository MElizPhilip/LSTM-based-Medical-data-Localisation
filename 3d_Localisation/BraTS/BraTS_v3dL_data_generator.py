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
        img_path = os.path.join(self.data_path, self.vol_list[idx]+'_flairC.nii.gz')
        Rmin = self.Rmin_list[idx] 
        Rmax = self.Rmax_list[idx]
        Cmin = self.Cmin_list[idx]
        Cmax = self.Cmax_list[idx]
        Zmin = self.Zmin_list[idx]
        Zmax = self.Zmax_list[idx]
        
        
        #volume
        vol_sitk = sitk.ReadImage(img_path)
        vol = sitk.GetArrayFromImage(vol_sitk)
        vol_norm = (vol-np.min(vol))/(np.max(vol) - np.min(vol))
        vol_norm = vol_norm.astype(np.float32)

        vol_norm_t = torch.from_numpy(vol_norm)        
        nor_vol_pad = torch.unsqueeze(vol_norm_t, 0)

        #label
        lab_Rmin = Rmin/160
        label_Rmin = torch.tensor(lab_Rmin)
        
        lab_Rmax = Rmax/160
        label_Rmax = torch.tensor(lab_Rmax)
        
        lab_Cmin = Cmin/224
        label_Cmin = torch.tensor(lab_Cmin)
        
        lab_Cmax = Cmax/224
        label_Cmax = torch.tensor(lab_Cmax)
        
        lab_Zmin = Zmin/224
        label_Zmin = torch.tensor(lab_Zmin)
        
        lab_Zmax = Zmax/224
        label_Zmax = torch.tensor(lab_Zmax)
        
        return nor_vol_pad, label_Rmin, label_Rmax, label_Cmin, label_Cmax, label_Zmin, label_Zmax

