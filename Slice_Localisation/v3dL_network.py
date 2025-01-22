# -*- coding: utf-8 -*-
"""
Created on Mon Sep 16 12:00:06 2024

@author: MElizPhilip
"""
import torch
import torch.nn as nn

class Slice_LOC(nn.Module):
    # has only 4 blocks unlike the 5 previoulsy
    def __init__(self, data_sz, kernelSz,lstm_layers=1):
        super(Slice_LOC, self).__init__()
        self.num_layers = lstm_layers
        lstm_input_Sz = (data_sz//16) * (data_sz//16) * (data_sz//16)
        lstm_output_Sz = 144
        
        self.a_block1 = Conv3DBlock(1,24,kernelSz) # 192->96
        self.a_block2 = Conv3DBlock(24,48,kernelSz) # 96->48
        self.a_block3 = Conv3DBlock(48,96,kernelSz) # 48->24
        self.a_block4 = Conv3DBlock(96,data_sz,kernelSz) # 24->12

        self.lstmST = nn.LSTM(lstm_input_Sz,lstm_output_Sz,self.num_layers,batch_first=True)
        self.lstmEND = nn.LSTM(lstm_input_Sz,lstm_output_Sz,self.num_layers,batch_first=True)
        self.fcST = nn.Linear(lstm_output_Sz, 1)
        self.fcEND = nn.Linear(lstm_output_Sz, 1)
        
        self.attenST = CBAM(data_sz)
        self.attenEND = CBAM(data_sz)
    
    
    def forward(self, vol3D):
        fe1 = self.a_block1(vol3D)
        fe2 = self.a_block2(fe1)
        fe3 = self.a_block3(fe2)
        fe4 = self.a_block4(fe3)
        
        attnST = self.attenST(fe4)
        attn_flatST = torch.flatten(attnST,start_dim=2)
        _,(hn1,_) = self.lstmST(attn_flatST)
        predST  = self.fcST(hn1.squeeze(0))
        predST = predST.squeeze(1)
        
        attnEND = self.attenEND(fe4)
        attn_flatEND = torch.flatten(attnEND,start_dim=2)
        _,(hn2,_) = self.lstmEND(attn_flatEND)
        predEND  = self.fcEND(hn2.squeeze(0))
        predEND = predEND.squeeze(1)
        return predST, predEND    
    
    

class Conv3DBlock(nn.Module):
    """
    The basic block for double 3x3x3 convolutions in the analysis path
    -- __init__()
    :param in_channels -> number of input channels
    :param out_channels -> desired number of output channels
    -- forward()
    :param input -> input Tensor to be convolved
    :return -> Tensor
    Adapted from: https://github.com/AghdamAmir/3D-UNet/blob/main/unet3d.py    
    """
    
    def __init__(self, in_channels, out_channels,kSz) -> None:
        super(Conv3DBlock, self).__init__()
        self.conv1 = nn.Conv3d(in_channels= in_channels, out_channels=out_channels//2, kernel_size=kSz, padding=(kSz-1)//2)
        self.bn1 = nn.BatchNorm3d(num_features=out_channels//2)
        self.conv2 = nn.Conv3d(in_channels= out_channels//2, out_channels=out_channels, kernel_size=kSz, padding=(kSz-1)//2)
        self.bn2 = nn.BatchNorm3d(num_features=out_channels)
        self.relu = nn.ReLU()
        self.pooling = nn.MaxPool3d(kernel_size=(2,2,2), stride=2)

    
    def forward(self, input):
        res = self.relu(self.bn1(self.conv1(input)))
        res = self.relu(self.bn2(self.conv2(res)))
        out = self.pooling(res)
        return out
   


class ChannelAttention(nn.Module):
    def __init__(self, channel, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.max_pool = nn.AdaptiveMaxPool3d(1)
        self.fc = nn.Sequential(
            nn.Conv3d(channel, channel // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv3d(channel // reduction, channel, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return x * self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv3d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.concat([avg_out, max_out], dim=1)
        out = self.conv(out)
        return x * self.sigmoid(out) 
    
class CBAM(nn.Module):
    def __init__(self, channel, reduction=16, kernel_size=7):
        super().__init__()
        self.ca = ChannelAttention(channel, reduction)
        self.sa = SpatialAttention(kernel_size)
    
    def forward(self, x):
        x = self.ca(x)
        x = self.sa(x)
        return x

class SE(nn.Module):
    # https://github.com/changzy00/pytorch-attention/blob/master/attention_mechanisms/se_module.py
    def __init__(self, channel, reduction=16):
        super(SE, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, w, h,z = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1,1)
        return x * y.expand_as(x)

# ===================================MODEL_SIZE==========================================
# from torchinfo import summary
# net = Slice_LOC(192,3)
# print(summary(model = net,
# 		input_size = (3,1,192,192,192), # (batch, color_channe,s,h,w)
# 		col_names = ["input_size","output_size","num_params"],
# 		col_width = 20,
# row_settings = ["var_names"]))
# =============================================================================
