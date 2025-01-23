# LSTM-based-Medical-data-Localisation

This is the implementation of the paper, **CONTEXT-AWARE LSTM FOR 3D MEDICAL IMAGE LOCALISATION**, accepted for publication at **ISBI 2025**.
## Abstract
Medical images contain higher spatial correlation between data points compared to natural images. In this work we exploit this feature to develop a localisation network for application in 3D medical imaging data. 
Long-Short-term memory layers, are used to learn the distinct pattern in the organ anatomy which we propose will robustly localize organs and also deformations in organs caused by tumors that disrupt typical patterns/appearance.  Our network achieves an average precision of 89% for fetal heart localisation on echocardiogram data, 83\% for tumor localisation on the BraTS 2021 dataset and 80% for prostrate localisation on the Promise12 dataset. With a trainable parameter size (3.2M parameters) comparable to MobileNet-V2 for slice localisation and less than 10M for 3D we present a computationally efficient network for 3D localisation on medical data. 

## Slice Localisation Vs 3D Localisation


