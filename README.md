# LSTM-based-Medical-data-Localisation

This is the implementation of the paper, **CONTEXT-AWARE LSTM FOR 3D MEDICAL IMAGE LOCALISATION**, accepted for publication at **ISBI 2025**.
## Abstract
Medical images contain higher spatial correlation between data points compared to natural images. In this work we exploit this feature to develop a localisation network for application in 3D medical imaging data. 
Long-Short-term memory layers, are used to learn the distinct pattern in the organ anatomy which we propose will robustly localize organs and also deformations in organs caused by tumors that disrupt typical patterns/appearance.  Our network achieves an average precision of 89% for fetal heart localisation on echocardiogram data, 83\% for tumor localisation on the BraTS 2021 dataset and 80% for prostrate localisation on the Promise12 dataset. With a trainable parameter size (3.2M parameters) comparable to MobileNet-V2 for slice localisation and less than 10M for 3D we present a computationally efficient network for 3D localisation on medical data. 

## Slice Localisation Vs 3D Localisation

Let _V_ be a 3D volume with _A_ coronal slices (c<sub>1</sub>, c<sub>2</sub>, ... , c<sub>A</sub>), _B_ sagittal slices (s<sub>1</sub>, s<sub>2</sub>, ... , s<sub>B</sub>) and _C_ axial slices (a<sub>1</sub>, a<sub>2</sub>, ... , a<sub>C</sub>). The network is learning to regress the six bounding box extremities (c<sub>st</sub>, c<sub>end</sub>, s<sub>st</sub>, s<sub>end</sub>, a<sub>st</sub>, a<sub>end</sub>) for 3D localisation and one of (c<sub>st</sub>, c<sub>end</sub>) / (s<sub>st</sub>, s<sub>end</sub>) / (a<sub>st</sub>, a<sub>end</sub>) for slice localisation. See image below for a pictorial representation of the difference.

![](/../main/assets/LOC_main.png)

## Network Architecture
In the image below, (a) Architecture; Slice localisation network highlighted in yellow. (b) Prediction branch structure.

![](/../main/assets/flow.png)
