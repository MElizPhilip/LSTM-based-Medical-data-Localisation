# 3D Localisation on Fetal Echocardiogram, In-house data

## DATA
 The model is trained and tested on the in-house, 3D fetal echocardiogram data at the end-diastole time point of the cardiac cycle. All 3D volumes are zero padded to change the size to (192,192, 192) with a voxel spacing of (0.5, 0.5, 0.5).

## RUN
To train the model with the default setting use 

```python Loc3D_v3dL_TRAIN.py any_name_to_distinguish_the_model ```

To change any default setting, say the batch size, use

```python Loc3D_v3dL_TRAIN.py any_name_to_distinguish_the_model -b 5 ```

