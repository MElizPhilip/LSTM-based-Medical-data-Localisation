# 3D Localisation on Promise12
## DATA
 The model is trained and tested on Promise12 dataset, from the MICCAI 2012 Prostrate Segmentation Challenge. The size of each individual volume is quite large, therefore it was resized by zero padding along the first dimension and rescaling along the other two to obtain a size of (64, 256, 256)

## RUN
To train the model with the default setting use 

```python Promise_v3dL_TRAIN.py any_name_to_distinguish_the_model ```

To change any default setting, say the batch size, use

```python Promise_v3dL_TRAIN.py any_name_to_distinguish_the_model -b 5 ```

## Results

![](/../main/assets/promise12.png)

