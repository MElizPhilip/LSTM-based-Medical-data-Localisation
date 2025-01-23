# 3D Localisation on BraTS, FLAIR dataset, from the MICCAI 2021 Challenge
## DATA
 The model is trained and tested on BraTS, FLAIR dataset, from the MICCAI 2021 Challenge. The surrounding zero intensity regions around all samples were removed and zero padding applied along the first dimension to obtain volumes of shape (160, 224, 224).

## RUN
To train the model with the default setting use 

```python BraTS_v3dL_TRAIN.py any_name_to_distinguish_the_model ```

To change any default setting, say the batch size, use

```python BraTS_v3dL_TRAIN.py any_name_to_distinguish_the_model -b 5 ```

## Results

![](/../main/assets/brats.png)

