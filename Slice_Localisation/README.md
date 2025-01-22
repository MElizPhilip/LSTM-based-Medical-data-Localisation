# Slice-Localisation
 Slice localisation is the 1D equivalent of the 3D localisation problem, where the subset of slices along one plane containing the foreground is identified.
 Let _D_ be a 3D volume with _A_ coronal slices (c<sub>1</sub>, c<sub>2</sub>, ... , c<sub>A</sub>), _B_ sagittal slices (s<sub>1</sub>, s<sub>2</sub>, ... , s<sub>B</sub>) and _C_ axial slices (a<sub>1</sub>, a<sub>2</sub>, ... , a<sub>C</sub>). The network is learning to regress the two slice extremities encompassing the foreground, if slice localisation is along the coronal plane then the results are denoted by (c<sub>st</sub>, c<sub>end</sub>), the start and end coronal slice extremities.

 ## DATA
 The model is trained and tested on the in-house, 3D fetal echocardiogram data at the end-diastole time point of the cardiac cycle. All 3D volumes are zero padded to change the size to (192,192, 192) with a voxel spacing of (0.5, 0.5, 0.5).

## RUN
To train the model with the default setting use 

```python v3dL_TRAIN.py any_name_to_distinguish_the_model ```

To change any default setting, say the batch size, use

```python v3dL_TRAIN.py any_name_to_distinguish_the_model -b 5 ```

## RESULTS
Image shows the slice localisation of four samples, shown on the sagittal slices.


![](/../main/assets/slice_loc_results.png)
