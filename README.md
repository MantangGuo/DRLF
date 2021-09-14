# DRLF

PyTorch implementation of IEEE TPAMI 2021 paper: "[Deep Spatial-angular Regularization for Light Field Imaging, Denoising, and Super-resolution](https://ieeexplore.ieee.org/abstract/document/9448470/)". The video demo is [here](https://drive.google.com/file/d/1pud0vAPNNjlBkSl84ro7HTxT2T5OBcWi/view?usp=sharing)

## Requrements
- Python 3.6.10
- PyTorch 1.7.1
- Matlab (for training/test data generation)





## Compressive LF Reconstruction

### 1. Dataset
We provide MATLAB code for preparing the training and test data. Please first download light field datasets, and put them into corresponding folders in LFData.

### 2. Test
We provide the pre-trained models for tasks 1 -> 49, 2 -> 49, and 4 -> 49 on the Lytro dataset. Enter the LFCA folder and run:

__Task 1 -> 49__
```
python lfca_test.py --measurementNum 1
```
__Task 2 -> 49__
```
python lfca_test.py --measurementNum 2
```
__Task 4 -> 49__
```
python lfca_test.py --measurementNum 4
```

### 3. Train
Enter the LFCA folder and run:
```
python lfca_train.py
```



## LF Denoising

### 1. Dataset
We provide MATLAB code for preparing the training and test data. Please first download light field datasets, and put them into corresponding folders in LFData. We used the same dataset, noise synthesis and preprocessing protocol as [APA](https://ieeexplore.ieee.org/abstract/document/8423122)

### 2. Test
We provide the pre-trained models for adding zero-mean Gaussian noise with the standard variance varying in the range of 10, 20, and 50 on the Lytro dataset. Enter the LFDN folder and run:

__Noise level 10__
```
python lfdn_test.py --sigma 10
```
__Noise level 20__
```
python lfdn_test.py --sigma 20
```
__Noise level 50__
```
python lfdn_test.py --sigma 50
```
### 3. Train
Enter the LFCA folder and run:
```
python lfdn_train.py
```




## LF Spatial SR

### 1. Dataset
We provide MATLAB code for preparing the training and test data. Please first download light field datasets, and put them into corresponding folders in LFData. We used the same dataset and protocol as those of [Pseudo-4D](https://ieeexplore.ieee.org/abstract/document/8561240/) to generate low-resolution LF images.

### 2. Test
We provide the pre-trained models for tasks 2x and 4x on the Lytro dataset. Enter the LFSSR folder and run:

__Task 2x__
```
python lfssr_test.py --scaleFactor 2
```
__Task 4x__
```
python lfssr_test.py --scaleFactor 4
```

### 3. Train
Enter the LFSSR folder and run:
```
python lfssr_train.py
```
