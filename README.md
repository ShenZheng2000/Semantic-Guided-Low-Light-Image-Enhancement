# Semantic-Guided-Low-Light-Image-Enhancement
This is the official Pytorch implementation for our paper "Semantic-Guided Zero-Shot Learning for Low-Light Image/Video Enhancement."

# Abstract
Low-light images challenge both human perceptions and computer vision algorithms. It is crucial to make algorithms robust to enlighten low-light images for computational photography and computer vision applications such as real-time detection and segmentation tasks. This paper proposes a semantic-guided zero-shot low-light enhancement network which is trained in the absence of paired images, unpaired datasets, and segmentation annotation. Firstly, we design an efficient **enhancement factor extraction** network using depthwise separable convolution. Secondly, we propose a **recurrent image enhancement** network for progressively enhancing the low-light image. Finally, we introduce an **unsupervised semantic segmentation** network for preserving the semantic information. Extensive experiments on various benchmark datasets and a low-light video demonstrate that our model outperforms the previous state-of-the-art qualitatively and quantitatively. We further discuss the benefits of the proposed method for low-light detection and segmentation.

# Sample Results
## 1. Low-Light Video Frames
From left to right, and from top to bottom: Dark, Retinex [1], KinD [2], EnlightenGAN [3], Zero-DCE [4], Ours.

<p float="left">
<p align="middle">
  <img src="Samples/F1.png" width="250" />
  <img src="Samples/retinex_net.png" width="250" /> 
  <img src="Samples/kind.png" width="250" />
  <img src="Samples/enlighten_gan.png" width="250" />
  <img src="Samples/zero_dce.png" width="250" /> 
  <img src="Samples/F1Crop.png" width="250" />
</p>

## 2. Low-Light Images (Real-World)
From left to right, and from top to bottom: Dark, PIE [5], LIME [6], Retinex [1], MBLLEN [7], KinD [2] , Zero-DCE [4], Ours

<p float="left">
<p align="middle">
  <img src="Samples/Dark7.jpg" width="200" />
  <img src="Samples/PIE7.jpg" width="200" /> 
  <img src="Samples/LIME7.jpg" width="200" />
  <img src="Samples/Retinex7.jpg" width="200" />
  <img src="Samples/mbllen7.jpg" width="200" /> 
  <img src="Samples/KinD7.jpg" width="200" />
  <img src="Samples/ZeroDCE7.jpg" width="200" /> 
  <img src="Samples/Ours7.jpg" width="200" />
</p>

# Get Started

## 1. Requirements
* CUDA 10.0
* Python 3.6+
* Pytorch 1.0+
* torchvision 0.4+
* opencv-python
* numpy
* pillow
* scikit-image

## 2. Prepare Datasets
The official testing dataset is at the folder `data/test_data/lowCUT/`. You can put other datasets in this folder for testing. 
Example: `data/test_data/yourDataset/`

The official training dataset can be downloaded from [BaiduYun](https://pan.baidu.com/s/19ez3dM079WksPRB0Xw98kg) with code `n93t`. After download, move the unzipped file into `data/`. After you unzip the file, the training images will be in `data/train_data`.

## 3. Training 
 ```
 python train.py
 ```
 
## 4. Testing
```
python test.py
```
# Pretrained Weights
TODO

# Hyperparameters
TODO

# Citations
TODO

# Others
Please reach zhengsh@kean.edu if you have any questions. This repository is heavily based upon [Zero-DCE](https://github.com/Li-Chongyi/Zero-DCE). Thanks for sharing the code!

# References
[1] Wei, Chen, et al. "Deep retinex decomposition for low-light enhancement." arXiv preprint arXiv:1808.04560 (2018).

[2] Zhang, Yonghua, Jiawan Zhang, and Xiaojie Guo. "Kindling the darkness: A practical low-light image enhancer." Proceedings of the 27th ACM international conference on multimedia. 2019.

[3] Jiang, Yifan, et al. "Enlightengan: Deep light enhancement without paired supervision." IEEE Transactions on Image Processing 30 (2021): 2340-2349.

[4] Guo, Chunle, et al. "Zero-reference deep curve estimation for low-light image enhancement." Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2020.

[5] Fu, Xueyang, et al. "A probabilistic method for image enhancement with simultaneous illumination and reflectance estimation." IEEE Transactions on Image Processing 24.12 (2015): 4965-4977.

[6] Guo, Xiaojie, Yu Li, and Haibin Ling. "LIME: Low-light image enhancement via illumination map estimation." IEEE Transactions on image processing 26.2 (2016): 982-993.

[7] Lv, Feifan, et al. "MBLLEN: Low-Light Image/Video Enhancement Using CNNs." BMVC. 2018.

