# Semantic-Guided-Low-Light-Image-Enhancement
This is the official Pytorch implementation for our paper "Semantic-Guided Zero-Shot Learning for Low-Light Image/Video Enhancement."

# Abstract
Low-light images challenge both human perceptions and computer vision algorithms. It is crucial to make algorithms robust to enlighten low-light images for computational photography and computer vision applications such as real-time detection and segmentation tasks. This paper proposes a semantic-guided zero-shot low-light enhancement network which is trained in the absence of paired images, unpaired datasets, and segmentation annotation. Firstly, we design an efficient enhancement factor extraction network using depthwise separable convolution. Secondly, we propose a recurrent image enhancement network for progressively enhancing the low-light image. Finally, we introduce an unsupervised semantic segmentation network for preserving the semantic information. Extensive experiments on various benchmark datasets and a low-light video demonstrate that our model outperforms the previous state-of-the-art qualitatively and quantitatively. We further discuss the benefits of the proposed method for low-light detection and segmentation.

# Sample Results
![alt-text-1](Samples/F1.png "Dark") ![alt-text-2](Samples/retinex_net.png "Retinex") ![alt-text-3](Samples/kind.png "KinD")


# Prepare Datasets
The official testing dataset is at the folder `data/test_data/lowCUT/`. You can put other datasets in this folder for testing. 
For example: `data/test_data/yourDataset/'

The official training dataset can be downloaded from [BaiduYun](https://pan.baidu.com/s/19ez3dM079WksPRB0Xw98kg) with code `n93t`. After download, move the unzipped file into `data/'. After you unzip the file, the training images will be in  `data/train_data`.

# Training 
 `python train.py`
 
# Testing
`python test.py`

# Results
TODO

# Citation 
TODO
