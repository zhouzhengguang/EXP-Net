# EXP-Net
This repository represents training examples for the paper "Progressive Learning of Low-Precision Networks for Image Classification"
# Method
Based on a lowprecision network, we equip each low-precision convolutional layer (LPconv) with another full-precision one during training.
A decreasing factor f (the blue curve) is used to reduce the output of full-precision layer gradually to zero. The fullprecision part is removed for network inference finally.
<img src="EXP-Net.png" alt="drawing" width="2000"/>
# Dependencies
Python 3   
TensorFlow    
TensorPack   
# Use
* For SVHN   
Baseline: `python svhn_dorefa.py --dorefa 1,1,32 --gpu 0`
EXP-Net:  `python exp_svhn_dorefa.py --dorefa 1,1,32 --gpu 0`
* For ImageNet
Baseline: ``

