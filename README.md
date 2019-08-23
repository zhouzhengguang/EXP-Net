# EXP-Net
This repository represents training examples for the paper "Progressive Learning of Low-Precision Networks for Image Classification"
# Method
Based on a lowprecision network, we equip each low-precision convolutional layer (LPconv) with another full-precision one during training.
A decreasing factor f (the blue curve) is used to reduce the output of full-precision layer gradually to zero. The fullprecision part is removed for network inference finally.
# Dependencies
Python 3   
TensorFlow >=1.7       
TensorPack >=0.8    
# Use
* For SVHN   
Baseline: `python svhn_dorefa.py --dorefa 1,1,32 --gpu 0`    
EXP-Net:  `python exp_svhn_dorefa.py --dorefa 1,1,32 --gpu 0`     
* For ImageNet      
Baseline: `python syq_alexnet.py --data <raw data path> --eta 0.0 --name <log name> --gpu 0`      
EXP-Net:  `python exp_syq_alexnet.py --data <raw data path> --eta 0.0 --name <log name> --gpu 0`      
* Result   

