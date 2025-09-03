# nnUNet/ Unet Calibration (Label Smoothing)
## Installation
This repository is based on original [nnUNet repository](https://github.com/MIC-DKFZ/nnUNet/tree/master) and all the commands for nnUNet should work for this repository. The main changes are
the posibility to train nnUNet with different loss functions and also adding other segmentation models (specifically from MONAI library) to nnUNet
preprocessing pipeline and train and test the model. In this repository we already have nnUNet and UNet with different loss functions including Dice,
Dice+CE, CE, Focal, Dice+Focal, and NACL.  
To start, please first install [PyTorch](https://pytorch.org/get-started/locally/) & [MONAI](https://docs.monai.io/en/latest/installation.html). Then clone this repository and install dependecies:  
