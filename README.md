# Segmentation Calibration (Label Smoothing)
This repository is based on original nnUNet repository and all the commands for nnUNet should work for this repository. The main changes are
the posibility to train nnUNet with different loss functions and also adding other segmentation models (specifically from MONAI library) to nnUNet
preprocessing pipeline and train and test the model. In this repository we already have nnUNet and UNet with different loss functions including Dice,
Dice+CE, CE, Focal, Dice+Focal, and NACL. All these models are located in nnunetv2/training/nnUNetTrainer directory. 
