# nnUNet/ Unet Calibration (Label Smoothing)
## Installation
This repository is based on original [nnUNet repository](https://github.com/MIC-DKFZ/nnUNet/tree/master) and all the commands for nnUNet should work for this repository. The main changes are
the posibility to train nnUNet with different loss functions and also adding other segmentation models (specifically from MONAI library) to nnUNet
preprocessing pipeline and train and test the model. In this repository we already have nnUNet and UNet with different loss functions including Dice,
Dice+CE, CE, Focal, Dice+Focal, and NACL.  
To start, please first install [PyTorch](https://pytorch.org/get-started/locally/) & [MONAI](https://docs.monai.io/en/latest/installation.html). Then clone this repository and install dependecies: 
```
git clone https://github.com/sahar90o/BMs_Segmentation_private.git
cd BMs_Segmentation_private
pip install e .
```
## Data Preparation
To use nnUNet, and also other models in this repository, you should prepare your data in specific format and naming similar to Medical Segmentation Decathlon. You can 
see the detailed explanation and examples in original nnUNet repository. Please check it [here](https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/dataset_format.md). After preparing the data you should also create a .json file for your data as described in the previous link. you should have three folders: nnUNet_raw, where you have your raw data. nnUNet_preprocessed, where the preprocessed data will be saved, and nnUNet_results, where the model weights will be saved. You should set environment variables for this folders. Therefore, before training your model please set your environment variables so that nnUNet knows where is your data. In your Linux terminal you can use these command: 
```
export nnUNet_raw="mnt/Data/sahar/segmentation/nnUNet_raw"
export nnUNet_preprocessed="mnt/Data/sahar/segmentation/nnUNet_preprocessed"
export nnUNet_results="mnt/Data/sahar/segmentation/nnUNet_results"
```
Please replace the above directories with your own directories :) Also add data fingerprint. 

## Train and Test Models 
Perfect! Now you can train your model. In this repository we have different trainers which are located in nnunetv2/training/nnUNetTrainer. To start training you should specify the model, the configurations,and the data fold for training. You can specify the model by ```-tr```, the configuration by ```-c```, the fold by ```-f```. For the configuration I only used 3d_fullres. For nnUNet you can also use 2d, 3d_lowres, and 3d_cascade_lowres depending on your preprocessed data. You can train the model for fold 0,1, 2, 3, 4, or train it on all folds using ```-f all```. You can train a model using command below: 
```
nnUNetv2_train Dataset_name_or_ID -tr nnUNetTrainerFocal -c 3d_fullres -f all
```
After model training for inference you can run this command: 
```
nnUNetv2_predict -i INPUT_FOLDER -o OUTPUT_FOLDER -tr -d DATASET_NAME_OR_ID -tr nnUNetTrainerFocal -c 3d_fullres -f all --save_probabilities
```
When you use ```save_probabilities``` the probability map of classes will be exported. 
For more detailed commands please check original nnUNet repository [here](https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/how_to_use_nnunet.md). 
Ok, now if you want to calibrate the model using label smoothing you can train the models with Cross Entropy (CE) or Dice+CE loss and increase the label_smoothing parameter in the loss function (the defualt value is zero). I recommend you to set the smoothing parameter between 0.2 and 0.4. You should modify label smoothing manually. For example to train MonaiUNet with Dice+CE loss. Please open nnUNetTrainerMonaiUnetDiceCE which is located in nnunetv2/training/nnUNetTrainer and in line 72 

  
