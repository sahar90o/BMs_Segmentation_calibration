# Sahar 18th June 2024
# 3D Unet for Uncertainty Estimation
import monai
from monai.utils import first
from monai.utils import set_determinism
from monai.transforms import (
    AsDiscrete,
    AsDiscreted,
    EnsureChannelFirstd,
    Compose,
    CropForegroundd,
    LoadImaged,
    Orientationd,
    RandCropByPosNegLabeld,
    SaveImaged,
    ScaleIntensityd,
    ScaleIntensityRanged,
    Spacingd,
    Resized,
    Invertd,
    NormalizeIntensityd
)

from monai.handlers.utils import from_engine
from monai.networks.nets import UNet
from monai.networks.layers import Norm
from torch.nn import CrossEntropyLoss
from monai.metrics import DiceMetric
from monai.losses import DiceLoss,DiceCELoss
from monai.losses import FocalLoss, DiceFocalLoss
from monai.inferers import sliding_window_inference
from monai.data import DataLoader, Dataset, CacheDataset, decollate_batch
from monai.config import print_config
import torch
import matplotlib.pyplot as plt
from torch import nn
import tempfile
import shutil
import os
import glob

print_config()

print(torch.cuda.is_available())

root_dir = '/mnt/data/zahra/Ensemble_data_deepmed/'
print(root_dir)
data_dir = os.path.join(root_dir,'DeepMed')
print(data_dir)
train_images = sorted(glob.glob(os.path.join(data_dir,"Train","*.nii.gz")))
train_labels = sorted(glob.glob(os.path.join(data_dir,"Train_labels","*.nii.gz")))
data_dicts = [{'image': image_name, 'label': label_name} for image_name, label_name in zip(train_images, train_labels)]
train_files, val_files = data_dicts[:-14], data_dicts[-14:]
print(val_files)

set_determinism(12)
# Set up train and validation transforms ------------------------------------------------------
train_transforms = Compose(
    [
        LoadImaged(keys=["image","label"]),
        EnsureChannelFirstd(keys=["image","label"]),
        NormalizeIntensityd(
            keys=["image"],
            nonzero=True,
            channel_wise=True
        ),
        Orientationd(keys=["image","label"], axcodes ="RAI"),
        Spacingd(keys=["image","label"], pixdim=(1.0, 1.0, 1.0), mode =("bilinear","nearest")),
        Resized(keys=["image","label"], spatial_size=(240,240,240))
    ]
    )

val_transforms = Compose(
    [
        LoadImaged(keys=["image","label"]),
        EnsureChannelFirstd(keys=["image","label"]),
        NormalizeIntensityd(
            keys=["image"],
            nonzero=True,
            channel_wise=True

        ),
        Orientationd(keys=["image","label"], axcodes='RAI'),
        Spacingd(keys=["image","label"], pixdim = (1.0, 1.0, 1.0), mode = ('bilinear','nearest')),
        Resized(keys=["image","label"],spatial_size=(240,240,240))

    ]
)

# Check the transforms in DataLoader ----------------------------------------------------------

#check_ds = Dataset(data=train_files, transform=train_transforms)
#check_loader = DataLoader(check_ds, batch_size=2)
#check_data = first(check_loader)
#image, label = (check_data["image"][0][0],check_data["label"][0][0])

#print(f'image shape: {image.shape}, label shape: {label.shape}')

#plt.figure('check',(12,6))
#plt.subplot(1,2,1)
#plt.title("image")
#plt.imshow(image[:,:,120],cmap='gray')

#plt.subplot(1,2,2)
#plt.title("label")
#plt.imshow(label[:,:,120])
#plt.show()

#Define Cache dataset and dataloader for train and validation ----------------------------------------
train_ds = CacheDataset(data=train_files, transform=train_transforms, cache_rate=0.5, num_workers=30)
train_loader = DataLoader(train_ds, batch_size=3, shuffle=True, num_workers=30)

val_ds = CacheDataset(data=val_files, transform=val_transforms, cache_rate=0.5, num_workers=30)
val_loader = DataLoader(val_ds, batch_size=3, num_workers=30)

#for i in train_loader:
    #print(i['image'].shape)

# Setup model and optimizer --------------------------------------------------------------------------
device = torch.device("cuda")
model = UNet(
    spatial_dims=3,
    in_channels=1,
    out_channels=2,
    channels=(16, 32, 64, 128, 256,512),
    strides=(2, 1, 1, 2, 2),
    num_res_units=2,
    dropout=0.3,
    norm=Norm.BATCH
)
model = nn.DataParallel(model)
model.to(device)

#loss_function = DiceLoss(to_onehot_y=True, softmax=True)
loss_function = DiceCELoss(to_onehot_y=True, softmax=True)
#loss_function = CrossEntropyLoss()
#loss_function = FocalLoss(gamma=3.0, to_onehot_y=True, use_softmax=True)
#loss_function = DiceFocalLoss(gamma=3.0, to_onehot_y=True, softmax=True)
optimizer = torch.optim.Adam(model.parameters(),1e-4)
dice_metric = DiceMetric(include_background=False, reduction="mean")

# Model Training ------------------------------------------------------------------------------------
max_epochs = 150
val_interval = 5
best_metric = -1
best_metric_epoch = -1
epoch_loss_values = []
metric_values = []
post_pred = Compose([AsDiscrete(argmax=True, to_onehot=2)])
post_label = Compose([AsDiscrete(to_onehot=2)])

for epoch in range(max_epochs):
    print('-'*20)
    print(f'epoch {epoch+1}/{max_epochs}')
    model.train()
    epoch_loss=0
    step=0
    for batch_data in train_loader:
        step += 1
        inputs, labels = (
            batch_data["image"].to(device),
            batch_data["label"].to(device)
        )

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_function(outputs,labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        print(f'{step}/{len(train_ds) // train_loader.batch_size},' f'train_loss: {loss.item(): .4f}')

    epoch_loss /= step
    epoch_loss_values.append(epoch_loss)
    print(f' epoch {epoch+1} average loss: {epoch_loss: .4f}')

    if (epoch+1) % val_interval ==0:
        model.eval()
        with torch.no_grad():
            for val_data in val_loader:
                val_inputs, val_labels = (
                    val_data['image'].to(device),
                    val_data['label'].to(device)
                )
                roi_size = (240, 240, 240)
                sw_batch_size = 3
                val_outputs = sliding_window_inference(val_inputs, roi_size, sw_batch_size, model)
                val_outputs = [post_pred(i) for i in decollate_batch(val_outputs)]
                val_labels = [post_label(i) for i in decollate_batch(val_labels)]
                dice_metric(y_pred=val_outputs, y=val_labels)

            metric = dice_metric.aggregate().item()
            dice_metric.reset()

            metric_values.append(metric)
            if metric > best_metric:

                best_metric = metric
                best_metric_epoch = epoch+1
                torch.save(model.state_dict(), os.path.join(root_dir,"Unet_deeper-4res-200.pth"))
                print('save the best model')

            print(
                f' Current epoch: {epoch+1} current mean dice: {metric: .4f}'
                f'\nbest mean dice: {best_metric: .4f}'
                f' at epoch: {best_metric_epoch}'
                )