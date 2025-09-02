from monai.networks.nets import Unet
import numpy as np
import torch


data = np.load("data.npz.npy")
print(data.shape)

model = Unet(
            spatial_dims=3,
            in_channels=1,
            out_channels=2,
            channels=(16, 32, 64, 128, 256, 512),
            strides=(2, 1, 1, 2, 2),
            dropout=0.3,
        ).to("cuda:1")
print(model(torch.tensor(data).to("cuda:1")).shape)