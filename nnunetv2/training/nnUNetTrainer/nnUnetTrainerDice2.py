import numpy as np
import torch
from monai.losses import DiceFocalLoss, FocalLoss, DiceLoss
from monai.networks.nets import Unet
from torch import nn
from torchinfo import summary

from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
from nnunetv2.training.loss.dice import MemoryEfficientSoftDiceLoss
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager, ConfigurationManager
from nnunetv2.training.loss.custom_losses import DC_and_FOCAL_loss


class nnUNetTrainerDice2(nnUNetTrainer):
    """ Swin-UMamba """

    def _build_loss(self):
        if self.label_manager.has_regions:
            raise ValueError("Why has regions is set to True :(")
        else:
            # loss = DC_and_FOCAL_loss({'batch_dice': self.configuration_manager.batch_dice,
            #                           'smooth': 1e-5, 'do_bg': False, 'ddp': self.is_ddp},
            #                          { "to_onehot_y":True, "use_softmax": True, "gamma": 0.3},
            #                          weight_focal=1, weight_dice=1,
            #                          ignore_label=self.label_manager.ignore_label,
            #                          dice_class=MemoryEfficientSoftDiceLoss)
            loss = DiceLoss(to_onehot_y=True, softmax=True)

        # if self._do_i_compile():
        #     loss.dc = torch.compile(loss.dc)

    # we give each output a weight which decreases exponentially (division by 2) as the resolution decreases
    # this gives higher resolution outputs more weight in the loss

        if self.enable_deep_supervision:
            deep_supervision_scales = self._get_deep_supervision_scales()
            weights = np.array([1 / (2 ** i) for i in range(len(deep_supervision_scales))])
            if self.is_ddp and not self._do_i_compile():
                # very strange and stupid interaction. DDP crashes and complains about unused parameters due to
                # weights[-1] = 0. Interestingly this crash doesn't happen with torch.compile enabled. Strange stuff.
                # Anywho, the simple fix is to set a very low weight to this.
                weights[-1] = 1e-6
            else:
                weights[-1] = 0

            # we don't use the lowest 2 outputs. Normalize weights so that they sum to 1
            weights = weights / weights.sum()
            # now wrap the loss
            loss = DeepSupervisionWrapper(loss, weights)

        return loss