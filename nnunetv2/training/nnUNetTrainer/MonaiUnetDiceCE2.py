import numpy as np
import torch
from monai.losses import DiceFocalLoss
from monai.losses import DiceLoss, DiceCELoss
from monai.networks.nets import Unet
from torch import nn
from torchinfo import summary

from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
from nnunetv2.training.loss.dice import MemoryEfficientSoftDiceLoss
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager, ConfigurationManager
from nnunetv2.training.loss.custom_losses import DC_and_FOCAL_loss


class MonaiUnetDiceCE2(nnUNetTrainer):
    """ Swin-UMamba """

    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.initial_lr = 1e-4
        self.weight_decay = 5e-2
        self.enable_deep_supervision = False
        self.freeze_encoder_epochs = -1  # Training from scratch
        self.early_stop_epoch = 10
        self.num_epochs = 10

    @staticmethod
    def build_network_architecture(
            architecture_class_name:str,
            arch_init_kwargs:dict,
            arch_init_kwargs_reg_import,
            num_input_channels: int,
            num_output_channels: int,
            #plans_manager: PlansManager,
            #dataset_json,
            #configuration_manager: ConfigurationManager,
            enable_deep_supervision: bool = True,
            use_pretrain: bool = False,
    ) -> nn.Module:
        # label_manager = plans_manager.get_label_manager(dataset_json)
        model = Unet(
            spatial_dims=3,
            in_channels=num_input_channels,
            out_channels=2,
            channels=(32,
                      64,
                      128,
                      256,
                      320,
                      320),
            strides=(1, 2, 2, 2, 2, 1),
            dropout=0.2,
        )
        print(model)

        # summary(model, input_size=[1, num_input_channels] + configuration_manager.patch_size)

        return model

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
            loss = DiceCELoss(softmax=True, to_onehot_y=True, label_smoothing=0.9)

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

    def _get_deep_supervision_scales(self):
        if self.enable_deep_supervision:
            deep_supervision_scales = [[1.0, 1.0]] * 7
        else:
            deep_supervision_scales = None  # for train and val_transforms
        return deep_supervision_scales

    # def configure_optimizers(self):
    #     optimizer = AdamW(
    #         self.network.parameters(),
    #         lr=self.initial_lr,
    #         weight_decay=self.weight_decay,
    #         eps=1e-5,
    #         betas=(0.9, 0.999),
    #     )
    #     scheduler = CosineAnnealingLR(optimizer, T_max=self.num_epochs, eta_min=1e-6)
    #
    #     self.print_to_log_file(f"Using optimizer {optimizer}")
    #     self.print_to_log_file(f"Using scheduler {scheduler}")
    #
    #     return optimizer, scheduler

    # def on_epoch_end(self):
    #     current_epoch = self.current_epoch
    #     if (current_epoch + 1) % self.save_every == 0:
    #         self.save_checkpoint(join(self.output_folder, f'checkpoint_{current_epoch}.pth'))
    #     super().on_epoch_end()

    # def on_train_epoch_start(self):
    #     # freeze the encoder if the epoch is less than 10
    #     if self.current_epoch < self.freeze_encoder_epochs:
    #         self.print_to_log_file("Freezing the encoder")
    #         if self.is_ddp:
    #             self.network.module.freeze_encoder()
    #         else:
    #             self.network.freeze_encoder()
    #     else:
    #         self.print_to_log_file("Unfreezing the encoder")
    #         if self.is_ddp:
    #             self.network.module.unfreeze_encoder()
    #         else:
    #             self.network.unfreeze_encoder()
    #     super().on_train_epoch_start()

    def set_deep_supervision_enabled(self, enabled: bool):
        """
        This function is specific for the default architecture in nnU-Net. If you change the architecture, there are
        chances you need to change this as well!
        """
        if self.is_ddp:
            self.network.module.deep_supervision = enabled
        else:
            self.network.deep_supervision = enabled