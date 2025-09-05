import numpy as np
import torch
from monai.losses import DiceFocalLoss
from monai.networks.nets import Unet
from torch import nn, autocast
from monai.losses import NACLLoss
from nnunetv2.utilities.helpers import dummy_context
from nnunetv2.training.loss.dice import get_tp_fp_fn_tn, MemoryEfficientSoftDiceLoss
from torchinfo import summary

from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
from nnunetv2.training.loss.dice import MemoryEfficientSoftDiceLoss
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager, ConfigurationManager
from nnunetv2.training.loss.custom_losses import DC_and_FOCAL_loss


class MonaiUnetNACL(nnUNetTrainer):


    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.initial_lr = 1e-4
        self.weight_decay = 5e-2
        self.enable_deep_supervision = False
        self.freeze_encoder_epochs = -1  # Training from scratch
        self.early_stop_epoch = 10
        self.num_epochs = 300

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
            enable_deep_supervision: bool = False,
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
            dropout=0.3,
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
            loss = NACLLoss(classes=2, dim=3)

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

    def train_step(self, batch: dict) -> dict:
        data = batch['data']
        target = batch['target']

        data = data.to(self.device, non_blocking=True)
        if isinstance(target, list):
            target = [i.to(self.device, non_blocking=True) for i in target]
        else:
            target = target.to(self.device, non_blocking=True)

        self.optimizer.zero_grad(set_to_none=True)

        with autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else dummy_context():
            output = self.network(data)
            # del data
            l = self.loss(output, target.squeeze(1).long())

        if self.grad_scaler is not None:
            self.grad_scaler.scale(l).backward()
            self.grad_scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            l.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.optimizer.step()
        return {'loss': l.detach().cpu().numpy()}

    def validation_step(self, batch: dict) -> dict:
        data = batch['data']
        target = batch['target']

        data = data.to(self.device, non_blocking=True)
        if isinstance(target, list):
            target = [i.to(self.device, non_blocking=True) for i in target]
        else:
            target = target.to(self.device, non_blocking=True)

        # Autocast can be annoying
        # If the device_type is 'cpu' then it's slow as heck and needs to be disabled.
        # If the device_type is 'mps' then it will complain that mps is not implemented, even if enabled=False is set. Whyyyyyyy. (this is why we don't make use of enabled=False)
        # So autocast will only be active if we have a cuda device.
        with autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else dummy_context():
            output = self.network(data)
            del data
            l = self.loss(output, target.squeeze(1).long())
            # we only need the output with the highest output resolution (if DS enabled)
            if self.enable_deep_supervision:
                output = output[0]
                target = target[0]

            # the following is needed for online evaluation. Fake dice (green line)
            axes = [0] + list(range(2, output.ndim))

            if self.label_manager.has_regions:
                predicted_segmentation_onehot = (torch.sigmoid(output) > 0.5).long()
            else:
                # no need for softmax
                output_seg = output.argmax(1)[:, None]
                predicted_segmentation_onehot = torch.zeros(output.shape, device=output.device, dtype=torch.float32)
                predicted_segmentation_onehot.scatter_(1, output_seg, 1)
                del output_seg
            if self.label_manager.has_ignore_label:
                if not self.label_manager.has_regions:
                    mask = (target != self.label_manager.ignore_label).float()
                    # CAREFUL that you don't rely on target after this line!
                    target[target == self.label_manager.ignore_label] = 0
                else:
                    if target.dtype == torch.bool:
                        mask = ~target[:, -1:]
                    else:
                        mask = 1 - target[:, -1:]
                    # CAREFUL that you don't rely on target after this line!
                    target = target[:, :-1]
            else:
                mask = None

            tp, fp, fn, _ = get_tp_fp_fn_tn(predicted_segmentation_onehot, target, axes=axes, mask=mask)

            tp_hard = tp.detach().cpu().numpy()
            fp_hard = fp.detach().cpu().numpy()
            fn_hard = fn.detach().cpu().numpy()
            if not self.label_manager.has_regions:
                # if we train with regions all segmentation heads predict some kind of foreground. In conventional
                # (softmax training) there needs tobe one output for the background. We are not interested in the
                # background Dice
                # [1:] in order to remove background
                tp_hard = tp_hard[1:]
                fp_hard = fp_hard[1:]
                fn_hard = fn_hard[1:]

            return {'loss': l.detach().cpu().numpy(), 'tp_hard': tp_hard, 'fp_hard': fp_hard, 'fn_hard': fn_hard}

