from nnunetv2.training.loss.dice import SoftDiceLoss
from torch import nn
import torch
from monai.losses.focal_loss import  FocalLoss
from monai.losses import DiceFocalLoss
from nnunetv2.utilities.helpers import softmax_helper_dim1
from monai.losses.dice import DiceLoss



class DC_and_FOCAL_loss(nn.Module):
    def __init__(self,
                 soft_dice_kwargs,
                 focal_kwargs,
                 weight_focal=1,
                 weight_dice=1,
                 ignore_label=None,
                 dice_class=SoftDiceLoss):
        """
        Weights for CE and Dice do not need to sum to one. You can set whatever you want.
        :param soft_dice_kwargs:
        :param ce_kwargs:
        :param aggregate:
        :param square_dice:
        :param weight_ce:
        :param weight_dice:
        """
        super().__init__()

        # if ignore_label is not None:
        #     ce_kwargs['ignore_index'] = ignore_label

        self.weight_dice = weight_dice
        self.weight_focal = weight_focal
        self.ignore_label = ignore_label

        self.fl = FocalLoss(**focal_kwargs)
        self.dc = dice_class(apply_nonlin=softmax_helper_dim1, **soft_dice_kwargs)

    def forward(self, net_output: torch.Tensor, target: torch.Tensor):
        """
        target must be b, c, x, y(, z) with c=1
        :param net_output:
        :param target:
        :return:
        """
        if self.ignore_label is not None:
            assert target.shape[1] == 1, 'ignore label is not implemented for one hot encoded target variables ' \
                                         '(DC_and_CE_loss)'
            mask = target != self.ignore_label
            # remove ignore label from target, replace with one of the known labels. It doesn't matter because we
            # ignore gradients in those areas anyway
            target_dice = torch.where(mask, target, 0)
            num_fg = mask.sum()
        else:
            target_dice = target
            mask = None

        dc_loss = self.dc(net_output, target_dice, loss_mask=mask) \
            if self.weight_dice != 0 else 0
        fl_loss = self.fl(net_output, target) \
            if self.weight_focal != 0 and (self.ignore_label is None or num_fg > 0) else 0

        result = self.weight_focal * fl_loss + self.weight_dice * dc_loss
        return result


class DC_FOCAL_Monai_loss(nn.Module):
    def forward(self, net_output: torch.Tensor, target: torch.Tensor):
        """
        target must be b, c, x, y(, z) with c=1
        :param net_output:
        :param target:
        :return:
        """
        # if self.ignore_label is not None:
        #     assert target.shape[1] == 1, 'ignore label is not implemented for one hot encoded target variables ' \
        #                                  '(DC_and_CE_loss)'
        #     mask = target != self.ignore_label
        #     # remove ignore label from target, replace with one of the known labels. It doesn't matter because we
        #     # ignore gradients in those areas anyway
        #     target_dice = torch.where(mask, target, 0)
        #     num_fg = mask.sum()
        # else:
        #     target_dice = target
        #     mask = None

        # dc_loss = self.dc(net_output, target_dice, loss_mask=mask) \
        #     if self.weight_dice != 0 else 0
        loss = self.dc_fl(net_output, target)

        return loss

    def __init__(self,
                 focal_dice_kwargs,
                 ignore_label=None,
                 dice_class=SoftDiceLoss):
        """
        Weights for CE and Dice do not need to sum to one. You can set whatever you want.
        :param soft_dice_kwargs:
        :param ce_kwargs:
        :param aggregate:
        :param square_dice:
        :param weight_ce:
        :param weight_dice:
        """
        super().__init__()

        # if ignore_label is not None:
        #     ce_kwargs['ignore_index'] = ignore_label
        #
        # self.weight_dice = weight_dice
        # self.weight_focal = weight_focal
        self.ignore_label = ignore_label

        self.dc_fl = DiceFocalLoss(**focal_dice_kwargs)
        # self.dc = dice_class(apply_nonlin=softmax_helper_dim1, **focal_dice_kwargs)

class DC_loss(nn.Module):

    def __init__(self,
                 soft_dice_kwargs,
                 weight_dice=1,
                 ignore_label=None,
                 dice_class=SoftDiceLoss):
        """
        Weights for CE and Dice do not need to sum to one. You can set whatever you want.
        :param soft_dice_kwargs:
        :param ce_kwargs:
        :param aggregate:
        :param square_dice:
        :param weight_ce:
        :param weight_dice:
        """
        super().__init__()

        # if ignore_label is not None:
        #     ce_kwargs['ignore_index'] = ignore_label

        self.weight_dice = weight_dice
        self.ignore_label = ignore_label

        self.dc = dice_class(apply_nonlin=softmax_helper_dim1, **soft_dice_kwargs)

    def forward(self, net_output: torch.Tensor, target: torch.Tensor):
        """
        target must be b, c, x, y(, z) with c=1
        :param net_output:
        :param target:
        :return:
        """
        if self.ignore_label is not None:
            assert target.shape[1] == 1, 'ignore label is not implemented for one hot encoded target variables ' \
                                         '(DC_and_CE_loss)'
            mask = target != self.ignore_label
            # remove ignore label from target, replace with one of the known labels. It doesn't matter because we
            # ignore gradients in those areas anyway
            target_dice = torch.where(mask, target, 0)
            num_fg = mask.sum()
        else:
            target_dice = target
            mask = None
        dc_loss = self.dc(net_output, target_dice, loss_mask=mask) \
            if self.weight_dice != 0 else 0

        result = self.weight_dice * dc_loss
        return result









