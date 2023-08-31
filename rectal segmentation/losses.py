import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

try:
    from LovaszSoftmax.pytorch.lovasz_losses import lovasz_hinge
except ImportError:
    pass


class BCEDiceLoss(nn.Module):
    def __init__(self):
        super(BCEDiceLoss, self).__init__()
    # 二分类问题的交叉熵损失函数、、、
    def forward(self, input, target):
        bce = F.binary_cross_entropy_with_logits(input, target)
        smooth = 1e-5   # 防止0除
        input = torch.sigmoid(input)
        num = target.size(0)
        input = input.view(num, -1)
        target = target.view(num, -1)
        intersection = (input * target)
        dice = (2. * intersection.sum(1) + smooth) / (input.sum(1) + target.sum(1) + smooth)
        dice = 1 - dice.sum() / num
        return 0.5 * bce + dice
        # return 0.5*bce

class LovaszHingeLoss(nn.Module):
    def __init__(self):
        super(LovaszHingeLoss, self).__init__()

    def forward(self, input, target):
        input = input.squeeze(1)
        target = target.squeeze(1)
        loss = lovasz_hinge(input, target, per_image=True)

        return loss


# Generalized Dice loss
def generalized_dice_coeff(target, input):
    # target shape=[num_label,H,W,C]
    smooth = 1e-5
    num = input.shape[0]
    w = torch.zeros(shape=(num,))
    w = torch.sum(target, axis=(1, 2, 3))
    w = 1 / (w ** 2 + 0.000001)
    # Compute gen dice coef:
    # 在dice loss基础上增加了w给每个类别加权
    intersection = w * torch.sum(target * input, axis=[1, 2, 3])
    union = w * torch.sum(target + input, axis=[1, 2, 3])
    return torch.mean((2. * intersection + smooth) / (union + smooth), axis=0)

def GENdiceloss(target, input):
    return 1 - generalized_dice_coeff(target, input)