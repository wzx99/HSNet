import torch
import torch.nn as nn
import torch.nn.functional as F
import os


class DistSegLoss(nn.Module):
    def __init__(self, k=5, gamma=0.25, balance=False, dice_scale=1, bce_scale=5):
        super(DistSegLoss, self).__init__()
        self.k = k
        self.gamma = gamma
        self.balance = balance
        self.dice_scale = dice_scale
        self.bce_scale = bce_scale

    def forward(self, pred, gt, dist, mask):
        '''
        pred gt mask: b,h,w
        dist : b,1,h,w
        '''
        dist = (1/dist)**self.gamma
        weight_dice = F.max_pool2d(dist, self.k, stride=1, padding=self.k//2)[:,0]
        weight_dice = weight_dice * mask
        if self.balance:
            sum_pos = dist[gt.unsqueeze(1)==1].sum()
            sum_neg = (gt==0).sum()
            dist[gt.unsqueeze(1)==0] = sum_pos/sum_neg
            weight_bce = F.max_pool2d(dist, self.k, stride=1, padding=self.k//2)[:,0]
            weight_bce = weight_bce * mask
        else:
            weight_bce = weight_dice

        loss_bce = F.binary_cross_entropy(pred, gt, reduction='none')
        loss1 = (loss_bce * weight_bce).sum() / (weight_bce.sum() + 1e-8)
        loss1 = torch.mean(loss1)
        
        intersection = (pred * gt * weight_dice).sum()
        union = (pred * weight_dice).sum() + (gt * weight_dice).sum() + 1e-6
        loss2 = 1 - 2.0 * intersection / union

        return self.bce_scale*loss1+self.dice_scale*loss2
