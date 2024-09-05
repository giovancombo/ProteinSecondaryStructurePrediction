# Deep Learning 2023 course, held by Professor Paolo Frasconi - University of Florence, Italy
# Created by Giovanni Colombo - Mat. 7092745
# Dedicated Repository on GitHub at https://github.com/giovancombo/ProteinSecondaryStructurePrediction

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, alpha = 1, gamma = 2, reduction = 'mean', ignore_index = -100):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.ignore_index = ignore_index

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction = 'none', ignore_index = self.ignore_index)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt)**self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss[targets != self.ignore_index].mean()
        elif self.reduction == 'sum':
            return focal_loss[targets != self.ignore_index].sum()
        else:
            return focal_loss
