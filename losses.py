# Deep Learning 2023 course, held by Professor Paolo Frasconi - University of Florence, Italy
# Created by Giovanni Colombo - Mat. 7092745
# Dedicated Repository on GitHub at https://github.com/giovancombo/ProteinSecondaryStructurePrediction

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, alpha = 1, gamma = 2, reduction = 'mean', ignore_index = -100):
        super().__init__()
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


class CombinedLoss(nn.Module):
    def __init__(self, alpha = 1, gamma = 2, ce_weight = 0.5, focal_weight = 0.5, reduction = 'mean', ignore_index = -100):
        super().__init__()

        self.focal_loss = FocalLoss(alpha, gamma, reduction, ignore_index)
        self.ce_loss = nn.CrossEntropyLoss(reduction = reduction, ignore_index = ignore_index)
        self.ce_weight = ce_weight
        self.focal_weight = focal_weight

    def forward(self, inputs, targets):
        focal = self.focal_loss(inputs, targets)
        ce = self.ce_loss(inputs, targets)
        return self.ce_weight * ce + self.focal_weight * focal
