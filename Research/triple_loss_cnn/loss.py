# encoding = utf-8

import torch.nn as nn
import torch.nn.functional as F


# 三元损失
class TripletLoss(nn.Module):

    def __init__(self, margin):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative, size_average=True):
        distance_positive = (anchor-positive).pow(2).sum(1)
        distance_negative = (anchor-negative).pow(2).sum(1)
        losses = F.relu(distance_positive - distance_negative + self.margin)
        return losses.mean() if size_average else losses.sum()

