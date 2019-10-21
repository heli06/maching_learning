# encoding = utf-8

import torch.nn as nn
import torch.nn.functional as F


# 定义CNN
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)               # 卷积层
        self.conv2 = nn.Conv2d(6, 16, 5)

        self.pool = nn.MaxPool2d(2, 2)                # 池化层

        self.Linear1 = nn.Linear(16 * 5 * 5, 120)     # 全连接层
        self.Linear2 = nn.Linear(120, 84)
        self.Linear3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)

        x = F.relu(self.Linear1(x))
        x = F.relu(self.Linear2(x))
        x = self.Linear3(x)
        return x
