#!/usr/bin/env python
# coding: utf-8

# In[3]:


import torch.nn as nn


# In[4]:


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential( # 输入数据大小 (3,96,96)
            nn.Conv2d(
                in_channels=3, # 输入的图片层数
                out_channels=16, # 卷积核数量
                kernel_size=5, # 卷积核大小
                stride=1,
                padding=2,
            ), # 输出数据大小 (16,96,96)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.conv2 = nn.Sequential( # 输入数据大小 (16,48,48)
            nn.Conv2d(16, 32, 5, 1, 2), # 输出数据大小 (32,48,48)
            nn.ReLU(),
            nn.MaxPool2d(2), # 输出数据大小 (32,24,24)
        )
        self.fc1 = nn.Linear(32*24*24, 32*24) # 全连接层
        self.out = nn.Linear(32*24, 10)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        xx = x.view(x.size(0), -1) # 展平多维的卷积图
        x = self.fc1(xx)
        output = self.out(x)
        return output, xx


# In[ ]:




