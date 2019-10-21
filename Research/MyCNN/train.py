#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as Data
import matplotlib.pyplot as plt
import torchvision
import torchvision.datasets
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.autograd import Variable
from net import CNN
from mydataset import MyDataSet
from myloss import TripletLoss

torch.manual_seed(10)


# In[2]:


EPOCH = 25
BATCH_SIZE = 4
LR = 0.001


# In[3]:


transform = transforms.Compose([   
        transforms.Resize([96, 96]),  
        transforms.ToTensor(),  
        transforms.Normalize(mean=(0,0,0),std=(1,1,1))
])

train_set = MyDataSet('./birds/train/',transform=transform, labels=None)
train_loader = Data.DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
test_set = MyDataSet('./birds/test/', transform=transform, labels=None)
test_loader = Data.DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)


# In[4]:


print(train_set.classes)
print(train_set.class_to_idx)
print(train_set.__len__)

print(test_set.classes)
print(test_set.class_to_idx)
print(test_set.__len__)


# In[5]:


cnn = CNN()
print(cnn)


# In[6]:


optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)
cross_loss = nn.CrossEntropyLoss()   # the target label is not one-hotted
triplet_loss = TripletLoss(0.5) # 选择损失函数
alpha = 0.5

for epoch in range(EPOCH):
    print('EPOCH ' + str(epoch))
    # for step, (b_x, b_y) in enumerate(train_loader):
    for step, (anchor, positive, negative) in enumerate(train_loader):
        #output = cnn(b_x)[0]
        #loss = loss_func(output, b_y)
        anchor_output = cnn(anchor[0])
        positive_output = cnn(positive[0])
        negative_output = cnn(negative[0])
        
        #print(anchor_output.detach().numpy().shape)   
        #output = torch.stack((anchor_output[0],  positive_output[0], negative_output[0]), dim=0)
        #b_y = torch.stack((anchor[1], positive[1], negative[1]), dim=0)
        #print(output)
        #print(anchor_output[0])
        #b_y = torch.zeros(4, 10).scatter_(1, anchor[1].unsqueeze(1), 1).long()
        #print(b_y)
        
        
        triplet_result = triplet_loss(anchor_output[1], positive_output[1], negative_output[1])
        cross_result = cross_loss(anchor_output[0], anchor[1])
        loss = alpha * triplet_result + (1-alpha) * cross_result
                                       
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if step % 50 == 0:
            print('训练三元损失：', triplet_result)
            print('训练交叉熵损失：', cross_result)
            
            correct = 0
            total = 0
            
            for _, (test, _, _) in enumerate(test_loader):
                test_output = cnn(test[0])[0]
                predicted = torch.max(test_output, 1).indices
                labels = test[1]
                total += labels.size(0)
                correct += (predicted == labels).sum()
                
            print('Accuracy on the test images: %d %%' % (100 * correct / total))
            
#             for data in test_loader:
#                 images,labels = data
#                 outputs = cnn(Variable(images))
#                 predicted = torch.max(outputs[0], 1).indices
#                 total += labels.size(0)
#                 correct += (predicted == labels).sum()
                
#             print('Accuracy of the network on the test images: %d %%' % (100 * correct / total))
print('End')


# In[ ]:




