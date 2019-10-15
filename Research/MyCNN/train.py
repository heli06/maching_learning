#!/usr/bin/env python
# coding: utf-8

# In[1]:


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


# In[5]:


print(train_set.classes)
print(train_set.class_to_idx)
print(train_set.__len__)

print(test_set.classes)
print(test_set.class_to_idx)
print(test_set.__len__)


# In[6]:


cnn = CNN()
print(cnn)


# In[7]:


optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)
loss_func = nn.CrossEntropyLoss() # 选择损失函数

for epoch in range(EPOCH):
    print('EPOCH ' + str(epoch))
    for step, (b_x, b_y) in enumerate(train_loader):
        output = cnn(b_x)[0]
        loss = loss_func(output, b_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if step % 50 == 0:
            correct = 0
            total = 0
            
            for data in test_loader:
                images,labels = data
                outputs = cnn(Variable(images))
                predicted = torch.max(outputs[0], 1).indices
                total += labels.size(0)
                correct += (predicted == labels).sum()

            print('Accuracy of the network on the test images: %d %%' % (100 * correct / total))
            
            
print('End')


# In[ ]:




