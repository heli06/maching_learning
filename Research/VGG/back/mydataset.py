#!/usr/bin/env python
# coding: utf-8

# In[9]:


from torch.utils.data import Dataset, DataLoader
import torch
import torchvision.transforms as transforms
import os
import numpy as np
from PIL import Image
import cv2


# In[60]:


class MyDataSet(Dataset):#定义数据读取类
    def __init__(self, basepath, labels=None, transform=None):#输入参数：数据集路径,和转化方法
        self.basepath = basepath
        self.transforms = transform
        self.classes = sorted(os.listdir(basepath))
        # self.classes.remove('.DS_Store')#非mac系统不需要这句
        self.filelist = []
        self.dataset = []
        self.classset = []
        self.total = 0
        for idx,Set in enumerate(sorted(self.classes)):#按类载入所有训练数据的文件名
            files = os.listdir(os.path.join(basepath,Set))
            # files.remove('.DS_Store')#非mac系统不需要这句
            self.filelist.append(files)
            self.total += len(files)
            
        self.num_classes = len(self.classes)#类别数
        self.class_to_idx = dict()#类别对应标签
        for i,classes in enumerate(self.classes):
            self.class_to_idx[classes] = i
            self.classset.append([])
        
        self.getfile()
            
    def __len__(self):#统计数据集样本总数
        return self.total
    
    def __getitem__(self, index):
        anchor_class = 0
        count = 0
        # 寻找index所指样本的类别
        for i in range(self.num_classes+1):
            if count <= index:
                count += len(self.classset[i])
            else:
                anchor_class = i-1
                break
                
                
        anchor_index, positive_index = np.random.choice(len(self.classset[anchor_class]), 2, replace=False)
        anchor = self.classset[anchor_class][anchor_index]
        positive = self.classset[anchor_class][positive_index]
        
        negative_class = anchor_class
        # 随机出一个别的类别的样本作为负样本
        while (negative_class == anchor_class):
            negative_class = np.random.choice(len(self.classset), 1, replace=False)[0]
            
        negative_index = np.random.choice(len(self.classset[negative_class]), 1, replace=False)[0]
        negative = self.classset[negative_class][negative_index]
        return (anchor, positive, negative)
        #return self.dataset[index]
    
    def getfile(self):#获取训练数据
        classesname = sorted(self.classes)
        for i,Set in enumerate(self.filelist):#逐类读取
            for j,file in enumerate(Set):#在每个类中逐个读取样本
                image = np.array(cv2.imread(os.path.join(self.basepath,classesname[i],file)))#读取图像  
                image = Image.fromarray(image.astype('uint8')).convert('RGB')#转化为PIL图像
                image = self.transforms(image)#预处理
                self.classset[i].append([image,torch.tensor(self.class_to_idx[classesname[i]])])
                #self.dataset.extend([image,torch.tensor(self.class_to_idx[classesname[i]])])#载入图像和标签
                
        for i in range(self.num_classes):
            self.dataset.extend(self.classset[i])#载入图像和标签
        # return self.dataset


# In[61]:


# 往下都是测试代码


# In[64]:


if __name__ == '__main__':
    data_transform = transforms.Compose([
        transforms.Resize([112,112]),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0,0,0),std=(1,1,1))]
    )
    train_path = './Birds/train/'
    #获取训练测试样本
    trainset = MyDataSet(train_path, transform=data_transform, labels=None)
    #print(trainset.__getitem__(0))
    #print(trainset.__getitem__(50))
    #print(trainset.__getitem__(100))
    #traindata = DataLoader(trainset,batch_size=4,shuffle=True)
    for i in range(0, trainset.__len__(), 20):
        anchor, positive, negative = trainset.__getitem__(i)
        print(anchor[1], i)
        print('-------------------------')


# In[ ]:




