#!/usr/bin/env python
# coding: utf-8

# In[42]:


from torch.utils.data import Dataset, DataLoader
import torch
import torchvision.transforms as transforms
import os
import numpy as np
from PIL import Image
import cv2


# In[43]:


class MyDataSet(Dataset):#定义数据读取类
    def __init__(self, basepath, labels=None, transform=None):#输入参数：数据集路径,和转化方法
        self.basepath = basepath
        self.transforms = transform
        self.classes = sorted(os.listdir(basepath))
        # self.classes.remove('.DS_Store')#非mac系统不需要这句
        self.filelist = []
        self.dataset = []
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
        
        self.getfile()
            
    def __len__(self):#统计数据集样本总数
        return self.total
    
    def __getitem__(self, index):
        return self.dataset[index]
    
    def getfile(self):#获取训练数据
        classesname = sorted(self.classes)
        for i,Set in enumerate(self.filelist):#逐类读取
            for j,file in enumerate(Set):#在每个类中逐个读取样本
                image = np.array(cv2.imread(os.path.join(self.basepath,classesname[i],file)))#读取图像  
                image = Image.fromarray(image.astype('uint8')).convert('RGB')#转化为PIL图像
                image = self.transforms(image)#预处理
                self.dataset.append([image,torch.tensor(self.class_to_idx[classesname[i]])])#载入图像和标签
        # return self.dataset


# In[44]:


if __name__ == '__main__':
    data_transform = transforms.Compose([
        transforms.Resize([112,112]),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0,0,0),std=(1,1,1))]
    )
    train_path = './Birds/train/'
    #获取训练测试样本
    trainset = MyDataset(train_path, transform=data_transform, labels=None)
    traindata = DataLoader(trainset,batch_size=4,shuffle=True)


# In[ ]:




