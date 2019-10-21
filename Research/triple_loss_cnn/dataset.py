# encoding = utf-8

import torch
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
import os
import cv2

torch.manual_seed(10)  # 使实验结果确定


class MyDataset(Dataset):
    # 参数预定义
    def __init__(self, filepath, transforms):
        self.filepath = filepath
        self.transforms = transforms
        self.classes = sorted(os.listdir(filepath))    # 读取根目录下的类名，列表
        self.num_classes = len(self.classes)           # 类的数量

        # 每个类下的文件名列表 filelist
        self.filelist = []
        self.total = 0
        for i, classes in enumerate(sorted(self.classes)):
            files = os.listdir(os.path.join(filepath, classes))  # 每个类下的文件名
            self.filelist.append(files)
            self.total += len(files)

        # 给类建的索引，类名-类索引
        self.class_to_idx = dict()
        for i, classes in enumerate(self.classes):
            self.class_to_idx[classes] = i

        # 获取图像集和标签集
        classesname = sorted(self.classes)             # 类名列表
        imgPath_set = []
        classLabels_set = []
        for i, imgName in enumerate(self.filelist):
            for j, file in enumerate(imgName):
                img_path = os.path.join(self.filepath, classesname[i], file)
                imgPath_set.append(img_path)
                classLabels_set.append(self.class_to_idx[classesname[i]])
        self.imgPath_set = imgPath_set                 # 图片完整路径列表
        self.classLabels_set = classLabels_set         # 类名索引集

        # 给相同类标签下的图像建的路径索引，字典
        self.label_to_idx = {label: np.where(np.array(self.classLabels_set) == np.array(label))[0]
                             for label in self.classLabels_set}

    def __len__(self):
        return len(self.imgPath_set)                                          # 数据集样本总数

    def __getitem__(self, index):
        img1, label1 = self.imgPath_set[index], self.classLabels_set[index]   # label1是img1的类名

        positive_index = index
        while positive_index == index:
            positive_index = np.random.choice(self.label_to_idx[label1])      # 正例路径索引

        negative_label = np.random.choice(list(set(self.classLabels_set) - set([label1])))  # 负例的类名
        negative_index = np.random.choice(self.label_to_idx[negative_label])  # 负例的路径索引

        imgp = self.imgPath_set[positive_index]                               # 取正例路径
        imgn = self.imgPath_set[negative_index]                               # 取负例路径

        img1 = np.array(cv2.imread(img1))
        imgp = np.array(cv2.imread(imgp))
        imgn = np.array(cv2.imread(imgn))

        img1 = Image.fromarray(img1.astype('uint8')).convert('RGB')           # 转化为PIL图像
        imgp = Image.fromarray(imgp.astype('uint8')).convert('RGB')
        imgn = Image.fromarray(imgn.astype('uint8')).convert('RGB')

        # 预处理
        if self.transforms is not None:
            img1 = self.transforms(img1)
            imgp = self.transforms(imgp)
            imgn = self.transforms(imgn)

        return img1, imgp, imgn, torch.tensor(label1)




