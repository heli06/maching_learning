# encoding = utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torchvision
from torchvision import transforms
from torchvision.datasets import ImageFolder
import matplotlib.pyplot as plt                            # 显示图片
from PIL import Image                                      # 读取图片

# 将本地数据划分为训练集和测试集
normalize = transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
# 因为全连接层是固定尺寸的输入输出，所以在卷积层之前的输入要求是固定的，CIFAR10 32x32
train_transformer = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.CenterCrop(32),
    transforms.ToTensor(),
    normalize
])

test_transformer = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.CenterCrop(32),
    transforms.ToTensor(),
    normalize
])


class MyDataset(Dataset):
    # 参数预定义
    def __init__(self, filenames, labels, transform):
        self.filenames = filenames
        self.labels = labels
        self.transform = transform

    # 返回图片个数
    def __len__(self):
        return len(self.filenames)

    # 获取每个图片
    def __getitem__(self, idx):
        image = Image.open(self.filenames[idx]).convert('RGB')
        image = self.transform(image)
        return image, self.labels[idx]


def split_train_test_data(data_dir, ratio):
    dataset = ImageFolder(data_dir)                        # ratio的和为1，data_dir精确到分类目录的上一级
    character = [[] for i in range(len(dataset.classes))]
    for x, y in dataset.samples:                           # 将数据按类标签存放
        character[y].append(x)
    # print(dataset.samples)

    train_inputs, test_inputs = [], []
    train_labels, test_labels = [], []
    for i, data in enumerate(character):                   # data为一类图片
        num_sample_train = int(len(data) * ratio[0])
        num_sample_test = int(len(data) * ratio[1])

        num_test_index = num_sample_train + num_sample_test

        for x in data[:num_sample_train]:
            train_inputs.append(str(x))
            train_labels.append(i)
        for x in data[num_sample_train:num_test_index]:
            test_inputs.append(str(x))
            test_labels.append(i)

    train_loader = DataLoader(MyDataset(train_inputs, train_labels, train_transformer),
                              batch_size=100, shuffle=True)   # batch_size 一次性读入多少批量的图片
    test_loader = DataLoader(MyDataset(test_inputs, test_labels, test_transformer),
                             batch_size=100, shuffle=False)

    return train_loader, test_loader


# 划分训练集和测试集
train_loader, test_loader = split_train_test_data('E:\\datasets\\Birds', [0.8, 0.2])


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


# 定义10个类的标签名称
classes = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')

# training CNN
net = Net()
# 定义损失函数和优化器
optimizer = optim.SGD(net.parameters(), lr=0.001, weight_decay=1e-6, momentum=0.9)  # 学习率0.001
loss_function = nn.CrossEntropyLoss()           # 用交叉熵损失函数
# 30 epochs
for epoch in range(30):
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):   # for data in test_loader:
        inputs, labels = data
        inputs, labels = Variable(inputs), Variable(labels)  # 转换数据格式为Variable
        optimizer.zero_grad()                   # 梯度置零，因为反向传播过程中梯度会累加上一次循环的梯度

        output = net(inputs)
        loss = loss_function(input=output, target=labels)    # 损失计算 loss = loss_function(output, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()            # 损失累加
        if i % 100 == 99:
            print("[%d %5d] loss:%.3f" % (epoch + 1, i + 1, running_loss / 200))  # 计算100次的平均损失
            running_loss = 0.0

# testing CNN
dataiter = iter(test_loader)
images, labels = dataiter.next()

img = torchvision.utils.make_grid(images, nrow=10)     # 让每行显示10张图片
img = img.numpy().transpose(1, 2, 0)  # 格式转换
img = img / 2 + 0.5    # img = img * std + mean
plt.imshow(img)

correct = 0
total = 0
for data in test_loader:
    outputs = net(Variable(images))
    _, predicts = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicts == labels).sum()
# 正确率
print("Accuracy: %.6f " % (correct / total * 100))
# 真实值
print("Actual:", "".join("%5s" % classes[labels[j]] for j in range(100)))      # 打印前100张图片的标签名称
# 预测值
print("Predict:", "".join("%5s" % classes[predicts[j]] for j in range(100)))

plt.show()
