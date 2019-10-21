# encoding = utf-8

import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms

from dataset import MyDataset
from net import Net
from loss import TripletLoss

# 归一化
normalize = transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
transforms = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.CenterCrop(32),
    transforms.ToTensor(),
    normalize
])

# 训练集和测试集
train_loader = DataLoader(MyDataset('E:\\DS\\train', transforms), batch_size=4, shuffle=True)
test_loader = DataLoader(MyDataset('E:\\DS\\test', transforms), batch_size=4, shuffle=False)

# 实例化一个网络
net = Net()

# 定义优化和损失函数
optimizer = optim.Adam(net.parameters(), lr=0.001)         # Adam 最好的优化函数
crossloss = nn.CrossEntropyLoss()                          # 交叉熵损失
tripletloss = TripletLoss(10.0)                          # 三元损失

# trainning CNN
# 将所有训练样本训练epoch遍, 测试样本测试epoch遍
for epoch in range(10):
    running_loss = 0.0
    for i, data in enumerate(train_loader):
        a, p, n, label_a = data
        a_outputs = net(a)
        p_outputs = net(p)
        n_outputs = net(n)
        optimizer.zero_grad()
        loss = crossloss(a_outputs, label_a)
        loss += tripletloss(a_outputs, p_outputs, n_outputs)

        # inputs, labels = data
        # inputs, labels = Variable(inputs), Variable(labels)  # 转换数据格式为Variable
        # optimizer.zero_grad()
        # output = net(inputs)
        # loss = loss_function(output, labels)    # 损失计算 loss = loss_function(output, labels)

        loss.backward()  # loss求导
        optimizer.step()  # 更新参数

        running_loss += loss.item()  # tensor.item()  获取tensor的数值,把tensor转换为数值
        if i % 20 == 19:
            print("[%d  %5d ] loss:%.3f" % (epoch + 1, i + 1, running_loss / 20))  # 计算每20次迭代，输出loss平均值
            running_loss = 0.0

    # testing CNN
    # dataiter = iter(test_loader)
    # a, _, _, label = dataiter.__next__()
    correct = 0
    total = 0
    for j, data in enumerate(test_loader, 0):
        a, p, n, labels = data
        outputs = net(a)
        _, predicts = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicts == labels).sum()
    # 计算正确率
    print("Accuracy: %.6f %%" % (100.0 * correct / total))


"""
margin : 10.0   23% 
         1.0    36%
         0.5    41%
         0.3    48%
         0      57%

"""