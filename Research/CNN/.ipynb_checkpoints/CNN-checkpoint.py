{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [],
   "source": [
    "import torch.nn as nn\n",
    "import torch.nn.functional as F"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {},
   "outputs": [],
   "source": [
    "# 定义CNN\n",
    "class Net(nn.Module):\n",
    "    def __init__(self):\n",
    "        super(Net, self).__init__()\n",
    "        self.conv1 = nn.Conv2d(3, 6, 5)               # 卷积层\n",
    "        self.conv2 = nn.Conv2d(6, 16, 5)\n",
    "\n",
    "        self.pool = nn.MaxPool2d(2, 2)                # 池化层\n",
    "\n",
    "        self.Linear1 = nn.Linear(16 * 5 * 5, 120)     # 全连接层\n",
    "        self.Linear2 = nn.Linear(120, 84)\n",
    "        self.Linear3 = nn.Linear(84, 10)\n",
    "\n",
    "    def forward(self, x):\n",
    "        x = self.pool(F.relu(self.conv1(x)))\n",
    "        x = self.pool(F.relu(self.conv2(x)))\n",
    "        x = x.view(-1, 16 * 5 * 5)\n",
    "\n",
    "        x = F.relu(self.Linear1(x))\n",
    "        x = F.relu(self.Linear2(x))\n",
    "        x = self.Linear3(x)\n",
    "        return x"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.6.8"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
