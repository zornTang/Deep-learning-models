# -*- encoding: utf-8 -*-
#@Time    :   2023/07/24 15:56:06
#@Author  :   Tang
#@File    :   test.py
#@Software:   Pycharm
import torch.nn as nn
import torchsummary


class Net(nn.Module):
    def __init__(self):
        """
        Define the network architecture.
        """
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3,
                               out_channels=6,
                               kernel_size=5,
                               stride=1,
                               padding=0)
        self.conv2 = nn.Conv2d(in_channels=6,
                               out_channels=16,
                               kernel_size=5,
                               stride=1,
                               padding=0)
        self.pool = nn.MaxPool2d(kernel_size=2,
                                 stride=2,
                                 padding=0)
        self.fc1 = nn.Linear(in_features=16 * 5 * 5,
                             out_features=120)
        self.fc2 = nn.Linear(in_features=120,
                             out_features=84)
        self.fc3 = nn.Linear(in_features=84,
                             out_features=10)

    def forward(self, x):
        """
        Forward pass function of the network.
        """
        x = self.pool(nn.functional.relu(self.conv1(x)))
        x = self.pool(nn.functional.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# print model summary
model = Net().cuda()
torchsummary.summary(model, (3, 32, 32))
