# -*- coding: utf-8 -*-
# @Time    : 2023/7/31 下午2:34
# @Author  : Tang
# @File    : model.py
# @Software: PyCharm
import torch.nn as nn
import torchsummary


class ResiualBlock(nn.Module):
    """
    Define the residual block.
    """

    def __init__(self, in_channels, out_channels, stride=1):
        """
        Define the building blocks of the residual block.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            stride (int): Stride value for the first convolutional layer.
                            Default is 1.
        """
        super(ResiualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels,
                               out_channels=out_channels,
                               kernel_size=3,
                               stride=stride,
                               padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels=out_channels,
                               out_channels=out_channels,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels=in_channels,
                          out_channels=out_channels,
                          kernel_size=1,
                          stride=stride,
                          bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        """
        Forward pass function of the residual block.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        residual = x
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        if self.downsample:
            residual = self.downsample(residual)
        x += residual
        x = self.relu(x)
        return x

class ResNet(nn.Module):
    """
    Define the ResNet network.
    """

    def __init__(self, block, num_classes=10):
        """
        Define the building blocks of the ResNet network.

        Args:
            block (ResidualBlock): The residual block.
            num_classes (int): Number of classes. Default is 10.
        """
        super(ResNet, self).__init__()
        self.in_channels = 16
        self.conv = nn.Conv2d(in_channels=3,
                              out_channels=16,
                              kernel_size=3,
                              stride=1,
                              padding=1,
                              bias=False)
        self.bn = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self.make_layer(block=block,
                                      out_channels=16,
                                      num_blocks=3,
                                      stride=1)
        self.layer2 = self.make_layer(block=block,
                                      out_channels=32,
                                      num_blocks=3,
                                      stride=2)
        self.layer3 = self.make_layer(block=block,
                                      out_channels=64,
                                      num_blocks=3,
                                      stride=2)
        self.avg_pool = nn.AvgPool2d(kernel_size=8)
        self.fc = nn.Linear(in_features=64,
                            out_features=num_classes)

    def make_layer(self, block, out_channels, num_blocks, stride):
        """
        Define the residual blocks.

        Args:
            block (ResidualBlock): The residual block.
            out_channels (int): Number of output channels.
            num_blocks (int): Number of residual blocks.
            stride (int): Stride value for the first convolutional layer.

        Returns:
            nn.Sequential: A sequential layer of residual blocks.
        """
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(in_channels=self.in_channels,
                                out_channels=out_channels,
                                stride=stride))
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        """
        Forward pass function of the ResNet network.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        x = self.relu(self.bn(self.conv(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# print the network structure
resnet = ResNet(ResiualBlock).cuda()
torchsummary.summary(resnet, (3, 32, 32))