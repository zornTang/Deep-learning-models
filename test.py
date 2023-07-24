# -*- coding: utf-8 -*-
#@Time    : 2023/07/18 22:25:55
#@Author  : Tang
#@File    : test.py
#@Software: VScode
import torch
import torch.nn as nn
import torchsummary as summary


# 定义残差块
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        """
        定义残差块
        :param in_channels: 输入通道数
        :param out_channels: 输出通道数
        :param stride: 步长
        """

        super(ResidualBlock,self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False) # 第一层卷积
        self.bn1 = nn.BatchNorm2d(out_channels) # 第一层BN
        self.relu = nn.ReLU(inplace=True) # 激活函数
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False) # 第二层卷积
        self.bn2 = nn.BatchNorm2d(out_channels) # 第二层BN
        self.downsample = nn.Sequential() # 下采样
        if stride != 1 or in_channels != out_channels: # 如果stride不为1或者in_channels不等于out_channels
            self.downsample = nn.Sequential( # 下采样
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False), # 卷积
                nn.BatchNorm2d(out_channels) # BN
            )

    def forward(self, x):
        """
        前向传播
        :param x: 输入
        :return: 输出
        """

        identity = self.downsample(x) # 下采样
        out = self.conv1(x) # 第一层卷积
        out = self.bn1(out) # 第一层BN
        out = self.relu(out) # 激活函数
        out = self.conv2(out) # 第二层卷积
        out = self.bn2(out) # 第二层BN
        out += identity # 残差连接
        out = self.relu(out) # 激活函数
        return out

class ResNet(nn.Module):
    def __init__(self, block, num_classes=10):
        """
        定义ResNet
        :param block: 残差块
        :param num_classes: 分类数
        """

        super(ResNet,self).__init__()
        self.in_channels = 16 # 输入通道数
        self.conv = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False) # 第一层卷积
        self.bn = nn.BatchNorm2d(16) # 第一层BN
        self.relu = nn.ReLU(inplace=True) # 激活函数
        self.layer1 = self.make_layer(block, 16, 2, stride=1) # 第一层残差块
        self.layer2 = self.make_layer(block, 32, 2, stride=2) # 第二层残差块
        self.layer3 = self.make_layer(block, 64, 2, stride=2) # 第三层残差块
        self.avg_pool = nn.AvgPool2d(8) # 平均池化
        self.fc = nn.Linear(64, num_classes) # 全连接层

    def make_layer(self, block, out_channels, num_blocks, stride):
        """
        定义残差层
        :param block: 残差块
        :param out_channels: 输出通道数
        :param num_blocks: 残差块的数量
        :param stride: 步长
        :return: 残差层
        """

        strides = [stride] + [1] * (num_blocks - 1) # 步长
        layers = [] # 残差层
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride)) # 添加残差块
            self.in_channels = out_channels # 更新输入通道数
        return nn.Sequential(*layers) # 返回残差层

    def forward(self, x):
        """
        前向传播
        :param x: 输入
        :return: 输出
        """

        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        out = self.layer1(out) # 第一层残差层
        out = self.layer2(out) # 第二层残差层
        out = self.layer3(out) # 第三层残差层
        out = self.avg_pool(out) # 平均池化
        out = out.view(out.size(0), -1) # 展平
        out = self.fc(out) # 全连接层
        
        return out

# 打印模型结构
model = ResNet(ResidualBlock)
summary.summary(model, (3, 32, 32), device='cpu')
