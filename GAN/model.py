# -*- coding: utf-8 -*-
# @Time    : 2023/7/12 22:01
# @Author  : Tang
# @File    : model.py
# @Software: PyCharm
import numpy as np
import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, input_dim, output_shape):
        """
        Define the generator model structure.

        Args:
            input_dim (int): The dimension of the input noise vector.
            output_shape (tuple): The shape of the generator output.
                                  It represents the shape of the generated image.
                                  Example: (3, 64, 64) for RGB images of size 64x64.
        """
        super(Generator, self).__init__()
        self.output_shape = output_shape

        def block(input, output, normalize=True):
            """
            Helper function to create a block of layers for the generator model.

            Args:
                input (int): Number of input channels/neurons.
                output (int): Number of output channels/neurons.
                normalize (bool): Whether to apply batch normalization.
                                  Default is True.

            Returns:
                list: List of layers for the block.
            """
            layers = [nn.Linear(input, output)]
            if normalize:
                layers.append(nn.BatchNorm1d(output))
            layers.append(nn.ReLU())

            return layers

        # Generator model architecture
        self.model = nn.Sequential(
            *block(input_dim, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(np.prod(self.output_shape))),  # Output layer
            nn.Tanh()  # To map output values between -1 and 1
        )

    def forward(self, input):
        """
        Forward pass function of the generator.

        Args:
            input (torch.Tensor): The input noise tensor.
                                  Shape: (batch_size, input_dim)

        Returns:
            torch.Tensor: The generated image tensor.
                          Shape: (batch_size, *output_shape)
        """
        image = self.model(input)
        image = image.view(image.size(0), *self.output_shape)

        return image


class Discriminator(nn.Module):
    def __init__(self, image_shape):
        """
        Define the discriminator model structure.

        Args:
            image_shape (tuple): The shape of the discriminator input.
                                 It represents the shape of the image.
                                 Example: (3, 64, 64) for RGB images of size 64x64.
        """
        super(Discriminator, self).__init__()

        self.image_shape = image_shape

        # Discriminator model architecture
        self.block1 = nn.Sequential(
            nn.Linear(int(np.prod(self.image_shape)), 512),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.block2 = nn.Sequential(
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.block3 = nn.Sequential(
            nn.Linear(256, 1),
            nn.Sigmoid()  # To produce the probability of being real/fake
        )

    def forward(self, image):
        """
        Forward pass function of the discriminator.

        Args:
            image (torch.Tensor): The input image tensor.
                                  Shape: (batch_size, *image_shape)

        Returns:
            torch.Tensor: The discriminator's prediction on the input image.
                          Shape: (batch_size, 1)
        """
        image_flatten = image.view(image.size(0), -1)
        pred = self.block1(image_flatten)
        pred = self.block2(pred)
        pred = self.block3(pred)

        return pred

# test model input and output
# from torchsummary import summary
# Generator = Generator(100, (1 ,28, 28)).cuda()
# Discriminator = Discriminator((1, 28, 28)).cuda()
# summary(Generator, (100, ))
# summary(Discriminator, (784, ))
