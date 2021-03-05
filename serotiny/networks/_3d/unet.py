#!/usr/bin/env python
# coding: utf-8

import torch
from torch import nn
from torch.nn import functional as F

from ..layers._3d.double_convolution import DoubleConvolution

class Unet(nn.Module):
    def __init__(
            self,
            in_channels=1,
            channel_fan=2,
            dimensions=(176, 104, 52),
            pooling='average'):
        super().__init__()
        self.network_name = "Unet"
        
        out_channels = in_channels * channel_fan
            
        self.step_down = DoubleConvolution(in_channels, out_channels)
        self.step_bottom = DoubleConvolution(out_channels, out_channels * channel_fan)
        self.step_up = DoubleConvolution(out_channels * channel_fan, out_channels)

        if pooling == 'average':
            self.pooling = nn.AvgPool3d(kernel_size=channel_fan)
        elif pooling == 'max':
            self.pooling = nn.MaxPool3d(kernel_size=channel_fan)
        else:
            # use convolutional stride to pool 
            self.pooling = torch.nn.Conv3d(
                n_out_channels, n_out_channels, channel_fan, stride=channel_fan
            )

        self.bn0 = torch.nn.BatchNorm3d(out_channels)
        self.relu0 = torch.nn.ReLU()
        self.convt = torch.nn.ConvTranspose3d(
            channel_fan * out_channels,
            out_channels,
            kernel_size=channel_fan,
            stride=2
        )

        self.bn1 = torch.nn.BatchNorm3d(out_channels)
        self.relu1 = torch.nn.ReLU()

    def forward(self, x):
        x_down = self.step_down(x)
        x_pool = self.pooling(x_down)
        x_bn0 = self.bn0(x_pool)
        x_relu0 = self.relu0(x_bn0)
        x_bottom = self.step_bottom(x_relu0)
        x_convt = self.convt(x_bottom)
        x_bn1 = self.bn1(x_convt)
        x_relu1 = self.relu1(x_bn1)
        x_cat = torch.cat((x_down, x_relu1), 1)  # concatenate
        x_up = self.step_up(x_cat)

        return x_up
