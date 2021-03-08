#!/usr/bin/env python
# coding: utf-8

import torch
from torch import nn
from torch.nn import functional as F

from ..layers._3d.double_convolution import DoubleConvolution

class Unet(nn.Module):
    def __init__(
            self,
            depth: int = 4,
            in_channels: int = 1,
            channel_fan: int = 2,
            kernel_size: int = 3,
            padding: int = 1,
            pooling: str = 'average',
            # dimensions: tuple, # =(176, 104, 52),
        ):
        '''
        Implementation of the Unet network architecture https://arxiv.org/pdf/1505.04597.pdf
        Just one level for now TODO: add more levels : )

        Parameters:
            pooling: options are (average, max)
        '''

        super().__init__()
        self.network_name = "Unet"
        
        current_depth = depth
        n_in, n_out = in_channels, in_channels * channel_fan
        subnetworks = []
        for current_depth in range(depth + 1):
            if current_depth == 0:
                # we are at the bottom
                subnetworks.append(DoubleConvolution(n_in, n_out))
            else:
                down = DoubleConvolution(n_in, n_out, kernel_size=kernel_size, padding=padding)
                subnetwork =
            n_in = n_out
            n_out *= channel_fan

        out_channels = in_channels * channel_fan
            
        self.step_down = DoubleConvolution(in_channels, out_channels, kernel_size=kernel_size, padding=padding)
        self.step_bottom = DoubleConvolution(out_channels, out_channels * channel_fan, kernel_size=kernel_size, padding=padding)
        self.step_up = DoubleConvolution(out_channels * channel_fan, out_channels, kernel_size=kernel_size, padding=padding)

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
