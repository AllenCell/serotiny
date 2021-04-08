#!/usr/bin/env python
# coding: utf-8

import torch
from torch import nn
#from torch.nn import functional as F

from ..layers._3d.unet_downconv import DownConvolution
from ..layers._3d.unet_upconv import UpConvolution


class Unet(nn.Module):
    def __init__(
        self,
        depth: int = 4,
        
        input_channels=None,
        output_channels=None,
        
        channel_fan_top: int = 64,        # Original paper = 64
        channel_fan: int = 2,             # Original paper = 2
        
        # Parameters for double convolution
        kernel_size_doubleconv: int = 3,  # Original paper = 3
        stride_doubleconv: int = 1,       # Original paper = no mention, label-free = default
        padding_doubleconv: int = 1,      # Original paper = 0, label-free = 1
        
        # Parameters for pooling
        pooling: str = "mean",            # Original paper = max
        kernel_size_pooling: int = 2,     # Original paper = 2
        stride_pooling: int = 2,          # Original paper = no mention, label-free = 2
        padding_pooling: int = 0,         # Original paper = no mention, label-free = default
        
        # Parameters for up convolution
        kernel_size_upconv: int = 2,      # Original paper = 2
        stride_upconv: int = 2,           # Original paper = no mention, label-free = 2
        padding_upconv: int = 0,          # Original paper = no mention, label-free = default
        
        # Parameters for final convolution
        kernel_size_finalconv: int = 3,   # Original paper = 1, label-free = 3
        stride_finalconv: int = 1,        # Original paper = no mention, label-free = default
        padding_finalconv: int = 1,       # Original paper = no mention, label-free = 1
        
        # dimensions: tuple, # =(176, 104, 52),
        **kwargs
    ):

        """
        Implementation of the Unet network architecture https://arxiv.org/pdf/1505.04597.pdf
        Just one level for now TODO: add more levels : )

        In the original paper:
          Double convolution: k=3, p=0
          Pooling (max):      k=2, s=2
          Up convolution:     k=2
          Final convolution:  k=1

        Parameters:
            pooling: options are (average, max)

        TODO:
            - Add auto-padding so we don't see mismatches when performing the cross-branch
              concatenations -> auto-pad at first level depending on depth and other
              network parameters
              
            - Separate channel_fan between first layer and the next for double_conv (right
              now this calculation is entirely done in the unet.py)? Yes, separate into 2
              parameters
              
            - Separate kernel_size, stride, and padding for double_conv, pooling, up_conv,
              UpConv, DownConv, and conv_out? No, just follow original paper
        """

        super().__init__()

        self.network_name = "Unet"
        self.depth = depth
        self.channel_fan_top = channel_fan_top
        self.channel_fan = channel_fan

        self.input_channels = input_channels
        self.output_channels = output_channels

        """
        An example of n_in and n_out for a network of depth = 4:
        
          L4: 3 -> 6,     12 -> 6
          L3: 6 -> 12,    24 -> 12
          L2: 12 -> 24,   48 -> 24
          L1: 24 -> 48,   96 -> 48
          L0:       48 -> 96
          
        """

        # Create the down pathway by traversing the downward path, including the bottom-most layer

        # NOTE: To store nn modules, we need to use nn.ModuleDict{} instead of a regular dictionary,
        #       and the keys (which indicates the network levels) must be strings. Also, we cannot
        #       store non-module info in nn.ModuleDict{}, such as the n_in and n_out tuples, so we
        #       need to use separate channels_down{} and channels_up{} dictionaries for that purpose,
        #       and also use their keys (integers) to traverse up and down the network
        self.channels_down = {}
        self.networks_down = nn.ModuleDict({})

        for current_depth in range(depth, -1, -1):

            self.networks_down[str(current_depth)] = nn.ModuleDict({})

            # At the top layer
            if current_depth == depth:
                n_in = len(self.input_channels)
                n_out = channel_fan_top  # Similar to original paper and label-free, apply channel_fan_top only in the top layer

            else:
                n_in = self.channels_down[current_depth + 1][1]
                n_out = n_in * channel_fan  # Is hardcoded to 2 in original paper and label-free

            self.channels_down[current_depth] = (n_in, n_out)
            self.networks_down[str(current_depth)]["subnet"] = DownConvolution(
                current_depth,
                n_in,
                n_out,
                
                kernel_size_doubleconv=kernel_size_doubleconv,
                stride_doubleconv=stride_doubleconv,
                padding_doubleconv=padding_doubleconv,
                
                pooling=pooling,
                kernel_size_pooling=kernel_size_pooling,
                stride_pooling=stride_pooling,
                padding_pooling=padding_pooling,
            )

        # Create the up pathway by traversing the upward path

        self.channels_up = {}
        self.networks_up = nn.ModuleDict({})

        for current_depth in sorted(list(self.channels_down.keys())):

            if current_depth == 0:
                pass
            else:
                self.networks_up[str(current_depth)] = nn.ModuleDict({})

                n_in = self.channels_down[current_depth - 1][1]
                n_out = n_in // channel_fan  # Is hardcoded to 2 in original paper and label-free

                self.channels_up[current_depth] = (n_in, n_out)
                self.networks_up[str(current_depth)]["subnet"] = UpConvolution(
                    current_depth,
                    n_in,
                    n_out,
                    
                    kernel_size_upconv=kernel_size_upconv,
                    stride_upconv=stride_upconv,
                    padding_upconv=padding_upconv,
                    
                    kernel_size_doubleconv=kernel_size_doubleconv,
                    stride_doubleconv=stride_doubleconv,
                    padding_doubleconv=padding_doubleconv,
                )

        # In original paper, kernel_size = 1
        self.conv_out = nn.Conv3d(
            self.channels_up[depth][1],
            len(self.output_channels),
            
            kernel_size=kernel_size_finalconv,
            stride=stride_finalconv,
            padding=padding_finalconv,
        )

    def print_network(self):
        def print_subnet(network_dict, reverse=False):

            network_keys = sorted(network_dict.keys(), reverse=reverse)

            for level in network_keys:
                print(f"Level = {level}")
                # print(f'{network_dict[level]}')

                for key in network_dict[level].keys():
                    print(f"  {key} = {network_dict[level][key]}")

        print("networks_down:")
        print_subnet(self.networks_down, reverse=True)
        print()

        print("networks_up:")
        print_subnet(self.networks_up, reverse=False)
        print()

        print("conv_out:")
        print(self.conv_out)
        print()

    def forward(self, x):

        network_layers_down = sorted(self.channels_down.keys(), reverse=True)
        network_layers_up = sorted(self.channels_up.keys(), reverse=False)

        x_previous_layer = x
        doubleconv_down_out = {}
        
        for current_depth in network_layers_down:
            #print(f"Level = {current_depth}")

            x_previous_layer, x_doubleconv_down = self.networks_down[str(current_depth)]["subnet"](x_previous_layer)
            
            doubleconv_down_out[current_depth] = x_doubleconv_down  # Save the double conv output for concatenation
            
        for current_depth in network_layers_up:
            #print(f"Level = {current_depth}")

            x_previous_layer = self.networks_up[str(current_depth)]["subnet"](x_previous_layer, doubleconv_down_out[current_depth])
            
        return self.conv_out(x_previous_layer)
