#!/usr/bin/env python
# coding: utf-8

import torch
from torch import nn
from torch.nn import functional as F

from ..layers._3d.unet_downconv import DownConvolution
from ..layers._3d.unet_upconv import UpConvolution


class Unet(nn.Module):
    def __init__(
        self,
        depth: int = 4,
        in_channels: int = 1,
        out_channels: int = 1,
        channel_fan: int = 2,
        kernel_size: int = 3,
        padding: int = 1,
        pooling: str = "average",
        # dimensions: tuple, # =(176, 104, 52),
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
              concatenations
            - Separate channel_fan between first layer and the next for double_conv (right
              now this calculation is entirely done in the unet.py)
            - Separate kernel_size, stride, and padding for double_conv, pooling, up_conv,
              UpConv, DownConv, and conv_out
        """

        super().__init__()

        self.network_name = "Unet"
        self.depth = depth

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
                n_in = in_channels
                n_out = (
                    n_in * channel_fan
                )  # BUG? Was only channel_fan before. Not bug: label-free applied channel_fan only in the top layer

            else:
                n_in = self.channels_down[current_depth + 1][1]
                n_out = n_in * channel_fan  # TODO: Was hardcoded to 2 in label-free?

            self.channels_down[current_depth] = (n_in, n_out)
            self.networks_down[str(current_depth)]["subnet"] = DownConvolution(
                current_depth,
                n_in,
                n_out,
                kernel_size=kernel_size,
                padding=padding,
                pooling=pooling,
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
                n_out = (
                    n_in // channel_fan
                )  # TODO: Support values of channel_fan other than 2 (was hardcoded to 2 in label-free for calling _Net_recurse)

                self.channels_up[current_depth] = (n_in, n_out)
                self.networks_up[str(current_depth)]["subnet"] = UpConvolution(
                    current_depth, n_in, n_out, kernel_size=kernel_size, padding=padding
                )

        # In original paper, kernel_size = 1
        self.conv_out = nn.Conv3d(
            self.channels_up[depth][1],
            out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
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
            print(f"Level = {current_depth}")

            x_previous_layer, x_doubleconv_down = self.networks_down[str(current_depth)]["subnet"](x_previous_layer)
            
            doubleconv_down_out[current_depth] = x_doubleconv_down  # Save the double conv output for concatenation
            
        for current_depth in network_layers_up:
            print(f"Level = {current_depth}")

            x_previous_layer = self.networks_up[str(current_depth)]["subnet"](x_previous_layer, doubleconv_down_out[current_depth])
            
        return self.conv_out(x_previous_layer)
