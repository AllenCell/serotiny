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
            out_channels: int = 1,
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
        self.depth = depth
        
        # 4: 3 -> 6,     12 -> 6
        # 3: 6 -> 12,    24 -> 12
        # 2: 12 -> 24,   48 -> 24
        # 1: 24 -> 48,   96 -> 48
        # 0:       48 -> 96
        
        # Create the down pathway
        channels_down = {}
        self.networks_down = {}
        for current_depth in range(depth, -1, -1):
            #print(current_depth)
            self.networks_down[current_depth] = {}
            
            # At the top layer
            if current_depth == depth:
                n_in = in_channels
                n_out = channel_fan
                
            else:
                n_in = channels_down[current_depth+1][1]
                n_out = n_in * channel_fan  # TODO: Was hardcoded to 2 in label-free
                
            channels_down[current_depth] = (n_in, n_out)
            
            # If we're at the bottom, only instantiate one double_conv
            if current_depth == 0:
                self.networks_down[current_depth]['double_conv'] = DoubleConvolution(n_in, n_out, kernel_size=kernel_size, padding=padding)
            else:
                self.networks_down[current_depth]['double_conv'] = DoubleConvolution(n_in, n_out, kernel_size=kernel_size, padding=padding)
                self.networks_down[current_depth]['pooling']     = nn.MaxPool3d(kernel_size=2, stride=2)  # TODO: Use channel_fan?
                self.networks_down[current_depth]['batch_norm']  = nn.BatchNorm3d(n_out)
                self.networks_down[current_depth]['activation']  = nn.ReLU()            
            
        print(f'channels_down = {channels_down}')
        
        # Create the up pathway
        channels_up = {}
        self.networks_up = {}
        for current_depth in sorted(list(channels_down.keys())):
            #print(current_depth)
            
            if current_depth == 0:
                pass
            else:
                self.networks_up[current_depth] = {}
                
                n_in = channels_down[current_depth-1][1]
                n_out = n_in // channel_fan  # TODO: Support values of channel_fan other than 2 (was hardcoded to 2 in label-free)
                
                channels_up[current_depth] = (n_in, n_out)
                
                self.networks_up[current_depth]['double_conv'] = DoubleConvolution(n_in, n_out, kernel_size=kernel_size, padding=padding)
                self.networks_up[current_depth]['batch_norm']  = nn.BatchNorm3d(n_out)
                self.networks_up[current_depth]['activation']  = nn.ReLU()            
                self.networks_up[current_depth]['up_conv']     = nn.ConvTranspose3d(n_in, n_out, kernel_size=2, stride=2)  # TODO: parameterize kernel_size and stride using channel_fan or others?           
        
        print(f'channels_up = {channels_up}')
        
        print('networks_down:')
        self.print_network(self.networks_down, reverse=True)
        print()
        
        print('networks_up:')
        self.print_network(self.networks_up, reverse=True)
        
        self.conv_out = torch.nn.Conv3d(
            channels_up[depth][1], out_channels, kernel_size=kernel_size, stride=1, padding=padding
        )
        print(self.conv_out)

        '''
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
        '''
        
        '''
        out_channels = in_channels * channel_fan
        
        # Paper: k=3, p=0
        self.step_down = DoubleConvolution(in_channels, out_channels, kernel_size=kernel_size, padding=padding)
        self.step_bottom = DoubleConvolution(out_channels, out_channels * channel_fan, kernel_size=kernel_size, padding=padding)
        self.step_up = DoubleConvolution(out_channels * channel_fan, out_channels, kernel_size=kernel_size, padding=padding)

        # Paper: max, k=2, s=2
        if pooling == 'average':
            self.pooling = nn.AvgPool3d(kernel_size=channel_fan, stride=channel_fan)  # Use channel_fan?
        elif pooling == 'max':
            self.pooling = nn.MaxPool3d(kernel_size=channel_fan, stride=channel_fan)  # Use channel_fan?
        else:
            # use convolutional stride to pool 
            self.pooling = torch.nn.Conv3d(
                n_out_channels, n_out_channels, channel_fan, stride=channel_fan  # Use channel_fan?
            )

        self.bn0 = torch.nn.BatchNorm3d(out_channels)
        self.relu0 = torch.nn.ReLU()
        
        # Paper: k=2
        self.convt = torch.nn.ConvTranspose3d(
            channel_fan * out_channels,
            out_channels,
            kernel_size=channel_fan,  # Use channel_fan?
            stride=2  # TODO: parameterize?
        )

        self.bn1 = torch.nn.BatchNorm3d(out_channels)
        self.relu1 = torch.nn.ReLU()
        '''
        
    def print_network(self, network_dict, reverse=False):
        
        network_keys = sorted(network_dict.keys(), reverse=reverse)
        
        for level in network_keys:
            print(f'Level = {level}')
            #print(f'{network_dict[level]}')
            
            for key in network_dict[level].keys():
                print(f'  {key} = {network_dict[level][key]}')
        
    def forward(self, x):
        
        network_layers_down = sorted(self.networks_down.keys(), reverse=True)
        network_layers_up = sorted(self.networks_up.keys(), reverse=False)
        
        x_previous_layer = x
        doubleconv_down_out = {}
        
        for layer in network_layers_down:
            print(f'layer = {layer}')
            
            if layer != 0:
                x_doubleconv_down = self.networks_down[layer]['double_conv'](x_previous_layer)
                doubleconv_down_out[layer] = x_doubleconv_down  # Save the double conv output for concatenation

                x_pool_down = self.networks_down[layer]['pooling'](x_doubleconv_down)
                x_batchnorm_down = self.networks_down[layer]['batch_norm'](x_pool_down)
                x_activation_down = self.networks_down[layer]['activation'](x_batchnorm_down)
                x_previous_layer = x_activation_down
            
            else:
                x_doubleconv_down = self. networks_down[layer]['double_conv'](x_previous_layer)
                #x_network_down_out = x_doubleconv_down
            
        x_previous_layer = x_doubleconv_down 
        for layer in network_layers_up:
            print(f'layer = {layer}')
            
            
            x_upconv_up = self.networks_up[layer]['up_conv'](x_previous_layer)
            x_doubleconv_up = self.networks_up[layer]['double_conv'](
                torch.cat((x_upconv_up, doubleconv_down_out[layer]), 1)
            )
            x_batchnorm_up = self.networks_up[layer]['batch_norm'](x_doubleconv_up)
            x_activation_up = self.networks_up[layer]['activation'](x_batchnorm_up)
            x_previous_layer = x_activation_up
            
            
        return self.conv_out(x_previous_layer)
        
        '''
        print(f'x.shape = {x.shape}')
        x_down = self.step_down(x)
        print(f'x_down.shape = {x_down.shape}')
        x_pool = self.pooling(x_down)
        print(f'x_pool.shape = {x_pool.shape}')
        x_bn0 = self.bn0(x_pool)
        print(f'x_bn0.shape = {x_bn0.shape}')
        x_relu0 = self.relu0(x_bn0)
        print(f'x_relu0.shape = {x_relu0.shape}')
        x_bottom = self.step_bottom(x_relu0)
        print(f'x_bottom.shape = {x_bottom.shape}')
        x_convt = self.convt(x_bottom)
        print(f'x_convt.shape = {x_convt.shape}')
        x_bn1 = self.bn1(x_convt)
        print(f'x_bn1.shape = {x_bn1.shape}')
        x_relu1 = self.relu1(x_bn1)
        print(f'x_relu1.shape = {x_relu1.shape}')
        x_cat = torch.cat((x_down, x_relu1), 1)  # concatenate (need to crop x_down if x_relu1 is smaller, or, set padding = 1 in double_conv)
        print(f'x_cat.shape = {x_cat.shape}')
        x_up = self.step_up(x_cat)
        print(f'x_up.shape = {x_up.shape}')
        '''

        #return x_up
