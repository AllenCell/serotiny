import torch
from torch import nn

from .double_convolution import DoubleConvolution

class DownConvolution(nn.Module):
    def __init__(
            self,
            current_depth: int,
            n_in: int,
            n_out: int,
            kernel_size: int,
            padding: int,
            pooling: str = 'average',
            ):
        super().__init__()
        
        self.current_depth = current_depth
        
        # If we're at the bottom, only instantiate one double_conv
        if self.current_depth == 0:
            self.double_conv = DoubleConvolution(n_in, n_out, kernel_size=kernel_size, padding=padding)
        else:
            self.double_conv = DoubleConvolution(n_in, n_out, kernel_size=kernel_size, padding=padding)
                        
            # In original paper, kernel_size = 2 and stride = 2 (TODO: Use channel_fan?)
            if pooling == 'average':
                self.pooling = nn.AvgPool3d(kernel_size=2, stride=2)
            elif pooling == 'max':
                self.pooling = nn.MaxPool3d(kernel_size=2, stride=2)
            else:
                # Use convolution to perform pooling 
                self.pooling = nn.Conv3d(n_out, n_out, kernel_size=2, stride=2)
            
            self.batch_norm  = nn.BatchNorm3d(n_out)
            self.activation  = nn.ReLU()

    def forward(self, x):
        
        if self.current_depth != 0:
            x_doubleconv_down = self.double_conv(x)
            #doubleconv_down_out[layer] = x_doubleconv_down  # Save the double conv output for concatenation

            x_pool_down = self.pooling(x_doubleconv_down)
            x_batchnorm_down = self.batch_norm(x_pool_down)
            x_activation_down = self.activation(x_batchnorm_down)
            #x_previous_layer = x_activation_down
            x_return = x_activation_down

        else:
            x_doubleconv_down = self.double_conv(x)
            #x_network_down_out = x_doubleconv_down
            x_return = x_doubleconv_down
            
        return x_return, x_doubleconv_down
