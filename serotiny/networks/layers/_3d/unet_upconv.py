import torch
from torch import nn

from .double_convolution import DoubleConvolution

class UpConvolution(nn.Module):
    def __init__(
            self,
            current_depth: int,
            n_in: int,
            n_out: int,
            kernel_size: int,
            padding: int,
            ):
        super().__init__()
        
        self.current_depth = current_depth
        
        # If we're at the bottom, do nothing, since the bottom-most layer is taken cared of by DownConvolution
        if self.current_depth == 0:
            pass
        else:
            self.up_conv     = nn.ConvTranspose3d(n_in, n_out, kernel_size=2, stride=2)  # In original paper, kernel_size = 2 (TODO: parameterize kernel_size and stride using channel_fan or others?)
            self.double_conv = DoubleConvolution(n_in, n_out, kernel_size=kernel_size, padding=padding)
            self.batch_norm  = nn.BatchNorm3d(n_out)
            self.activation  = nn.ReLU()
            
    def forward(self, x, x_doubleconv_down):
        
        x_upconv_up = self.up_conv(x)
        x_doubleconv_up = self.double_conv(
            torch.cat((x_upconv_up, x_doubleconv_down), 1)
        )
        x_batchnorm_up = self.batch_norm(x_doubleconv_up)
        x_activation_up = self.activation(x_batchnorm_up)
        #x_previous_layer = x_activation_up
            
        return x_activation_up
