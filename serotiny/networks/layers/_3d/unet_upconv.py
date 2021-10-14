import torch
from torch import nn

from .double_convolution import DoubleConvolution


class UpConvolution(nn.Module):
    def __init__(
        self,
        current_depth: int,
        n_in: int,
        n_out: int,
        kernel_size_upconv: int = 2,  # Original paper = 2
        stride_upconv: int = 2,  # Original paper = no mention, label-free = 2
        padding_upconv: int = 0,  # Original paper = no mention, label-free = default
        kernel_size_doubleconv: int = 3,  # Original paper = 3
        stride_doubleconv: int = 1,  # Original paper = no mention, label-free = default
        padding_doubleconv: int = 1,  # Original paper = 0, label-free = 1
    ):
        super().__init__()

        self.current_depth = current_depth

        # If we're at the bottom, do nothing, since the bottom-most layer is
        # taken care of by DownConvolution
        if self.current_depth == 0:
            pass
        else:
            self.up_conv = nn.ConvTranspose3d(
                n_in,
                n_out,
                kernel_size=kernel_size_upconv,
                stride=stride_upconv,
                padding=padding_upconv,
            )
            self.double_conv = DoubleConvolution(
                n_in,
                n_out,
                kernel_size=kernel_size_doubleconv,
                stride=stride_doubleconv,
                padding=padding_doubleconv,
            )
            self.batch_norm = nn.BatchNorm3d(n_out)
            self.activation = nn.ReLU()

    def forward(self, x, x_doubleconv_down):

        x_upconv_up = self.up_conv(x)

        # TODO: For most cases there is only a 1-pixel mismatch between x_upconv_up and
        #       x_doubleconv_down. How to padd assymetrically so that the two images are
        #       not shifted over by 1 pixel?
        x_doubleconv_up = self.double_conv(
            torch.cat((x_upconv_up, x_doubleconv_down), 1)
        )

        x_batchnorm_up = self.batch_norm(x_doubleconv_up)

        x_activation_up = self.activation(x_batchnorm_up)

        return x_activation_up
