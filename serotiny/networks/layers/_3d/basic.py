import numpy as np
import torch
from torch import nn

from torch.nn.utils import spectral_norm
from ..activation import activation_map


class BasicLayer(nn.Module):
    def __init__(
        self,
        ch_in: int,
        ch_out: int,
        ksize: int, # =4,
        dstep: int, # =2,
        padding: int, # =1,
        activation: str, # ="relu",
        bn: bool, # =True,
    ):
        self, ch_in, ch_out, ksize=4, dstep=2, padding=1, activation="relu", bn=True
    ):
        super(BasicLayer, self).__init__()

        self.conv = spectral_norm(
            nn.Conv3d(ch_in, ch_out, ksize, dstep, padding=padding, bias=True)
        )

        if bn:
            self.bn = nn.BatchNorm3d(ch_out)
        else:
            self.bn = None

        self.activation = activation_map(activation)

    def forward(self, x):
        x = self.conv(x)

        if self.bn is not None:
            x = self.bn(x)

        x = self.activation(x)

        return x
