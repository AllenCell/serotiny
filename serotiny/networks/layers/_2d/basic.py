import numpy as np
import torch

from torch import nn

from ..activation import activation_map
from torch.nn.utils import spectral_norm


class BasicLayer(nn.Module):
    def __init__(
        self,
        ch_in: int,
        ch_out: int,
        ksize: int,
        dstep: int,
        padding: int,
        activation: str,
        bn: bool,
    ):
        super(BasicLayer, self).__init__()

        self.conv = spectral_norm(
            nn.Conv2d(ch_in, ch_out, ksize, dstep, padding=padding, bias=True)
        )

        if bn:
            self.bn = nn.BatchNorm2d(ch_out)
        else:
            self.bn = None

        self.activation = activation_map(activation)

    def forward(self, x):
        x = self.conv(x)

        if self.bn is not None:
            x = self.bn(x)

        x = self.activation(x)

        return x
