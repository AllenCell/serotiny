from typing import Optional
import numpy as np
import torch

from torch import nn

from torch.nn.utils import spectral_norm
from ..activation import activation_map
from .pad import PadLayer
from .basic import BasicLayer


class UpResidualLayer(nn.Module):
    def __init__(
        self,
        ch_in: int,
        ch_out: int,
        activation: str,
        output_padding: int,
        ch_cond_list: Optional[list],
        activation_last: Optional[str],
    ):
        super().__init__()

        if activation_last is None:
            activation_last = activation
        if ch_cond_list is None:
            ch_cond_list = []

        self.bypass = nn.Sequential(
            spectral_norm(nn.Conv2d(ch_in, ch_out, 1, 1, padding=0, bias=True)),
            nn.Upsample(scale_factor=2),
            PadLayer(output_padding),
        )

        self.resid = nn.Sequential(
            spectral_norm(
                nn.ConvTranspose2d(
                    ch_in,
                    ch_in,
                    4,
                    2,
                    padding=1,
                    output_padding=output_padding,
                    bias=True,
                )
            ),
            nn.BatchNorm2d(ch_in),
            activation_map(activation),
            spectral_norm(nn.Conv2d(ch_in, ch_out, 3, 1, padding=1, bias=True)),
            nn.BatchNorm2d(ch_out),
        )

        self.cond_paths = nn.ModuleList([])
        for ch_cond in ch_cond_list:
            self.cond_paths.append(
                BasicLayer(ch_cond, ch_out, 1, 1, 0, activation=activation, bn=True)
            )

        self.activation = activation_map(activation_last)

    def forward(self, x, *x_cond):
        x = self.bypass(x) + self.resid(x)

        for x_c, cond_path in zip(x_cond, self.cond_paths):
            if len(x.shape) != len(x_c.shape):
                x_c = x_c.unsqueeze(2).unsqueeze(3)

            x_c = cond_path(x_c)

            if x.shape != x_c.shape:
                x_c.expand(x.shape[0], x.shape[1], x.shape[2], x.shape[3])

            x = x + x_c

        x = self.activation(x)

        return x
