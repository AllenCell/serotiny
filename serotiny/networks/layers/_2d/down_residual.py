from typing import Optional, Sequence
from torch import nn

from torch.nn.utils import spectral_norm
from ..activation import activation_map
from .basic import BasicLayer


class DownResidualLayer(nn.Module):
    def __init__(
        self,
        ch_in: int,
        ch_out: int,
        activation: str = "relu",
        ch_cond_list: Optional[Sequence[int]] = None,
        activation_last: Optional[str] = None,
    ):
        """
        Basic layer used in the CBVAE model (extracted from pytorch_integrated_cell).

        This network is composed of two main paths + some number of conditional paths
        The two main paths:
            - The bypass path applies a 1x1 conv preceded by 2x2 average pooling
            - The residual path applies
                - a 4x4 conv with striding=2 and padding=1
                - followed by batchnorm and an activation function
                - followed by a 3x3 conv with striding=1 and padding=1
                - followed by batchnorm

        For each conditioning input there is a conditional path which applies:
            - A 1x1 conv with striding=1 and padding=0
            - batchnorm and an activation function
        (When the conditioning input is a vector, this reduces to a fully connected
        layer)

        The outputs of all of the paths are summed and passed through a final
        activation layer.

        Parameters
        ----------
        ch_in: int
            Number of input channels
        ch_out: int
            Number of output channels
        activation: str
            Activation function to use. Defaults to "relu"
        ch_cond_list: Optional[list]
            Number of channels of each of the conditioning inputs. This is also
            used to know how many conditioning inputs there will be. Defaults to []
        activation_last: Optional[str]
            Activation function to use in the last layer. If None, the same as
            `activation` gets used
        """
        super().__init__()

        if activation_last is None:
            activation_last = activation
        if ch_cond_list is None:
            ch_cond_list = []

        self.bypass = nn.Sequential(
            nn.AvgPool2d(2, stride=2, padding=0),
            spectral_norm(nn.Conv2d(ch_in, ch_out, 1, 1, padding=0, bias=True)),
        )

        self.resid = nn.Sequential(
            spectral_norm(nn.Conv2d(ch_in, ch_in, 4, 2, padding=1, bias=True)),
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
