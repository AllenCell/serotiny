from typing import Sequence

import numpy as np
import torch

from torch import nn

from torch.nn.utils import spectral_norm


class CBVAEEncoder(nn.Module):
    def __init__(
        self,
        dimensionality: int,
        n_latent_dim: int,
        n_classes: int,
        n_ch_target: int,
        n_ch_ref: int,
        conv_channels_list: Sequence[int],
        input_dims: Sequence[int],
        activation: str,
    ):
        """
        Decoder used in the final version of the pytorch_integrated_cell project.

        Leverages DownResidualLayers, defined and described `layers/_xd/down_residual.py`

        This class works for both 2d and 3d data, according to what is given by
        `dimensionality`.

        Parameters
        ----------
        dimensionality: int
            Whether to instantiate a 2d or 3d model
        n_latent_dim: int
            Dimensionality of the latent space
        n_classes: int
            Number of classes (for conditioning)
        n_ch_target: int
            Number of channels on the target input
        n_ch_ref: int
            Number of channels on the reference input
        conv_channels_list: Sequence[int]
            Number of channels on the intermediate conv layers. Also used
            to tell how many layers to use.
        input_dims:
            Dimensions of the input images
        activation: str
            String to specify activation function to be used in inner layers
        """

        assert len(input_dims) == dimensionality

        if dimensionality == 2:
            from ..layers._2d.down_residual import DownResidualLayer
        elif dimensionality == 3:
            from ..layers._3d.down_residual import DownResidualLayer
        else:
            raise ValueError("`dimensionality` has to be 2 or 3")

        super().__init__()

        self.n_latent_dim = n_latent_dim

        target_cond_list = []
        if n_ch_ref > 0:
            target_cond_list.append(n_ch_ref)

        if n_classes > 0:
            target_cond_list.append(n_classes)

        self.target_path = nn.ModuleList(
            [
                DownResidualLayer(
                    n_ch_target,
                    conv_channels_list[0],
                    ch_cond_list=target_cond_list,
                    activation=activation,
                    activation_last=activation,
                )
            ]
        )

        for ch_in, ch_out in zip(conv_channels_list[0:-1], conv_channels_list[1:]):
            self.target_path.append(
                DownResidualLayer(
                    ch_in,
                    ch_out,
                    ch_cond_list=target_cond_list,
                    activation=activation,
                    activation_last=activation,
                )
            )

            ch_in = ch_out

        # pass a dummy input through the convolutions to obtain the dimensions
        # before the last linear layer
        with torch.no_grad():
            self.eval()
            self.imsize_compressed = tuple(
                self.conv_forward(
                    torch.zeros(1, n_ch_target, *input_dims),
                    (torch.zeros(1, n_ch_ref, *input_dims) if n_ch_ref > 0 else None),
                    (torch.zeros(1, n_classes) if n_classes > 0 else None),
                ).shape[2:]
            )
        self.train()

        if self.n_latent_dim > 0:
            self.latent_out_mu = spectral_norm(
                nn.Linear(
                    ch_in * int(np.prod(self.imsize_compressed)),
                    self.n_latent_dim,
                    bias=True,
                )
            )

            self.latent_out_sigma = spectral_norm(
                nn.Linear(
                    ch_in * int(np.prod(self.imsize_compressed)),
                    self.n_latent_dim,
                    bias=True,
                )
            )

    def conv_forward(self, x_target, x_ref=None, x_class=None):
        scales = 1 / (2 ** torch.arange(0, len(self.target_path) + 1).float())

        if x_ref is None:
            x_ref = [None] * len(scales)
        else:
            x_ref = [
                torch.nn.functional.interpolate(x_ref, scale_factor=scale.item())
                for scale in scales[1:]
            ]

        for ref, target_path in zip(x_ref, self.target_path):
            x_target = target_path(x_target, ref, x_class)

        return x_target

    def forward(self, x_target, x_ref=None, x_class=None):

        x_target = self.conv_forward(x_target, x_ref, x_class)
        x_target = x_target.view(x_target.size()[0], -1)

        mu = self.latent_out_mu(x_target)
        logsigma = self.latent_out_sigma(x_target)

        return [mu, logsigma]
