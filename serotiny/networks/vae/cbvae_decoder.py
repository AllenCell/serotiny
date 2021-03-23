from typing import Sequence, Optional

import numpy as np
import torch

from torch import nn

from torch.nn.utils import spectral_norm


class CBVAEDecoder(nn.Module):
    def __init__(
        self,
        dimensionality: int,
        n_latent_dim: int,
        n_classes: int,
        imsize_compressed: Sequence[int],
        n_ch_target: int,
        n_ch_ref: int,
        conv_channels_list: Sequence[int],
        activation: str,
        activation_last: str,
        padding_latent: Optional[Sequence[int]] = None,
    ):
        """
        Decoder used in the final version of the pytorch_integrated_cell project.

        Leverages UpResidualLayer, defined and described in `layers/_xd/up_residual.py`

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
        imsize_compressed: Sequence[int]
            Dimensions of the input before being fed through the convolutional
            portion of the net. (Corresponds to the dimensions of the output
            of the encoder's convolutional path)
        n_ch_target: int
            Number of channels on the target input
        n_ch_ref: int
            Number of channels on the reference input
        conv_channels_list: Sequence[int]
            Number of channels on the intermediate conv layers. Also used
            to tell how many layers to use.
        activation: str
            String to specify activation function to be used in inner layers
        activation_last: str
            String to specify activation function to be used in final layers
        padding_latent: Optional[Sequence[int]] = None
            Size of padding to apply to inner layers
        """
        if dimensionality == 2:
            from ..layers._2d.up_residual import UpResidualLayer
            batch_norm = nn.BatchNorm2d
        elif dimensionality == 3:
            from ..layers._3d.up_residual import UpResidualLayer
            batch_norm = nn.BatchNorm3d
        else:
            raise ValueError("`dimensionality` has to be 2 or 3")

        if padding_latent is None:
            padding_latent = [0] * dimensionality

        assert len(padding_latent) == dimensionality
        assert len(imsize_compressed) == dimensionality

        super().__init__()

        self.padding_latent = padding_latent
        self.imsize_compressed = imsize_compressed

        self.ch_first = conv_channels_list[0]

        self.n_latent_dim = n_latent_dim
        self.n_classes = n_classes
        self.target_fc = spectral_norm(
            nn.Linear(
                self.n_latent_dim,
                conv_channels_list[0] * int(np.prod(self.imsize_compressed)),
                bias=True,
            )
        )

        self.target_bn_relu = nn.Sequential(
            batch_norm(conv_channels_list[0]), nn.ReLU(inplace=True)
        )

        self.target_path = nn.ModuleList([])

        target_cond_list = []
        if n_ch_ref > 0:
            target_cond_list.append(n_ch_ref)

        if n_classes > 0:
            target_cond_list.append(n_classes)

        l_sizes = conv_channels_list
        for i in range(len(l_sizes) - 1):
            if i == 0:
                padding = padding_latent
            else:
                padding = 0

            self.target_path.append(
                UpResidualLayer(
                    l_sizes[i],
                    l_sizes[i + 1],
                    ch_cond_list=target_cond_list,
                    activation=activation,
                    activation_last=activation,
                    output_padding=padding,
                )
            )

        self.target_path.append(
            UpResidualLayer(
                l_sizes[i + 1],
                n_ch_target,
                ch_cond_list=target_cond_list,
                activation=activation,
                activation_last=activation_last,
                output_padding=0
            )
        )

    def forward(self, z_target, x_ref=None, x_class=None):
        scales = 1 / (2 ** torch.arange(0, len(self.target_path)).float())

        if x_ref is None:
            x_ref = [None] * (len(scales) + 1)
        else:
            x_ref = [x_ref] + [
                torch.nn.functional.interpolate(x_ref, scale_factor=scale.item())
                for scale in scales[1:]
            ]
            x_ref = x_ref[::-1]

        x_target = self.target_fc(z_target).view(
            z_target.size()[0],
            self.ch_first,
            *self.imsize_compressed,
        )
        x_target = self.target_bn_relu(x_target)

        for ref, target_path in zip(x_ref, self.target_path):

            target_cond = [ref, x_class]

            x_target = target_path(x_target, *target_cond)

        return x_target