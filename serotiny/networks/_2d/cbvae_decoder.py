import numpy as np
import torch

from torch import nn

from ..norm import spectral_norm
from ..layers._2d.up_residual import UpResidualLayer


class CBVAEDecoder(nn.Module):
    def __init__(
        self,
        n_latent_dim,
        n_classes,
        padding_latent=[0, 0],
        imsize_compressed=[5, 3],
        n_ch_target=1,
        n_ch_ref=2,
        conv_channels_list=[1024, 512, 256, 128, 64],
        activation_last="sigmoid",
    ):

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
            nn.BatchNorm2d(conv_channels_list[0]), nn.ReLU(inplace=True)
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
                    output_padding=padding,
                    ch_cond_list=target_cond_list,
                )
            )

        self.target_path.append(
            UpResidualLayer(
                l_sizes[i + 1],
                n_ch_target,
                ch_cond_list=target_cond_list,
                activation_last=activation_last,
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
            self.imsize_compressed[0],
            self.imsize_compressed[1],
        )
        x_target = self.target_bn_relu(x_target)

        for ref, target_path in zip(x_ref, self.target_path):

            target_cond = [ref, x_class]

            x_target = target_path(x_target, *target_cond)

        return x_target
