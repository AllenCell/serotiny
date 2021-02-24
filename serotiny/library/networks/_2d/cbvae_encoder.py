import numpy as np
import torch

from torch import nn

from ...norm import spectral_norm
from ...layers.activation import activation_map
from ...layers._2d.down_residual import DownResidualLayer

class CBVAEEncoder(nn.Module):
    def __init__(
        self,
        n_latent_dim,
        n_classes,
        gpu_ids,
        n_ch_target=1,
        n_ch_ref=2,
        conv_channels_list=[64, 128, 256, 512, 1024],
        imsize_compressed=[5, 3],
    ):
        super(Enc, self).__init__()

        self.gpu_ids = gpu_ids

        self.n_latent_dim = n_latent_dim

        target_cond_list = []
        if n_ch_ref > 0:
            target_cond_list.append(n_ch_ref)

        if n_classes > 0:
            target_cond_list.append(n_classes)

        self.target_path = nn.ModuleList(
            [
                DownResidualLayer(
                    n_ch_target, conv_channels_list[0], ch_cond_list=target_cond_list
                )
            ]
        )

        for ch_in, ch_out in zip(conv_channels_list[0:-1], conv_channels_list[1:]):
            self.target_path.append(
                DownResidualLayer(ch_in, ch_out, ch_cond_list=target_cond_list)
            )

            ch_in = ch_out

        if self.n_latent_dim > 0:
            self.latent_out_mu = spectral_norm(
                nn.Linear(
                    ch_in * int(np.prod(imsize_compressed)),
                    self.n_latent_dim,
                    bias=True,
                )
            )

            self.latent_out_sigma = spectral_norm(
                nn.Linear(
                    ch_in * int(np.prod(imsize_compressed)),
                    self.n_latent_dim,
                    bias=True,
                )
            )

    def forward(self, x_target, x_ref=None, x_class=None):

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

        x_target = x_target.view(x_target.size()[0], -1)

        mu = self.latent_out_mu(x_target)
        logsigma = self.latent_out_sigma(x_target)

        return [mu, logsigma]
