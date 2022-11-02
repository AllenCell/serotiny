import logging
from typing import Optional, Sequence, Union
from unittest import skip

import numpy as np
import torch
import torch.nn as nn
from torch.nn.modules.loss import _Loss as Loss
from serotiny.models.vae import ImageVAE

from .base_vae import BaseVAE
from .priors import Prior

Array = Union[torch.Tensor, np.ndarray, Sequence[float]]
logger = logging.getLogger("lightning")
logger.propagate = False
from omegaconf import DictConfig
import torch.nn.functional as F


def get_rotation_matrix(v, eps=10e-5):
    v = v / (torch.norm(v, dim=-1, keepdim=True) + eps)
    rot = torch.stack(
        (
            torch.stack((v[:, 0], v[:, 1]), dim=-1),
            torch.stack((-v[:, 1], v[:, 0]), dim=-1),
            torch.zeros(v.size(0), 2).type_as(v),
        ),
        dim=-1,
    )
    return rot


def rot_img(x, rot):
    grid = F.affine_grid(rot, x.size(), align_corners=False).type_as(x)
    x = F.grid_sample(x, grid, align_corners=False)
    return x


class EqImageVAE(ImageVAE):
    def __init__(
        self,
        latent_dim: Union[int, Sequence[int]],
        x_label: str,
        optimizer: torch.optim.Optimizer = torch.optim.Adam,
        reconstruction_loss: Loss = nn.MSELoss(reduction="none"),
        beta: float = 1.0,
        prior: Optional[Sequence[Prior]] = None,
        encoder: Optional[nn.Module] = None,
        decoder: Optional[nn.Module] = None,
        in_channels: Optional[int] = None,
        hidden_channels: Optional[Sequence[int]] = None,
        max_pool_layers: Optional[Sequence[int]] = None,
        input_dims: Optional[Sequence[int]] = None,
        id_label: Optional[str] = None,
        non_linearity: Optional[nn.Module] = None,
        decoder_non_linearity: Optional[nn.Module] = None,
        loss_mask_label: Optional[str] = None,
        skip_connections: Optional[bool] = True,
        batch_norm: Optional[bool] = True,
        mode: Optional[str] = "3d",
        kernel_size: Optional[int] = 3,
        cache_outputs: Optional[Sequence] = ("test",),
        encoder_clamp: Optional[int] = 6,
        # final_non_linearity: Optional[nn.Module] = None,
    ):
        self.x_label = x_label
        super().__init__(
            latent_dim=latent_dim,
            x_label=x_label,
            optimizer=optimizer,
            reconstruction_loss=reconstruction_loss,
            beta=beta,
            prior=prior,
            encoder=encoder,
            decoder=decoder,
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            max_pool_layers=max_pool_layers,
            input_dims=input_dims,
            id_label=id_label,
            non_linearity=non_linearity,
            decoder_non_linearity=decoder_non_linearity,
            loss_mask_label=loss_mask_label,
            skip_connections=skip_connections,
            batch_norm=batch_norm,
            mode=mode,
            kernel_size=kernel_size,
            cache_outputs=cache_outputs,
            encoder_clamp=encoder_clamp,
        )

    def forward(self, batch, do_rot=True, decode=False, compute_loss=False, **kwargs):
        if isinstance(batch, list):
            batch = batch[0]
        z_parts_params_plus_angle = self.encode(batch)
        z_parts_params = {self.x_label: z_parts_params_plus_angle[self.x_label][0]}
        v = z_parts_params_plus_angle[self.x_label][1]
        rot = get_rotation_matrix(v)

        z_parts = self.sample_z(z_parts_params)

        x_hat, z_composed = self.decode(z_parts)

        if do_rot:
            x_hat[self.x_label] = rot_img(x_hat[self.x_label], rot)

        if not decode:
            return z_parts_params, z_composed

        if not compute_loss:
            return x_hat, z_parts, z_parts_params, z_composed

        (
            loss,
            reconstruction_loss,
            kld_loss,
            kld_per_part,
        ) = self.calculate_elbo(batch, x_hat, z_parts_params)

        z_parts_params["angle"] = rot

        return (
            x_hat,
            z_parts,
            z_parts_params,
            z_composed,
            loss,
            reconstruction_loss,
            kld_loss,
            kld_per_part,
        )
