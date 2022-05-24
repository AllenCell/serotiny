import logging
from typing import Optional, Sequence, Union

import numpy as np
import torch
import torch.nn as nn
from torch.nn.modules.loss import _Loss as Loss

from serotiny.networks import BasicCNN
from serotiny.networks.vae import ImplicitDecoder
from serotiny.networks.utils import weight_init
import torch.nn.functional as F

from .base_vae import BaseVAE

Array = Union[torch.Tensor, np.ndarray, Sequence[float]]
logger = logging.getLogger("lightning")
logger.propagate = False


class ImplicitDecoderVAE(BaseVAE):
    def __init__(
        self,
        latent_dim: Union[int, Sequence[int]],
        in_channels: int,
        hidden_channels: Sequence[int],
        decoder_hidden_channels: Sequence[int],
        max_pool_layers: Sequence[int],
        input_dims: Sequence[int],
        x_label: str,
        beta: float = 1.0,
        optimizer=torch.optim.Adam,
        id_label: Optional[str] = None,
        non_linearity: Optional[nn.Module] = None,
        decoder_non_linearity: Optional[nn.Module] = None,
        loss_mask_label: Optional[str] = None,
        reconstruction_loss: Loss = nn.MSELoss(reduction="none"),
        reconstruction_reduce: str = "mean",
        prior_mode: str = "isotropic",
        prior_logvar: Optional[Array] = None,
        learn_prior_logvar: bool = False,
        skip_connections: bool = True,
        mode: str = "3d",
        kernel_size: int = 3,
        cache_outputs: Sequence = ("test",),
        temperature: Optional[float] = None,
    ):

        encoder = BasicCNN(
            in_channels=in_channels,
            input_dims=input_dims,
            output_dim=latent_dim * 2,  # because it has to return mu and sigma
            hidden_channels=hidden_channels,
            max_pool_layers=max_pool_layers,
            mode=mode,
            kernel_size=kernel_size,
            non_linearity=non_linearity,
            # final_non_linearity=torch.nn.Sigmoid(),
            skip_connections=skip_connections,
        )
        encoder.apply(weight_init)
        nn.utils.spectral_norm(encoder.output)

        decoder = ImplicitDecoder(
            latent_dims=latent_dim,
            hidden_channels=decoder_hidden_channels,
            non_linearity=non_linearity,
            final_non_linearity=decoder_non_linearity,
            mode=mode,
        )
        decoder.apply(weight_init)

        if decoder_non_linearity is not None:
            decoder = nn.Sequential(decoder, decoder_non_linearity)

        self._current_step = 0
        self._temperature = temperature

        super().__init__(
            encoder=encoder,
            decoder=decoder,
            latent_dim=latent_dim,
            optimizer=optimizer,
            x_label=x_label,
            id_label=id_label,
            loss_mask_label=loss_mask_label,
            beta=beta,
            reconstruction_loss=reconstruction_loss,
            reconstruction_reduce=reconstruction_reduce,
            prior_mode=prior_mode,
            prior_logvar=prior_logvar,
            learn_prior_logvar=learn_prior_logvar,
            cache_outputs=cache_outputs,
        )

    def _current_decoding_dims(self, orig_dims):
        if self._temperature is not None:
            self._current_step = max(self._current_step, self.global_step)

            # TODO: add more flexibility for "scale scheduling"
            self._scale = 1 - 0.5 * np.exp(-self._temperature * self._current_step)
        else:
            self._scale = 1
        return [int(self._scale * dim) for dim in orig_dims]

    def calculate_elbo(self, x, x_hat, mu, logvar, mask=None):
        x_hat = F.interpolate(x_hat, size=x.shape[2:])
        return super().calculate_elbo(x, x_hat, mu, logvar, mask)

    def log_metrics(self, stage, reconstruction_loss, kld_loss, loss, logger):
        self.log(f"{stage} reconstruction loss", reconstruction_loss, logger=logger)
        self.log(f"{stage} kld loss", kld_loss, logger=logger)
        self.log(f"{stage}_loss", loss, logger=logger)
        self.log(f"{stage}_scale", self._scale, logger=logger)
        self.log(f"{stage}_step", self._current_step, logger=logger)

    def decode(self, z_params, **kwargs):
        z = self.sample_z(z_params)
        x_hat = self.decoder_non_linearity(
            self.decoder(
                z, **{k: v for k, v in kwargs.items() if k in self.decoder_args}
            )
        )

        return z, x_hat

    def parse_batch(self, batch):
        x, kwargs = super().parse_batch(batch)

        kwargs["input_dims"] = self._current_decoding_dims(x.shape[2:])
        return x, kwargs
