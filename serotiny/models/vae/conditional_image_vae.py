import logging
from typing import Optional, Sequence, Union

import numpy as np
import torch
import torch.nn as nn
from torch.nn.modules.loss import _Loss as Loss

from serotiny.networks import BasicCNN
from serotiny.networks import MLP
from serotiny.networks.vae import ImageDecoder
from serotiny.networks.utils import weight_init

from .base_vae import BaseVAE, get_args

Array = Union[torch.Tensor, np.array, Sequence[float]]
logger = logging.getLogger("lightning")
logger.propagate = False


class ConditionalImageVAE(BaseVAE):
    def __init__(
        self,
        latent_dim: Union[int, Sequence[int]],
        in_channels: int,
        hidden_channels: Sequence[int],
        max_pool_layers: Sequence[int],
        input_dims: Sequence[int],
        x_label: str,
        c_label: str,
        optimizer: torch.optim.Optimizer = torch.optim.Adam,
        beta: float = 1.0,
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
        batch_norm: bool = True,
        mode: str = "3d",
        kernel_size: int = 3,
        cache_outputs: Sequence = ("test",),
        linear_hidden_layers: list = [256],
        c_dim: int = 1,
    ):

        self.c_label = c_label

        encoder = BasicCNN(
            in_channels=in_channels,
            input_dims=input_dims,
            output_dim=latent_dim * 2,  # because it has to return mu and sigma
            hidden_channels=hidden_channels,
            max_pool_layers=max_pool_layers,
            mode=mode,
            kernel_size=kernel_size,
            non_linearity=non_linearity,
            skip_connections=skip_connections,
            batch_norm=batch_norm,
            flat_output=False,
        )
        encoder.apply(weight_init)
        # nn.utils.spectral_norm(encoder.output)

        dummy_out, intermediate_sizes = encoder.conv_forward(
            torch.zeros(1, in_channels, *input_dims), return_sizes=True
        )
        flat_dim = dummy_out.nelement()

        compressed_img_shape = dummy_out.shape[2:]

        intermediate_sizes = [input_dims] + intermediate_sizes[:-1]
        intermediate_sizes = intermediate_sizes[::-1]

        decoder = ImageDecoder(
            compressed_img_shape=compressed_img_shape,
            hidden_channels=list(reversed(hidden_channels)),
            intermediate_sizes=intermediate_sizes,
            latent_dim=latent_dim,
            output_dims=tuple(input_dims),
            output_channels=in_channels,
            mode=mode,
            non_linearity=non_linearity,
            skip_connections=skip_connections,
            batch_norm=batch_norm,
            kernel_size=kernel_size,
            c_dim=c_dim,
            linear_hidden_layers=linear_hidden_layers,
        )
        decoder.apply(weight_init)
        # nn.utils.spectral_norm(decoder.linear_decompress)

        if decoder_non_linearity is not None:
            decoder = nn.Sequential(decoder, decoder_non_linearity)

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

        self.flat_layers = [nn.Flatten(), MLP(flat_dim + c_dim, latent_dim*2, hidden_layers=linear_hidden_layers)]
        self.flat_layers = nn.Sequential(*self.flat_layers)
        self.flat_layer_kwargs = get_args(self.flat_layers)

    def parse_batch(self, batch):
        if self.hparams.loss_mask_label is not None:
            mask = batch[self.hparams.loss_mask_label].float()
        else:
            mask = None
        
        return batch[self.hparams.x_label].float(), {"condition": batch[self.c_label].float()}

    def encode(self, x, **kwargs):
        _tmp = self.encoder(
            x, **{k: v for k, v in kwargs.items() if k in self.encoder_args}
        ) # # Convolutions
        _tmp = self.flat_layers[0](_tmp) # Flatten

        mu_logvar = self.flat_layers[1](
            _tmp, **{k: v for k, v in kwargs.items() if k in 'condition'}
        ) # Linear compress

        mu, logvar = torch.split(mu_logvar, mu_logvar.shape[1] // 2, dim=1)

        assert mu.shape == logvar.shape
        return mu, logvar

    def decode(self, mu, logvar, **kwargs):
        z = self.sample_z(mu, logvar)
        x_hat = self.decoder[0](
            z, **kwargs
        ) #condition
        x_hat = self.decoder[1](x_hat) # sigmoid
        return z, x_hat


