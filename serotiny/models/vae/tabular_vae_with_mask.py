from typing import Union, Optional, Sequence

import torch
import numpy as np

from serotiny.networks.mlp import MLP
from .base_vae import BaseVAE

Array = Union[torch.Tensor, np.array, Sequence[float]]


class TabularVAEWithMask(BaseVAE):
    def __init__(
        self,
        x_dim: int,
        latent_dim: int,
        hidden_layers: Sequence[int],
        x_label: str,
        optimizer: str = "Adam",
        lr: float = 1e-3,
        beta: float = 1.0,
        prior_mode: str = "isotropic",
        prior_logvar: Optional[Array] = None,
        learn_prior_logvar: bool = False,
    ):
        encoder = MLP(
            x_dim,
            2 * latent_dim,
            hidden_layers=hidden_layers,
        )

        decoder = MLP(
            latent_dim,
            x_dim,
            hidden_layers=hidden_layers,
        )

        super().__init__(
            encoder=encoder,
            decoder=decoder,
            latent_dim=latent_dim,
            optimizer=optimizer,
            x_label=x_label,
            lr=lr,
            beta=beta,
            prior_mode=prior_mode,
            prior_logvar=prior_logvar,
            learn_prior_logvar=learn_prior_logvar,
        )

    def parse_batch(self, batch, parse_mask):
        x = batch.x.float()
        cellid = batch.cellid
        mask_dict = {"train": batch.train_mask, "val": batch.val_mask, "test": batch.test_mask}
        mask = mask_dict[parse_mask]

        return x, {"mask": mask, "cellid": cellid}

    def training_step(self, batch, batch_idx):
        results = self._step("train", batch, batch_idx, logger=True, parse_mask="train")
        return results

    def validation_step(self, batch, batch_idx):
        return self._step("val", batch, batch_idx, logger=True, parse_mask="val")

    def test_step(self, batch, batch_idx):
        return self._step("test", batch, batch_idx, logger=False, parse_mask="test")
