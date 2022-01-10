from typing import Union, Optional, Sequence

import torch
import numpy as np

from serotiny.networks.gcn import GCN
from serotiny.networks.mlp import MLP
from serotiny.models.vae import BaseVAE

Array = Union[torch.Tensor, np.array, Sequence[float]]


class GraphVAE(BaseVAE):
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
        encoder = GCN(x_dim, 2 * latent_dim, hidden_layers=hidden_layers,)

        decoder = MLP(latent_dim, x_dim, hidden_layers=hidden_layers,)

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
        edge_index = batch.edge_index
        edge_weight = batch.edge_weight
        cellid = batch.cellid
        if parse_mask == "train":
            mask = batch.train_mask
        elif parse_mask == "val":
            mask = batch.val_mask
        elif parse_mask == "test":
            mask = batch.test_mask

        return x, {"edge_index": edge_index, "edge_weight": edge_weight, "mask": mask, "cellid": cellid}

    def training_step(self, batch, batch_idx):
        results = self._step("train", batch, batch_idx, logger=True, parse_mask="train")
        return results

    def validation_step(self, batch, batch_idx):
        return self._step("val", batch, batch_idx, logger=True, parse_mask="val")

    def test_step(self, batch, batch_idx):
        return self._step("test", batch, batch_idx, logger=False, parse_mask="test")
