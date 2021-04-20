from typing import Optional
from pathlib import Path

import json
import numpy as np
import torch
from pytorch_lightning import Callback, LightningModule, Trainer

from serotiny.utils.model_utils import marginal_kl
from serotiny.utils.model_utils import to_device
from serotiny.loss_formulations.elbo import diagonal_gaussian_kl

def batch_update_stats(mu, var, z_batch, n_samples):
    batch_size = len(z_batch)

    old_weight = (n_samples/(n_samples + batch_size)
    new_weight = 1 - old_weight
    weight_prod = old_weight * new_weight

    new_mu = old_weight * mu + new_weight * z_batch.mean(axis=0)
    new_var = (
        old_weight * var +
        new_weight * z_batch.var(axis=0) +
        weight_prod * (new_mu - mu).pow(2)
    )

    return new_mu, new_var, n_samples + batch_size


class EmpiricalKL(Callback):
    """
    Callback to be used at the end of training, to compute kl divergence
    between empirical train q(z) and test q(z)
    """

    def __init__(
        self,
        x_label: str,
        c_label: str,
        embedding_dim: int,
        verbose: bool = True,
    ):
        """"""
        super().__init__()

        self.embedding_dim = embedding_dim
        self.x_label = x_label
        self.c_label = c_label

        self.kl_div = KLDLoss(mode=mode, reduction="sum")

    def on_test_epoch_end(self, trainer: Trainer, pl_module: LightningModule):

        save_dir = Path(trainer.logger[1].save_dir)


        with torch.no_grad():
            train_mu = torch.zeros(self.embedding_dim).to(pl_module.device)
            train_var = torch.zeros(self.embedding_dim).to(pl_module.device)
            n_train_samples = 0
            for batch in trainer.train_dataloader:
                this_x, this_c = to_device(
                    batch[self.x_label].float(),
                    batch[self.c_label].float(),
                    pl_module.device,
                )
                _, mu, _, _, _, _, _, _ = pl_module(this_x, this_c)

                train_mu, train_var, n_train_samples = batch_update_stats(
                    train_mu, train_var, mu, n_train_samples
                )

            test_mu = torch.zeros(self.embedding_dim).to(pl_module.device)
            test_var = torch.zeros(self.embedding_dim).to(pl_module.device)
            n_test_samples = 0
            for batch in trainer.test_dataloaders[0]:
                this_x, this_c = to_device(
                    batch[self.x_label].float(),
                    batch[self.c_label].float(),
                    pl_module.device,
                )
                _, mu, _, _, _, _, _, _ = pl_module(this_x, this_c)

                test_mu, test_var, n_test_samples = batch_update_stats(
                    test_mu, test_var, mu, n_test_samples
                )

        symmetric_kl = 0.5 * (
            diagonal_gaussian_kl(train_mu, train_var.log(), test_mu, test_var.log()) +
            diagonal_gaussian_kl(test_mu, test_var.log(), train_mu, train_var.log())
        ).sum(axis=1).mean()

        with open(save_dir / "train_test_empirical_kl.json", "w") as f:
            json.dump(
                {
                    "train_test_empirical_kl": symmetric_kl.item(),
                    "train_mu": train_mu.cpu().numpy().tolist(),
                    "train_var": train_var.cpu().numpy().tolist(),
                    "test_mu": test_mu.cpu().numpy().tolist(),
                    "test_var": test_var.cpu().numpy().tolist(),
                },
                f,
            )
