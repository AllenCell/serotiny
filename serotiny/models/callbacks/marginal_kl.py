from typing import Optional
from pathlib import Path

import json
import numpy as np
import torch
from pytorch_lightning import Callback, LightningModule, Trainer

from serotiny.utils.model_utils import marginal_kl
from serotiny.utils.model_utils import to_device
from serotiny.losses.elbo import diagonal_gaussian_kl


class MarginalKL(Callback):
    """
    Callback to be used at the end of training, to compute marginal kl divergence
    and index-code mutual information, as proposed in
    http://approximateinference.org/accepted/HoffmanJohnson2016.pdf
    """

    def __init__(
        self,
        n_samples: int,
        x_label: str,
        c_label: str,
        embedding_dim: int,
        mode: str = "isotropic",
        verbose: bool = True,
        prior_mean: Optional[np.array] = None,
        prior_logvar: Optional[np.array] = None,
    ):
        """"""
        super().__init__()

        self.n_samples = n_samples
        self.embedding_dim = embedding_dim
        self.prior_mean = prior_mean
        self.prior_logvar = prior_logvar
        self.mode = mode
        self.x_label = x_label
        self.c_label = c_label


    def on_test_epoch_end(self, trainer: Trainer, pl_module: LightningModule):

        save_dir = Path(trainer.logger[1].save_dir)

        if self.prior_mean is None:
            self.prior_mean = np.zeros(self.embedding_dim)
        if self.prior_logvar is None:
            self.prior_logvar = np.zeros(self.embedding_dim)

        marginal_kl_result = marginal_kl(
            pl_module,
            trainer.train_dataloader,
            self.prior_mean,
            self.prior_logvar,
            self.n_samples,
            self.x_label,
            self.c_label,
        )

        with torch.no_grad():
            train_dataloader = trainer.train_dataloader
            total_kld = 0
            for batch in train_dataloader:
                this_x, this_c = to_device(
                    batch[self.x_label].float(),
                    batch[self.c_label].float(),
                    pl_module.device,
                )
                _, mu, logvar, _, _, _, _, _ = pl_module(this_x, this_c)

                total_kld += diagonal_gaussian_kl(
                    mu,
                    prior_mu=self.prior_mean,
                    logvar,
                    prior_logvar=self.prior_logvar,
                )

        dataset_size = len(train_dataloader.dataset)
        avg_kld = total_kld / dataset_size
        # import ipdb
        # ipdb.set_trace()
        with open(save_dir / "elbo_terms.json", "w") as f:
            json.dump(
                {
                    "avg_kld": avg_kld.item(),
                    "marginal_kld": marginal_kl_result.item(),
                    "index-code mutual-information": avg_kld.item()
                    - marginal_kl_result.item(),
                    "logN": np.log(dataset_size),
                },
                f,
            )
