from typing import Optional, Sequence, Tuple, Union
from pathlib import Path

import random
import math
from scipy.stats import multivariate_normal

import json
import pandas as pd
import numpy as np
import torch
from pytorch_lightning import Callback, LightningModule, Trainer
from tqdm import trange, tqdm

from serotiny.losses import KLDLoss
from serotiny.utils.model_utils import q_batch, marginal_kl


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
        prior_mean: Optional[np.array],
        prior_logvar: Optional[np.array],
        mode: str = "isotropic"
        save_dir: Union[Path, str],
        verbose: bool = True,
    ):
        """
        """
        super().__init__()

        self.n_samples = n_samples
        self.save_dir = save_dir
        self.embedding_dim = embedding_dim
        self.prior_mean = prior_mean
        self.prior_logvar = prior_logvar
        self.mode = mode

        self.kl_div = KLDLoss(mode=mode)

    def on_test_epoch_end(self, trainer: Trainer, pl_module: LightningModule):

        if prior_mean is None:
            prior_mean = np.zeros(self.embedding_dim)
        if prior_var is None:
            prior_logvar = np.zeros(self.embedding_dim)

        marginal_kl = marginal_kl(pl_module, trainer.train_dataloader, prior_mean,
                                  prior_logvar, self.n_samples, self.x_label,
                                  self.c_label)

        with torch.no_grad():
            total_kld = 0
            for batch in dataloader:
                _, mu, logvar, _, _, _, _, _ = model.forward(
                    batch[self.x_label].float(),
                    batch[self.c_label].float()
                )

                total_kld += self.kl_div(mu, logvar, prior_mu=self.prior_mean,
                                         prior_logvar=self.prior_logvar)

        avg_kld = total_kld / dataset_size

        with open(save_dir / "elbo_terms.json", "w") as f:
            json.dump({
                "avg_kld": avg_kld,
                "marginal_kld": marginal_kld,
                "index-code mutual-information": avg_kld - marginal_kld,
                "logN": np.log(dataset_size)
            }, f)
