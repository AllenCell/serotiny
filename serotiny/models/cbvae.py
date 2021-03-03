"""
General conditional beta variational autoencoder module, implemented as a Pytorch Lightning module
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Optional
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
import pytorch_lightning as pl

from ..data import index_to_onehot
from ..networks._2d import CBVAEEncoder as CBVAE2DEncoder
from ..networks._2d import CBVAEDecoder as CBVAE2DDecoder
from ..networks._3d import CBVAEEncoder as CBVAE3DEncoder
from ..networks._3d import CBVAEDecoder as CBVAE3DDecoder
from ..losses import KLDLoss
# from ._utils import acc_prec_recall, add_pr_curve_tensorboard

AVAILABLE_NETWORKS = {
    "cbvae_2d_encoder": CBVAE2DEncoder,
    "cbvae_2d_decoder": CBVAE2DDecoder,
    "cbvae_3d_encoder": CBVAE3DEncoder,
    "cbvae_3d_decoder": CBVAE3DDecoder,
}

AVAILABLE_OPTIMIZERS = {"adam": torch.optim.Adam, "sgd": torch.optim.SGD}
AVAILABLE_SCHEDULERS = {
    "reduce_lr_plateau": torch.optim.lr_scheduler.ReduceLROnPlateau
}

def find_optimizer(optimizer_name, parameters, lr):
    if optimizer_name in AVAILABLE_OPTIMIZERS:
        optimizer_class = AVAILABLE_OPTIMIZERS[optimizer_name]
        optimizer = optimizer_class(parameters, lr=lr)
    else:
        raise KeyError(
            f"optimizer {optimizer_name} not available, "
            f"options are {list(AVAILABLE_OPTIMIZERS.keys())}"
        )



class CBVAEModel(pl.LightningModule):
    def __init__(
        self,
        encoder,
        decoder,
        optimizer_encoder,
        optimizer_decoder,
        crit_recon,
        beta=1,
        beta_start=0,
        beta_step=1e-5,
        beta_min=0,
        beta_max=1,
        beta_n_cycles=10,
        beta_warmup_time=0.5,
        alpha=0.5,
        c_max=500,
        c_iters_max=80000,
        gamma=500,
        objective="H",
        kld_reduction="sum",
        x_label="x",
        class_label='class',
        num_classes=2,
        reference='ref',
        target_channel=0,
        reference_channels=[1, 2],
    ):
        super().__init__()
        """Save hyperparameters"""
        # Can be accessed via checkpoint['hyper_parameters']
        self.save_hyperparameters()

        """Configs"""
        self.log_grads = True

        """model"""
        self.network = network
        self.encoder = encoder
        self.decoder = decoder
        self.crit_recon = crit_recon
        self.kld_loss = KLDLoss(reduction=kld_reduction)

    def parse_batch(self, batch):
        x = batch[self.hparams.x_label].float()
        x_target = x[self.hparams.target_channel]
        x_reference = x[self.hparams.reference_channels]
        x_class = x[self.hparams.class_label]
        one_hot = index_to_onehot(x_class, self.hparams.num_classes)
        
        # Return floats because this is expected dtype for nn.Loss
        return x_target, x_reference, one_hot

    def forward(self, x_target, x_reference, x_class):
        #####################
        # train autoencoder
        #####################

        # Forward passes
        mu, logsigma = self.encoder(x_target, x_reference, x_class)

        kld_loss = self.kld_loss(mu, logsigma)

        z_latent = mu.data.cpu()
        z = self.reparameterize(mu, logsigma)

        x_hat = self.decoder(z, ref, classes_onehot)

        recon_loss = self.crit_recon(x_hat, x)
        loss = [recon_loss, kld_loss]

        return x_hat, z_latent, loss

    def training_step(self, batch, batch_idx):
        x_target, x_reference, x_class = self.parse_batch(batch)
        x_hat, z_latent, loss_terms = self(x)
        recon_loss, kld_loss = loss_terms
        loss = torch.sum(loss_terms)

        # Default logger=False for training_step
        # set it to true to log train loss to all lggers
        self.log("train reconstruction loss", recon_loss, logger=True)
        self.log("train kld loss", kld_loss, logger=True)

        return {
            'loss': loss,
            'batch_idx': batch_idx
        }

    def validation_step(self, batch, batch_idx):
        x_target, x_reference, x_class = self.parse_batch(batch)
        x_hat, z_latent, loss_terms = self(x)
        recon_loss, kld_loss = loss_terms
        loss = torch.sum(loss_terms)

        # Default logger=False for training_step
        # set it to true to log train loss to all lggers
        self.log("validation reconstruction loss", recon_loss, logger=True)
        self.log("validation kld loss", kld_loss, logger=True)

        return {
            'validation_loss': loss,
            'x_hat': x_hat,
            'x_class': x_class
            'z_latent': z_latent,
            'batch_idx': batch_idx
        }

    def test_step(self, batch, batch_idx):
        x_target, x_reference, x_class = self.parse_batch(batch)
        x_hat, z_latent, loss_terms = self(x)
        recon_loss, kld_loss = loss_terms
        loss = torch.sum(loss_terms)

        # Default logger=False for training_step
        # set it to true to log train loss to all lggers
        self.log("test reconstruction loss", recon_loss, logger=True)
        self.log("test kld loss", kld_loss, logger=True)

        return {
            'test_loss': loss,
            'x_hat': x_hat,
            'x_class': x_class
            'z_latent': z_latent,
            'batch_idx': batch_idx
        }

    def configure_optimizers(self):
        encoder_optimizer = find_optimizer(
            self.hparams.optimizer_encoder,
            self.encoder.parameters(),
            self.hparams.lr)

        decoder_optimizer = find_optimizer(
            self.hparams.optimizer_decoder,
            self.decoder.parameters(),
            self.hparams.lr)

        if self.hparams.scheduler in AVAILABLE_SCHEDULERS:
            scheduler_class = AVAILABLE_SCHEDULERS[self.hparams.scheduler]
            scheduler = scheduler_class(optimizer)
        else:
            raise Exception(
                f"scheduler {self.hparams.scheduler} not available, "
                f"options are {list(AVAILABLE_SCHEDULERS.keys())}"
            )

        return (
            {
                "optimizer": encoder_optimizer,
                "scheduler": scheduler,
                "monitor": "val_accuracy",
                "interval": "epoch",
                "frequency": 1,
                "strict": True,
            },
            {
                "optimizer": decoder_optimizer,
                "scheduler": scheduler,
                "monitor": "val_accuracy",
                "interval": "epoch",
                "frequency": 1,
                "strict": True,
            },
        )

    def on_after_backward(self):
        # example to inspect gradient information in tensorboard
        if (
            self.log_grads and self.trainer.global_step % 100 == 0
        ):  # don't make the file huge
            params = self.state_dict()
            for k, v in params.items():
                grads = v
                name = k
                # logger[0] is tensorboard logger
                self.logger[0].experiment.add_histogram(
                    tag=name,
                    values=grads,
                    global_step=self.trainer.global_step
                )

