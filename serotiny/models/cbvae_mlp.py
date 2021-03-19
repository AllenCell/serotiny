"""
General conditional beta variational autoencoder module, 
implemented as a Pytorch Lightning module
"""
# import inspect

import torch
import pytorch_lightning as pl

# from pytorch_lightning.utilities.parsing import get_init_args
from pathlib import Path
from typing import Optional

from ..loss_formulations import calculate_elbo
from .utils import log_metrics

AVAILABLE_OPTIMIZERS = {"adam": torch.optim.Adam, "sgd": torch.optim.SGD}
AVAILABLE_SCHEDULERS = {"reduce_lr_plateau": torch.optim.lr_scheduler.ReduceLROnPlateau}


class CBVAEMLPModel(pl.LightningModule):
    def __init__(
        self,
        encoder,
        decoder,
        optimizer,
        scheduler,
        lr=1e-3,
        beta=1,
        x_label="cell_coeffs",
        c_label="structure",
        num_classes: Optional[int] = None,
        c_label_ind: Optional[str] = None,
        mask: Optional[bool] = True,
    ):
        super().__init__()

        # store init args except for encoder & decoder, to avoid copying
        # large objects unnecessarily
        # frame = inspect.currentframe()
        # init_args = get_init_args(frame)
        # self.save_hyperparameters(
        #     *[arg for arg in init_args if arg not in ["encoder", "decoder"]]
        # )
        self.save_hyperparameters()

        self.log_grads = True

        self.encoder = encoder
        self.decoder = decoder
        self.beta = beta
        self.mask = mask

    def parse_batch(self, batch):
        x = batch[self.hparams.x_label].float()
        x_cond = batch[self.hparams.c_label]
        # Get c label indices for gaussian dataset
        if self.hparams.c_label_ind is not None:
            x_cond_inds = batch[self.hparams.c_label_ind]
        else:
            x_cond_inds = torch.ones(x.shape[0])

        return x, x_cond, x_cond_inds

    def forward(self, x, x_cond, z_inference=None):
        # z is for inference
        #####################
        # train autoencoder
        #####################

        mu, logsigma, z = self.encoder(x, x_cond)
        if z_inference is not None:
            x_hat = self.decoder(z_inference, x_cond)
        else:
            x_hat = self.decoder(z, x_cond)

        loss, recon_loss, kld_loss, rcl_per_element, kld_per_element = calculate_elbo(
            x, x_hat, mu, logsigma, self.beta, mask=self.mask
        )

        return (
            x_hat,
            mu,
            logsigma,
            loss,
            recon_loss,
            kld_loss,
            kld_per_element,
            rcl_per_element,
        )

    def training_step(self, batch, batch_idx):
        x, x_cond, x_cond_inds = self.parse_batch(batch)
        # kld_elem is batch * num_latent_dims
        # rcl_elem is batch * Y shape of input
        x_hat, _, _, loss, recon_loss, kld_loss, kld_elem, rcl_elem = self(x, x_cond)

        # Default logger=False for training_step
        # set it to true to log train loss to all lggers
        self.log("train reconstruction loss", recon_loss, logger=True)
        self.log("train kld loss", kld_loss, logger=True)
        self.log("train_loss", loss, logger=True)

        return {
            "loss": loss,
            "train_loss": loss,  # for epoch end logging purposes
            "recon_loss": recon_loss,
            "kld_loss": kld_loss,
            "input": x,
            "recon": x_hat,
            "cond": x_cond,
            "cond_labels": x_cond_inds,
            "kld_per_elem": kld_elem,
            "rcl_per_elem": rcl_elem,
            "batch_idx": batch_idx,
        }

    def training_epoch_end(self, outputs):

        log_metrics(outputs, "train", self.current_epoch, Path(self.logger[1].save_dir))

    def validation_step(self, batch, batch_idx):
        x, x_cond, x_cond_inds = self.parse_batch(batch)
        x_hat, _, _, loss, recon_loss, kld_loss, kld_elem, rcl_elem = self(x, x_cond)

        # Default logger=False for training_step
        # set it to true to log train loss to all lggers
        self.log("validation reconstruction loss", recon_loss, logger=True)
        self.log("validation kld loss", kld_loss, logger=True)
        self.log("val_loss", loss, logger=True)

        return {
            "val_loss": loss,
            "recon_loss": recon_loss,
            "kld_loss": kld_loss,
            "input": x,
            "recon": x_hat,
            "cond": x_cond,
            "cond_labels": x_cond_inds,
            "kld_per_elem": kld_elem,
            "rcl_per_elem": rcl_elem,
            "batch_idx": batch_idx,
            "batch": batch,
        }

    def validation_epoch_end(self, outputs):

        log_metrics(outputs, "val", self.current_epoch, Path(self.logger[1].save_dir))

    def test_step(self, batch, batch_idx):
        x, x_cond, x_cond_inds = self.parse_batch(batch)
        x_hat, _, _, loss, recon_loss, kld_loss, kld_elem, rcl_elem = self(x, x_cond)

        # Default logger=False for training_step
        # set it to true to log train loss to all lggers
        self.log("test reconstruction loss", recon_loss, logger=True)
        self.log("test kld loss", kld_loss, logger=True)
        self.log("test_loss", loss, logger=True)

        return {
            "test_loss": loss,
            "recon_loss": recon_loss,
            "kld_loss": kld_loss,
            "input": x,
            "recon": x_hat,
            "cond": x_cond,
            "cond_labels": x_cond_inds,
            "kld_per_elem": kld_elem,
            "rcl_per_elem": rcl_elem,
            "batch_idx": batch_idx,
        }

    def test_epoch_end(self, outputs):

        log_metrics(outputs, "test", self.current_epoch, Path(self.logger[1].save_dir))

    def configure_optimizers(self):
        # with it like shceduler
        if self.hparams.optimizer in AVAILABLE_OPTIMIZERS:
            optimizer_class = AVAILABLE_OPTIMIZERS[self.hparams.optimizer]
            optimizer = optimizer_class(self.parameters(), lr=self.hparams.lr)
        else:
            raise Exception(
                (
                    f"optimizer {self.hparams.optimizer} not available, "
                    f"options are {list(AVAILABLE_OPTIMIZERS.keys())}"
                )
            )

        if self.hparams.scheduler in AVAILABLE_SCHEDULERS:
            scheduler_class = AVAILABLE_SCHEDULERS[self.hparams.scheduler]
            scheduler = scheduler_class(optimizer)
        else:
            raise Exception(
                (
                    f"scheduler {self.hparams.scheduler} not available, "
                    f"options are {list(AVAILABLE_SCHEDULERS.keys())}"
                )
            )

        return (
            {
                "optimizer": optimizer,
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
                    tag=name, values=grads, global_step=self.trainer.global_step
                )