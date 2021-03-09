"""
General conditional beta variational autoencoder module, 
implemented as a Pytorch Lightning module
"""
import inspect

import torch
import pytorch_lightning as pl
from pytorch_lightning.utilities.parsing import get_init_args

from ..losses import KLDLoss
from .utils import index_to_onehot

AVAILABLE_OPTIMIZERS = {"adam": torch.optim.Adam, "sgd": torch.optim.SGD}
AVAILABLE_SCHEDULERS = {"reduce_lr_plateau": torch.optim.lr_scheduler.ReduceLROnPlateau}


class CBVAELinearModel(pl.LightningModule):
    def __init__(
        self,
        network,
        optimizer,
        scheduler,
        crit_recon,
        lr=1e-3,
        beta=1,
        kld_reduction="sum",
        x_label="cell_coeffs",
        y_label="structure",
        num_classes=25,
    ):
        super().__init__()

        # store init args except for encoder & decoder, to avoid copying
        # large objects unnecessarily
        frame = inspect.currentframe()
        init_args = get_init_args(frame)
        self.save_hyperparameters(*[arg for arg in init_args if arg not in ["network"]])

        self.log_grads = True

        self.network = network
        self.crit_recon = crit_recon
        self.kld_loss = KLDLoss(reduction=kld_reduction)
        self.mse_loss = torch.nn.MSELoss(size_average=False)
        self.beta = beta

    def parse_batch(self, batch):
        x = batch[self.hparams.x_label].float()
        x_class = batch[self.hparams.y_label]
        x_class = x_class.reshape(x_class.shape[0], 1)
        x_class_one_hot = index_to_onehot(x_class, self.hparams.num_classes)

        return x, x_class_one_hot

    def forward(self, x, x_class):
        #####################
        # train autoencoder
        #####################

        # Forward passes
        x_hat, mu, logsigma = self.network(x, x_class)

        kld_loss = self.kld_loss(mu, logsigma)

        z_latent = mu.data.cpu()

        # recon_loss = self.crit_recon(x_hat, x)
        recon_loss = self.mse_loss(x_hat, x)

        return x_hat, z_latent, recon_loss, kld_loss

    def training_step(self, batch, batch_idx):
        x, x_class = self.parse_batch(batch)
        _, _, recon_loss, kld_loss = self(x, x_class)

        loss = recon_loss + self.beta * kld_loss

        # Default logger=False for training_step
        # set it to true to log train loss to all lggers
        self.log("train reconstruction loss", recon_loss, logger=True)
        self.log("train kld loss", kld_loss, logger=True)

        return {"loss": loss, "batch_idx": batch_idx}

    def validation_step(self, batch, batch_idx):
        x, x_class = self.parse_batch(batch)
        x_hat, z_latent, recon_loss, kld_loss = self(x, x_class)
        loss = recon_loss + self.beta * kld_loss

        # Default logger=False for training_step
        # set it to true to log train loss to all lggers
        self.log("validation reconstruction loss", recon_loss, logger=True)
        self.log("validation kld loss", kld_loss, logger=True)

        return {
            "val_loss": loss,
            "x_hat": x_hat,
            "x_class": x_class,
            "z_latent": z_latent,
            "batch_idx": batch_idx,
        }

    def test_step(self, batch, batch_idx):
        x, x_class = self.parse_batch(batch)
        x_hat, z_latent, recon_loss, kld_loss = self(x, x_class)
        loss = recon_loss + self.beta * kld_loss

        # Default logger=False for training_step
        # set it to true to log train loss to all lggers
        self.log("test reconstruction loss", recon_loss, logger=True)
        self.log("test kld loss", kld_loss, logger=True)

        return {
            "test_loss": loss,
            "x_hat": x_hat,
            "x_class": x_class,
            "z_latent": z_latent,
            "batch_idx": batch_idx,
        }

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
