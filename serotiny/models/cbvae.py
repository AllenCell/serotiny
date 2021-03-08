"""
General conditional beta variational autoencoder module, implemented as a Pytorch Lightning module
"""
from pathlib import Path
import inspect

import torch
from torch.nn import functional as F
import pytorch_lightning as pl
from pytorch_lightning.utilities.parsing import get_init_args

from ..networks._2d import CBVAEEncoder as CBVAE2DEncoder
from ..networks._2d import CBVAEDecoder as CBVAE2DDecoder
from ..networks._3d import CBVAEEncoder as CBVAE3DEncoder
from ..networks._3d import CBVAEDecoder as CBVAE3DDecoder
from ..losses import KLDLoss
from .utils import index_to_onehot

AVAILABLE_OPTIMIZERS = {"adam": torch.optim.Adam, "sgd": torch.optim.SGD}
AVAILABLE_SCHEDULERS = {"reduce_lr_plateau": torch.optim.lr_scheduler.ReduceLROnPlateau}


def find_optimizer(optimizer_name, parameters, lr):
    if optimizer_name in AVAILABLE_OPTIMIZERS:
        optimizer_class = AVAILABLE_OPTIMIZERS[optimizer_name]
        optimizer = optimizer_class(parameters, lr=lr)
    else:
        raise KeyError(
            f"optimizer {optimizer_name} not available, "
            f"options are {list(AVAILABLE_OPTIMIZERS.keys())}"
        )
    return optimizer


def reparameterize(mu, log_var, add_noise=True):
    if add_noise:
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        out = mu + eps * std
    else:
        out = mu

    return out


class CBVAEModel(pl.LightningModule):
    def __init__(
        self,
        encoder,
        decoder,
        optimizer_encoder,
        optimizer_decoder,
        crit_recon,
        lr=1e-3,
        beta=1,
        kld_reduction="sum",
        x_label="x",
        class_label="class",
        num_classes=0,
        reference="ref",
        target_channels=0,
        reference_channels=[1, 2],
        input_dims=[28, 28],
        auto_padding=True,
    ):
        super().__init__()

        # store init args except for encoder & decoder, to avoid copying
        # large objects unnecessarily
        frame = inspect.currentframe()
        init_args = get_init_args(frame)
        self.save_hyperparameters(
            *[arg for arg in init_args if arg not in ["encoder", "decoder"]]
        )

        self.log_grads = True

        self.encoder = encoder
        self.decoder = decoder
        self.crit_recon = crit_recon
        self.kld_loss = KLDLoss(reduction=kld_reduction)
        self.beta = beta

    def parse_batch(self, batch):
        x = batch[self.hparams.x_label].float()
        print(x.shape)
        x_target = x[:, self.hparams.target_channels]
        if self.hparams.reference_channels is not None:
            x_reference = x[:,self.hparams.reference_channels]
        else:
            x_reference = None
        if self.hparams.num_classes > 0:
            x_class = batch[self.hparams.class_label]
            x_class_one_hot = index_to_onehot(x_class, self.hparams.num_classes)
        else:
            x_class_one_hot = None

        return x_target, x_reference, x_class_one_hot

    def forward(self, x_target, x_reference, x_class):
        #####################
        # train autoencoder
        #####################

        # because of the structure of the down residual layers, the output
        # shape of the convolutional portion of this network is going to be
        # equal to the input shape downscaled by np.ceil(dim/div)

        # Forward passes
        mu, logsigma = self.encoder(x_target, x_reference, x_class)

        kld_loss = self.kld_loss(mu, logsigma)

        z_latent = mu.data.cpu()
        z = reparameterize(mu, logsigma)

        x_hat = self.decoder(z, x_reference, x_class)

        if self.hparams.auto_padding:
            padding = [
                (x_dim - xhat_dim) // 2
                for x_dim, xhat_dim in zip(x_target.shape[2:], x_hat.shape[2:])
            ]

            # make padding symmetric in every dimension
            padding = sum([[p, p] for p in padding], [])
            x_hat = F.pad(x_hat, padding, mode="constant", value=0)

        recon_loss = self.crit_recon(x_hat, x_target)

        return x_hat, z_latent, recon_loss, kld_loss

    def training_step(self, batch, batch_idx, optimizer_idx):
        x_target, x_reference, x_class = self.parse_batch(batch)
        _, _, recon_loss, kld_loss = self(x_target, x_reference, x_class)

        loss = recon_loss + self.beta * kld_loss

        self.manual_backward(loss)
        (opt_enc, opt_dec) = self.optimizers()
        opt_enc.step()
        opt_dec.step()
        opt_enc.zero_grad()
        opt_dec.zero_grad()

        # Default logger=False for training_step
        # set it to true to log train loss to all lggers
        self.log("train reconstruction loss", recon_loss, logger=True)
        self.log("train kld loss", kld_loss, logger=True)

        return {"loss": loss, "batch_idx": batch_idx}

    def validation_step(self, batch, batch_idx):
        x_target, x_reference, x_class = self.parse_batch(batch)
        x_hat, z_latent, recon_loss, kld_loss = self(x_target, x_reference, x_class)
        loss = recon_loss + self.beta * kld_loss

        # Default logger=False for training_step
        # set it to true to log train loss to all lggers
        self.log("validation reconstruction loss", recon_loss, logger=True)
        self.log("validation kld loss", kld_loss, logger=True)

        return {
            "validation_loss": loss,
            "x_hat": x_hat,
            "x_class": x_class,
            "z_latent": z_latent,
            "batch_idx": batch_idx,
        }

    def test_step(self, batch, batch_idx):
        x_target, x_reference, x_class = self.parse_batch(batch)
        x_hat, z_latent, recon_loss, kld_loss = self(x_target, x_reference, x_class)
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
        encoder_optimizer = find_optimizer(
            self.hparams.optimizer_encoder, self.encoder.parameters(), self.hparams.lr
        )

        decoder_optimizer = find_optimizer(
            self.hparams.optimizer_decoder, self.decoder.parameters(), self.hparams.lr
        )

        return (
            {
                "optimizer": encoder_optimizer,
                "monitor": "val_accuracy",
                "interval": "epoch",
                "frequency": 1,
                "strict": True,
            },
            {
                "optimizer": decoder_optimizer,
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
