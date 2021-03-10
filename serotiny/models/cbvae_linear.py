"""
General conditional beta variational autoencoder module, 
implemented as a Pytorch Lightning module
"""
import inspect

import torch
import pytorch_lightning as pl
from pytorch_lightning.utilities.parsing import get_init_args
from pathlib import Path
from typing import Optional

from ..losses import KLDLoss
import pandas as pd

AVAILABLE_OPTIMIZERS = {"adam": torch.optim.Adam, "sgd": torch.optim.SGD}
AVAILABLE_SCHEDULERS = {"reduce_lr_plateau": torch.optim.lr_scheduler.ReduceLROnPlateau}


class CBVAELinearModel(pl.LightningModule):
    def __init__(
        self,
        encoder,
        decoder,
        optimizer,
        scheduler,
        lr=1e-3,
        beta=1,
        kld_reduction="sum",
        x_label="cell_coeffs",
        c_label="structure",
        num_classes: Optional[int] = None,
        c_label_ind: Optional[str] = None,
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
        self.kld_loss = KLDLoss(reduction=kld_reduction)
        self.mse_loss = torch.nn.MSELoss(size_average=False)
        self.mse_loss_per_element = torch.nn.MSELoss(size_average=False, reduce=False)
        self.beta = beta

    def parse_batch(self, batch):
        x = batch[self.hparams.x_label].float()
        x_cond = batch[self.hparams.c_label]
        # Get c label indices for gaussian dataset
        if self.hparams.c_label_ind is not None:
            x_cond_inds = batch[self.hparams.c_label_ind]
        else:
            x_cond_inds = torch.ones(x.shape[0])

        return x, x_cond, x_cond_inds

    def _log_metrics(self, outputs, prefix, CSV_logger=False):
        """
        Produce metrics and save them
        """

        dataframe = {
            "epoch": [],
            f"total_{prefix}_losses": [],
            f"total_{prefix}_ELBO": [],
            f"total_{prefix}_rcl": [],
            f"total_{prefix}_klds": [],
            "num_conds": [],
            f"{prefix}_rcl": [],
            f"{prefix}_kld": [],
        }

        batch_rcl, batch_kld = (
            torch.empty([0]),
            torch.empty([0]),
        )

        batch_length = 0
        loss, rcl_loss, kld_loss = 0, 0, 0

        total_x = torch.stack([x["input"] for x in outputs])
        num_batches = len(total_x)

        rcl_per_condition_loss, kld_per_condition_loss = (
            torch.zeros(total_x.size()[-1] + 1),
            torch.zeros(total_x.size()[-1] + 1),
        )

        for output in outputs:
            rcl_per_element = output["rcl_per_elem"]
            kld_per_element = output["kld_per_elem"]
            loss += output[f"{prefix}_loss"].item()
            rcl_loss += output["recon_loss"].item()
            kld_loss += output["kld_loss"].item()

            for jj, ii in enumerate(torch.unique(output["cond_labels"])):
                this_cond_positions = output["cond_labels"] == ii

                batch_rcl = torch.cat(
                    [
                        batch_rcl.type_as(rcl_per_element),
                        torch.sum(rcl_per_element[this_cond_positions], dim=0).view(
                            1, -1
                        ),
                    ],
                    0,
                )
                batch_kld = torch.cat(
                    [
                        batch_kld.type_as(kld_per_element),
                        torch.sum(kld_per_element[this_cond_positions], dim=0).view(
                            1, -1
                        ),
                    ],
                    0,
                )
                batch_length += 1

                this_cond_rcl = torch.sum(rcl_per_element[this_cond_positions])
                this_cond_kld = torch.sum(kld_per_element[this_cond_positions])

                rcl_per_condition_loss[jj] += this_cond_rcl.item()
                kld_per_condition_loss[jj] += this_cond_kld.item()

        loss = loss / num_batches
        rcl_loss = rcl_loss / num_batches
        kld_loss = kld_loss / num_batches
        rcl_per_condition_loss = rcl_per_condition_loss / num_batches
        kld_per_condition_loss = kld_per_condition_loss / num_batches

        for j in range(len(rcl_per_condition_loss)):
            dataframe["epoch"].append(self.current_epoch)
            dataframe["num_conds"].append(j)
            dataframe[f"total_{prefix}_losses"].append(loss)
            dataframe[f"total_{prefix}_ELBO"].append(rcl_loss + kld_loss)
            dataframe[f"total_{prefix}_rcl"].append(rcl_loss)
            dataframe[f"total_{prefix}_klds"].append(kld_loss)
            dataframe[f"{prefix}_rcl"].append(rcl_per_condition_loss[j].item())
            dataframe[f"{prefix}_kld"].append(kld_per_condition_loss[j].item())

        stats = pd.DataFrame(dataframe)

        path = (
            Path(self.logger[1].save_dir)
            / Path(self.logger[1].version)
            / f"stats_{prefix}.csv"
        )
        if prefix == "train":
            print(f"====> {prefix}")
            print("====> Epoch: {} losses: {:.4f}".format(self.current_epoch, loss))
            print("====> RCL loss: {:.4f}".format(rcl_loss))
            print("====> KLD loss: {:.4f}".format(kld_loss))

        if path.exists():
            stats.to_csv(path, mode="a", header=False, index=False)
        else:
            stats.to_csv(path, header="column_names", index=False)

        if prefix == "test":
            dataframe2 = {
                "epoch": [],
                "dimension": [],
                "test_kld_per_dim": [],
                "num_conds": [],
            }

            for j in range(len(rcl_per_condition_loss)):
                # this_cond_rcl = test_batch_rcl[j::X_train.shape[2] + 1, :]
                this_cond_kld = batch_kld[j :: total_x.shape[2] + 1, :]

                # summed_rcl = torch.sum(this_cond_rcl, dim=0) / batch_num_test
                summed_kld = torch.sum(this_cond_kld, dim=0) / batch_length

                for k in range(len(summed_kld)):
                    dataframe2["epoch"].append(self.current_epoch)
                    dataframe2["dimension"].append(k)
                    dataframe2["num_conds"].append(j)
                    dataframe2["test_kld_per_dim"].append(summed_kld[k].item())

            stats_per_dim = pd.DataFrame(dataframe2)

            path2 = (
                Path(self.logger[1].save_dir)
                / Path(self.logger[1].version)
                / f"stats_per_dim_{prefix}.csv"
            )
            if path2.exists():
                stats_per_dim.to_csv(path2, mode="a", header=False, index=False)
            else:
                stats_per_dim.to_csv(path2, header="column_names", index=False)

        return outputs

    def forward(self, x, x_cond, z_inference=None):
        # z is for inference
        #####################
        # train autoencoder
        #####################

        # Forward passes
        # x_hat, mu, logsigma = self.network(x, x_cond)
        mu, logsigma, z = self.encoder(x, x_cond)
        if z_inference:
            x_hat = self.decoder(z_inference, x_cond)
        else:
            x_hat = self.decoder(z, x_cond)

        # if x is masked, find indices in x
        # that are 0 and set reconstructed x to 0 as well
        indices = x == 0
        x_hat[indices] = 0

        kld_loss = self.kld_loss(mu, logsigma)
        kld_per_element = -0.5 * (1 + logsigma - mu.pow(2) - logsigma.exp())

        z_latent = mu.data.cpu()

        recon_loss = self.mse_loss(x_hat, x)
        rcl_per_element = self.mse_loss_per_element(x_hat, x)

        return x_hat, z_latent, recon_loss, kld_loss, kld_per_element, rcl_per_element

    def training_step(self, batch, batch_idx):
        x, x_cond, x_cond_inds = self.parse_batch(batch)
        x_hat, _, recon_loss, kld_loss, kld_elem, rcl_elem = self(x, x_cond)

        loss = recon_loss + self.beta * kld_loss

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

        # No CSV metric logging needed here
        outputs = self._log_metrics(outputs, "train", CSV_logger=False)

    def validation_step(self, batch, batch_idx):
        x, x_cond, x_cond_inds = self.parse_batch(batch)
        x_hat, z_latent, recon_loss, kld_loss, kld_elem, rcl_elem = self(x, x_cond)
        loss = recon_loss + self.beta * kld_loss

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
        }

    def validation_epoch_end(self, outputs):

        # No CSV metric logging needed here
        outputs = self._log_metrics(outputs, "val", CSV_logger=False)

    def test_step(self, batch, batch_idx):
        x, x_cond, x_cond_inds = self.parse_batch(batch)
        x_hat, z_latent, recon_loss, kld_loss, kld_elem, rcl_elem = self(x, x_cond)
        loss = recon_loss + self.beta * kld_loss

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
        # No CSV metric logging needed here
        outputs = self._log_metrics(outputs, "test", CSV_logger=False)

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
