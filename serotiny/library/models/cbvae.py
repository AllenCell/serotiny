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

from ..networks._2d import CBVAEEncoder as CBVAE2DEncoder
from ..networks._2d import CBVAEDecoder as CBVAE2DDecoder
from ..networks._3d import CBVAEEncoder as CBVAE3DEncoder
from ..networks._3d import CBVAEDecoder as CBVAE3DDecoder
from ._utils import acc_prec_recall, add_pr_curve_tensorboard

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


class CBVAEModel(pl.LightningModule):
    def __init__(
        self,
        encoder,
        decoder,
        opt_enc,
        opt_dec,
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

    def parse_batch(self, batch):
        x = batch[self.hparams.x_label].float()
        x_target = x[self.hparams.target_channel]
        x_reference = x[self.hparams.reference_channels]
        x_class = x[self.hparams.class_label]
        
        # Return floats because this is expected dtype for nn.Loss
        return x_target, x_reference, x_class

    def forward(self, x_target, x_reference, x_class):
        classes = x_class.type_as(x_target).long()
        classes_onehot = utils.index_to_onehot(
            classes, self.data_provider.get_n_classes()
        )
        ref = ref.cuda()

        for p in enc.parameters():
            p.requires_grad = True

        for p in dec.parameters():
            p.requires_grad = True

        opt_enc.zero_grad()
        opt_dec.zero_grad()

        #####################
        # train autoencoder
        #####################

        # Forward passes
        mu, logsigma = enc(x, ref, classes_onehot)

        kld_loss = self.kld_loss(mu, logsigma)

        zLatent = mu.data.cpu()

        z = self.reparameterize(mu, logsigma)

        xHat = dec(z, ref, classes_onehot)

        # output = self.network(x)
        # return output

    def _log_metrics(self, outputs, prefix, CSV_logger=False):

        logs = self._generate_logs(
            outputs['preds'],
            outputs['target'],
            prefix,
            self.metrics[prefix],
        )
        outputs.update(logs)

        # Tensorboard logging by default
        for key, value in logs.items():
            self.logger[0].experiment.add_scalar(key, value, self.global_step)

        # CSV logging
        if CSV_logger:
            for key, value in logs.items():
                # self.log logs to all loggers
                self.log(key, value, on_epoch=True)


            probs = F.softmax(outputs["preds"], dim=1).cpu().numpy()
            pred = np.expand_dims(np.argmax(probs, axis=1), 1)
            target = np.expand_dims(outputs['target'].cpu().numpy(), 1)

            df = pd.DataFrame(
                np.hstack([probs, pred, target]),
                columns=[f"prob_{i}" for i in range(len(self.hparams.classes))] +
                        ["pred", "target"]
            )

            path = Path(self.logger[1].save_dir) / "test_results.csv"
            if path.exists():
                df.to_csv(path, mode='a', header=False, index=False)
            else:
                df.to_csv(path, header="column_names", index=False)

        if outputs['batch_idx'] == 0:
            _, preds_batch = torch.max(outputs['preds'], 1)
            conf_mat = pl.metrics.functional.confusion_matrix(
                preds_batch,
                outputs['target'],
                num_classes=len(self.hparams.classes)
            )

            fig, ax = plt.subplots(1, 1, figsize=(10, 10))
            sns.heatmap(
                conf_mat.cpu(),
                cmap="RdBu_r",
                annot=True, ax=ax,
                annot_kws={"size": 8}
            )
            # logger[0] is tensorboard logger
            self.logger[0].experiment.add_figure(
                f"Confusion matrix {prefix}",
                fig,
                global_step=self.current_epoch,
            )

        return outputs

    def training_step(self, batch, batch_idx):
        x, y = self.parse_batch(batch)
        if self.current_epoch == 0:
            self.reference_image = x
            self.reference_label = y
        yhat = self(x)
        # TODO configure loss
        # This depends on if y is one hot or not, also on the last
        # layer of network
        loss = nn.CrossEntropyLoss()(yhat, y)

        # Default logger=False for training_step
        # set it to true to log train loss to all lggers
        self.log("train_loss", loss, logger=True)

        return {
            'loss': loss,
            'preds': yhat,
            'target': y,
            'batch_idx': batch_idx
        }

    def training_step_end(self, outputs):

        # No CSV metric logging needed here
        outputs = self._log_metrics(
            outputs,
            "train",
            CSV_logger=False
        )
        return outputs

    def validation_step(self, batch, batch_idx):

        x, y = self.parse_batch(batch)
        # if batch_idx == 0:
        #     self.logger.experiment.add_figure(
        #         "predictions vs actuals (val).",
        #         plot_classes_preds(
        # self.network, 
        # x, 
        # y, 
        # self.hparams.classes),
        #         global_step=self.current_epoch,
        #     )
        yhat = self(x)
        loss = nn.CrossEntropyLoss()(yhat, y)
        # Default logger for validation step is True
        self.log("val_loss", loss)

        return {
            'val_loss': loss,
            'preds': yhat,
            'target': y,
            'batch_idx': batch_idx
        }

    def validation_step_end(self, outputs):

        # No CSV metric logging needed here
        outputs = self._log_metrics(
            outputs, 
            "val",
            CSV_logger=False
        )

        return outputs

    def validation_epoch_end(self, outputs):

        avg_loss = torch.stack(
            [x["val_loss"] for x in outputs]).mean()
        avg_acc = torch.stack(
            [x["val_accuracy"] for x in outputs]).mean()

        val_epoch_end_metrics = {
            "val_loss_epoch":
            avg_loss, 
            "val_accuracy_epoch":
            avg_acc, 
        }
        for key, value in val_epoch_end_metrics.items():
            self.logger[0].experiment.add_scalar(key, value, self.global_step)

        class_probs = []
        class_preds = []
        for pred in outputs:
            pred = pred["preds"]
            class_probs_batch = [F.softmax(el, dim=0) for el in pred]
            _, class_preds_batch = torch.max(pred, 1)
            class_probs.append(class_probs_batch)
            class_preds.append(class_preds_batch)
        test_probs = torch.cat([torch.stack(batch) for batch in class_probs])
        test_preds = torch.cat(class_preds)
        # plot all the pr curves
        for i in range(len(self.hparams.classes)):
            add_pr_curve_tensorboard(
                logger=self.logger[0], 
                # logger[0] is tensorboard logger
                classes=self.hparams.classes, 
                class_index=i, 
                test_probs=test_probs, 
                test_preds=test_preds, 
                global_step=self.current_epoch
            )

    def test_step(self, batch, batch_idx):

        x, y = self.parse_batch(batch)

        yhat = self(x)
        loss = nn.CrossEntropyLoss()(yhat, y)

        self.log("test_loss", loss)

        return {
            "test_loss": loss,
            'preds': yhat,
            'target': y,
            'batch_idx': batch_idx
        }

    def test_step_end(self, outputs):

        # Log test metrics to CSV
        outputs = self._log_metrics(
            outputs, 
            "test",
            CSV_logger=True
        )

        return outputs

    def test_epoch_end(self, outputs):
        class_probs = []
        class_preds = []
        for pred in outputs:
            pred = pred["preds"]
            class_probs_batch = [F.softmax(el, dim=0) for el in pred]
            _, class_preds_batch = torch.max(pred, 1)
            class_probs.append(class_probs_batch)
            class_preds.append(class_preds_batch)
        test_probs = torch.cat([torch.stack(batch) for batch in class_probs])
        test_preds = torch.cat(class_preds)
        # plot all the pr curves
        for i in range(len(self.hparams.classes)):
            add_pr_curve_tensorboard(
                logger=self.logger[0], 
                # logger[0] is tensorboard logger 
                classes=self.hparams.classes, 
                class_index=i, 
                test_probs=test_probs, 
                test_preds=test_preds, 
                global_step=self.current_epoch,
                name="_test",
            )

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
                    tag=name,
                    values=grads,
                    global_step=self.trainer.global_step
                )

