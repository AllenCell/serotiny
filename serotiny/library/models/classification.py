"""
General 2d classifier module, implemented as a Pytorch Lightning module
"""

import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F
import pytorch_lightning as pl

from ..networks._2d import BasicNeuralNetwork, ResNet18Network
from ..networks._3d import BasicCNN_3D
from ._utils import acc_prec_recall, add_pr_curve_tensorboard

AVAILABLE_NETWORKS = {
    "basic": BasicNeuralNetwork, 
    "resnet18": ResNet18Network, 
    "basic3D": BasicCNN_3D
}
AVAILABLE_OPTIMIZERS = {"adam": torch.optim.Adam, "sgd": torch.optim.SGD}
AVAILABLE_SCHEDULERS = {
    "reduce_lr_plateau": torch.optim.lr_scheduler.ReduceLROnPlateau
}


class ClassificationModel(pl.LightningModule):
    def __init__(
        self,
        network,
        x_label="projection_image",
        y_label="mitotic_class",
        in_channels=3,
        classes=("M0", "M1/M2", "M3", "M4/M5", "M6/M7"),
        image_x=104,
        image_y=176,
        lr=1e-3,
        optimizer="adam",
        scheduler="reduce_lr_plateau",
        label: Optional[str] = None,
        projection: Optional[dict] = None,
    ):
        super().__init__()
        """Save hyperparameters"""
        # Can be accessed via checkpoint['hyper_parameters']
        self.save_hyperparameters()

        """Configs"""
        self.log_grads = True

        """model"""
        self.network = network

        # Print out network
        # self.example_input_array = torch.zeros(
        #     64,
        #     int(self.hparams.in_channels),
        #     image_y,
        #     image_x
        #     # 64, int(self.hparams.num_channels), 176, 104
        # )
        self.example_input_array = torch.zeros(
            1,
            int(self.hparams.in_channels),
            64,
            128,
            96
            # 64, int(self.hparams.num_channels), 176, 104
        )

        """Metrics"""
        self.train_metrics = acc_prec_recall(len(self.hparams.classes))
        self.val_metrics = acc_prec_recall(len(self.hparams.classes))
        self.test_metrics = acc_prec_recall(len(self.hparams.classes))
        self.metrics = {
            "train": self.train_metrics,
            "val": self.val_metrics,
            "test": self.test_metrics
        }

    def _generate_logs(self, preds, true, prefix, metrics):
        _, preds_batch = torch.max(preds, 1)
        return {
            # f"{prefix}_loss": loss,
            **{
                f"{prefix}_{key}": metrics[key](
                    preds_batch, true
                )
                for key in metrics.keys()
            },
        }

    def parse_batch(self, batch):
        x = batch[self.hparams.x_label]
        y = batch[self.hparams.y_label]
        # Return floats because this is expected dtype for nn.Loss
        return x.float(), y.long()

    def forward(self, x):
        output = self.network(x)
        return output

    def _log_metrics(self, outputs, prefix):

        logs = self._generate_logs(
            outputs['preds'],
            outputs['target'],
            prefix,
            self.metrics[prefix],
        )
        outputs.update(logs)

        for key, value in logs.items():
            # TODO configure outout logging
            self.log(key, value, on_step=True, on_epoch=True, prog_bar=True)

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
            self.logger.experiment.add_figure(
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

        self.log(
            "train_loss", loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True
        )

        return {
            'loss': loss,
            'preds': yhat,
            'target': y,
            'batch_idx': batch_idx
        }

    def training_step_end(self, outputs):
        outputs = self._log_metrics(outputs, "train")
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
        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True)

        return {
            'val_loss': loss,
            'preds': yhat,
            'target': y,
            'batch_idx': batch_idx
        }

    def validation_step_end(self, outputs):

        outputs = self._log_metrics(outputs, "val")

        return outputs

    def validation_epoch_end(self, outputs):

        avg_loss = torch.stack(
            [x["val_loss"] for x in outputs]).mean()
        avg_acc = torch.stack(
            [x["val_accuracy"] for x in outputs]).mean()
        self.log("val_loss_epoch", avg_loss)
        self.log("val_accuracy_epoch", avg_acc)

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
                logger=self.logger, 
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

        outputs = self._log_metrics(outputs, "test")

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
                logger=self.logger, 
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
                self.logger.experiment.add_histogram(
                    tag=name,
                    values=grads,
                    global_step=self.trainer.global_step
                )


