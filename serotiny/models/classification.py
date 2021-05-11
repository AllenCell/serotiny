"""
General classifier module, implemented as a Pytorch Lightning module
"""

from typing import Optional, Union
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

import torch
from torch import nn
from torch.nn import functional as F
import torch.optim as opt

import pytorch_lightning as pl

from serotiny.models._utils import (
    add_pr_curve_tensorboard,
    find_optimizer,
    find_lr_scheduler,
)


class ClassificationModel(pl.LightningModule):
    """
    Pytorch lightning module that implements classification model logic

    Parameters
    -----------
    network:
        Instantiated neural network for training a classification problem

    x_label: str
        Key for loading input image
        Example: "projection_image"

    y_label: str
        Key for loading image label
        Example: "mitotic_class"

    in_channels: int
        Number of input channels in the image
        Example: 3

    classes: tuple
        all available classes in the given class column
        Example: ("M0", "M1/M2", "M3", "M4/M5", "M6/M7")

    lr: float
        Learning rate for the optimizer. Example: 1e-3

    optimizer: str
        str key to instantiate optimizer. Example: "Adam"

    lr_scheduler: str
        str key to instantiate learning rate scheduler

    dimensions: Optional[tuple, list]:
        Dimensions of each image channel. Only used for printing
        out the network/model to the command line. Set to None
        if this is not wanted
    """

    def __init__(
        self,
        network,
        x_label: str,
        y_label: str,
        in_channels: int,
        classes: tuple,
        lr: float,
        optimizer: str,
        lr_scheduler: str,
        dimensions: Optional[Union[tuple, list]] = None,
    ):
        super().__init__()
        # Can be accessed via checkpoint['hyper_parameters']
        self.save_hyperparameters()

        self.log_grads = True

        self.network = network

        # Print out network
        if dimensions is not None:
            self.example_input_array = torch.zeros(
                1, int(self.hparams.in_channels), *dimensions
            )

        self.train_metrics = acc_prec_recall(len(self.hparams.classes))
        self.val_metrics = acc_prec_recall(len(self.hparams.classes))
        self.test_metrics = acc_prec_recall(len(self.hparams.classes))
        self.metrics = {
            "train": self.train_metrics,
            "val": self.val_metrics,
            "test": self.test_metrics,
        }

    def _generate_logs(self, preds, true, prefix, metrics):
        """
        Produce logs based on metrics dict

        Parameters
        ----------
        preds: torch.tensor
            the predictions given by the model
        true: torch.tensor
            the true labels
        prefix: str
            prefix to identify these logs
        metrics: dict
            metric dictionary with keys and callables
        """
        _, preds_batch = torch.max(preds, 1)
        return {
            # f"{prefix}_loss": loss,
            **{
                f"{prefix}_{key}": metrics[key](preds_batch, true)
                for key in metrics.keys()
            },
        }

    def parse_batch(self, batch):
        """
        Retrieve x and y from batch
        """
        x = batch[self.hparams.x_label]
        y = batch[self.hparams.y_label]
        # Return floats because this is expected dtype for nn.Loss
        return x.float(), y.long().squeeze()

    def forward(self, x):
        output = self.network(x)
        return output

    def _log_metrics(self, outputs, prefix, CSV_logger=False):
        """
        Produce metrics and save them
        """

        logs = self._generate_logs(
            outputs["preds"],
            outputs["target"],
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
            target = np.expand_dims(outputs["target"].cpu().numpy(), 1)

            df = pd.DataFrame(
                np.hstack([probs, pred, target]),
                columns=[f"prob_{i}" for i in range(len(self.hparams.classes))]
                + ["pred", "target"],
            )

            path = Path(self.logger[1].save_dir) / "test_results.csv"
            if path.exists():
                df.to_csv(path, mode="a", header=False, index=False)
            else:
                df.to_csv(path, header="column_names", index=False)

        if outputs["batch_idx"] == 0:
            _, preds_batch = torch.max(outputs["preds"], 1)
            conf_mat = pl.metrics.functional.confusion_matrix(
                preds_batch, outputs["target"], num_classes=len(self.hparams.classes)
            )

            fig, ax = plt.subplots(1, 1, figsize=(10, 10))
            sns.heatmap(
                conf_mat.cpu(), cmap="RdBu_r", annot=True, ax=ax, annot_kws={"size": 8}
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

        return {"loss": loss, "preds": yhat, "target": y, "batch_idx": batch_idx}

    def training_step_end(self, outputs):

        # No CSV metric logging needed here
        outputs = self._log_metrics(outputs, "train", CSV_logger=False)
        return outputs

    def validation_step(self, batch, batch_idx):

        x, y = self.parse_batch(batch)
        yhat = self(x)
        loss = nn.CrossEntropyLoss()(yhat, y)
        # Default logger for validation step is True
        self.log("val_loss", loss)

        return {"val_loss": loss, "preds": yhat, "target": y, "batch_idx": batch_idx}

    def validation_step_end(self, outputs):

        # No CSV metric logging needed here
        outputs = self._log_metrics(outputs, "val", CSV_logger=False)

        return outputs

    def validation_epoch_end(self, outputs):

        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        avg_acc = torch.stack([x["val_accuracy"] for x in outputs]).mean()

        val_epoch_end_metrics = {
            "val_loss_epoch": avg_loss,
            "val_accuracy_epoch": avg_acc,
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
                global_step=self.current_epoch,
            )

        return {"val_loss": avg_loss}

    def test_step(self, batch, batch_idx):

        x, y = self.parse_batch(batch)

        yhat = self(x)
        loss = nn.CrossEntropyLoss()(yhat, y)

        self.log("test_loss", loss)

        return {"test_loss": loss, "preds": yhat, "target": y, "batch_idx": batch_idx}

    def test_step_end(self, outputs):

        # Log test metrics to CSV
        outputs = self._log_metrics(outputs, "test", CSV_logger=True)

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
        optimizer_class = find_optimizer(self.hparams.optimizer)
        optimizer = optimizer_class(self.parameters(), lr=self.hparams.lr)

        scheduler_class = find_lr_scheduler(self.hparams.lr_scheduler)
        scheduler = scheduler_class(optimizer)

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

def acc_prec_recall(n_classes):
    """
    util function to instantiate a ModuleDict for metrics
    """
    return nn.ModuleDict(
        {
            "accuracy": pl.metrics.Accuracy(),
            "precision": pl.metrics.Precision(num_classes=n_classes, average="macro"),
            "recall": pl.metrics.Recall(num_classes=n_classes, average="macro"),
        }
    )
