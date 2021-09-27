"""
General regression module, implemented as a Pytorch Lightning module
"""

from typing import Union, Dict

import torch.nn as nn
from torch.nn.modules.loss import _Loss as Loss
import torch.nn.functional as F
import numpy as np

import pytorch_lightning as pl

from serotiny.utils import init
from serotiny.models._utils import find_optimizer
from serotiny.networks.weight_init import weight_init
from sklearn.metrics import precision_recall_fscore_support
import seaborn as sns
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt


class GraphNodePredictionModel(pl.LightningModule):
    """
    Pytorch lightning module that implements regression model logic

    Parameters
    -----------
    network:
        Neural network for training a regression problem

    x_label: str
        Key for loading input image
        Example: "projection_image"

    y_label: str
        Key for loading image label

    in_channels: int
        Number of input channels in the image
        Example: 3

    lr: float
        Learning rate for the optimizer. Example: 1e-3

    optimizer: str
        str key to instantiate optimizer. Example: "Adam"

    lr_scheduler: str
        str key to instantiate learning rate scheduler

    """

    def __init__(
        self,
        network: Union[nn.Module, Dict],
        weights: list,
        lr: float,
        optimizer: str,
        task: str,
        loss: Union[Loss, Dict] = nn.MSELoss(),
    ):
        super().__init__()
        # Can be accessed via checkpoint['hyper_parameters']
        # self.save_hyperparameters()
        if isinstance(loss, dict):
            self.loss = init(loss)
        else:
            self.loss = loss

        if isinstance(network, dict):
            network = init(network)

        self.network = network.apply(weight_init)
        self.weights = weights
        self.lr = lr
        self.optimizer = optimizer
        self.task = task
        self._cache = dict()

    def parse_batch(self, batch):
        """
        Retrieve x and y from batch
        """
        x = batch.x
        y = batch.y
        if self.task == "classification":
            y = y.long()
        else:
            y = y.float()
        edge_index = batch.edge_index
        edge_weight = batch.edge_weight
        edge_attr = batch.edge_attr

        return x.float(), edge_index, edge_weight, edge_attr, y

    def forward(self, x):
        return self.network(x)

    def _step(self, yhat, y):

        loss = F.nll_loss(yhat, y, reduction="none")
        pred = yhat.argmax(dim=-1)
        correct = pred.eq(y)

        return loss, pred, correct

    def training_step(self, batch, batch_idx):

        self.network.set_aggr("add")

        x, edge_index, edge_weight, edge_attr, y = self.parse_batch(batch)
        edge_weight = batch.edge_norm * edge_weight
        yhat = self.network(x, edge_index, edge_weight)

        loss, pred, correct = self._step(yhat, y)

        # Node norm requires graph saint random walk
        loss = (loss * batch.node_norm)[batch.train_mask].sum()

        accs = []
        for _, mask in batch("train_mask"):
            accs.append(correct[mask].sum().item() / mask.sum().item())

        self.log("train_loss", loss.detach(), logger=True)

        return {
            "loss": loss,
            "preds": pred,
            "correct": correct,
            "target": y,
            "accuracy": accs,
            "batch_idx": batch_idx,
        }

    def training_epoch_end(self, outputs):
        self._cache["train"] = outputs
        print(
            "train loss",
            np.mean([out["loss"].item() for out in outputs]),
            "accuracy",
            np.mean([out["accuracy"][0] for out in outputs]),
        )

    def validation_step(self, batch, batch_idx):
        self.network.set_aggr("mean")

        x, edge_index, edge_weight, edge_attr, y = self.parse_batch(batch)

        yhat = self.network(x, edge_index)

        loss, pred, correct = self._step(yhat, y)

        # Node norm requires graph saint random walk
        loss = (loss * batch.node_norm)[batch.val_mask].sum()

        accs = []
        for _, mask in batch("train_mask", "val_mask", "test_mask"):
            accs.append(correct[mask].sum().item() / mask.sum().item())

        self.log("val_loss", loss.detach(), logger=True)

        return {
            "val_loss": loss,
            "preds": pred,
            "correct": correct,
            "target": y,
            "accuracy": accs,
            "batch_idx": batch_idx,
        }

    def validation_epoch_end(self, outputs):
        self._cache["val"] = outputs
        print(
            "val loss",
            np.mean([out["val_loss"].item() for out in outputs]),
            "accuracy",
            np.mean([out["accuracy"][0] for out in outputs]),
        )

    def test_step(self, batch, batch_idx):
        self.network.set_aggr("mean")

        x, edge_index, edge_weight, edge_attr, y = self.parse_batch(batch)

        yhat = self.network(x, edge_index)

        loss, pred, correct = self._step(yhat, y)

        # Node norm requires graph saint random walk
        loss = (loss * batch.node_norm)[batch.test_mask].sum()

        accs = []
        for _, mask in batch("train_mask", "val_mask", "test_mask"):
            accs.append(correct[mask].sum().item() / mask.sum().item())

        self.log("val loss", loss.detach(), logger=True)

        return {
            "test_loss": loss,
            "preds": pred,
            "correct": correct,
            "target": y,
            "accuracy": accs,
            "batch_idx": batch_idx,
        }

    def test_step_end(self, outputs):
        return outputs

    def test_epoch_end(self, outputs):
        self._cache["test"] = outputs

        print(
            "Test loss",
            np.mean([out["test_loss"].item() for out in outputs]),
            "accuracy",
            np.mean([out["accuracy"][0] for out in outputs]),
        )

        precs = []
        recs = []
        fscore = []
        all_true, all_pred = [], []
        for pred in outputs:
            this_pred = pred["preds"].detach().cpu().numpy()
            this_true = pred["target"].detach().cpu().numpy()
            all_metrics = precision_recall_fscore_support(
                this_true, this_pred, average="weighted"
            )
            precs.append(all_metrics[0])
            recs.append(all_metrics[1])
            fscore.append(all_metrics[2])
            all_true.append(this_true)
            all_pred.append(this_pred)

        fig, ax = plt.subplots(1, 1, figsize=(7, 7))
        ax = sns.heatmap(
            confusion_matrix(this_true, this_pred),
            cmap="RdBu_r",
            annot=True,
            annot_kws={"size": 7},
        )
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        fig.savefig("confusion.png")

    def configure_optimizers(self):
        optimizer_class = find_optimizer(self.optimizer)
        optimizer = optimizer_class(self.parameters(), lr=self.lr)

        return (
            {
                "optimizer": optimizer,
                "monitor": "val_loss",
                "interval": "epoch",
                "frequency": 1,
                "strict": True,
            },
        )
