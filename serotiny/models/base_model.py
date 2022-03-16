from typing import Union, Optional, Sequence
import gc

import logging
import inspect

import numpy as np
import torch
import torch.nn as nn

import pytorch_lightning as pl
from pytorch_lightning.utilities.parsing import get_init_args

from serotiny.models.utils import find_optimizer

Array = Union[torch.Tensor, np.array, Sequence[float]]
logger = logging.getLogger("lightning")
logger.propagate = False


class BaseModel(pl.LightningModule):
    def __init__(
        self,
        x_label: str = "x",
        id_label: str = Optional[None],
        optimizer: torch.optim.Optimizer = torch.optim.Adam,
        cache_outputs: Sequence = ("test",),
        **kwargs
    ):
        super().__init__()

        frame = inspect.currentframe()
        init_args = get_init_args(frame)
        self.save_hyperparameters(
            *[arg for arg, v in init_args.items()
              if (isinstance(v, (nn.Module, torch.optim.Optimizer)) or
                  issubclass(v, (nn.Module, torch.optim.Optimizer)))
             ]
        )

        self.optimizer = optimizer
        self._cached_outputs = dict()


    def parse_batch(self, batch):
        return batch[self.hparams.x_label].float()


    def forward(self, x, **kwargs):
        raise NotImplementedError


    def _step(self, stage, batch, batch_idx, logger):
        raise NotImplementedError

        # Here you should implement the logic for a step in the training/validation/test
        # process. The stage (training/validation/test) is given by the variable `stage`.
        #x = self.parse_batch(batch)

        #if self.hparams.id_label is not None:
        #    if self.hparams.id_label in batch:
        #        ids = batch[self.hparams.id_label].detach().cpu()
        #        results.update({
        #            self.hparams.id_label: ids,
        #            "id": ids
        #        })

        #return results

    def training_step(self, batch, batch_idx):
        return self._step("train", batch, batch_idx, logger=True)

    def validation_step(self, batch, batch_idx):
        return self._step("val", batch, batch_idx, logger=True)

    def test_step(self, batch, batch_idx):
        return self._step("test", batch, batch_idx, logger=False)

    def _epoch_end(self, split, outputs):
        if split in self.hparams.cache_outputs:
            if split in self._cached_outputs:
                del self._cached_outputs[split]
                gc.collect()
            self._cached_outputs[split] = outputs

    def train_epoch_end(self, outputs):
        self._epoch_end("train", outputs)

    def validation_epoch_end(self, outputs):
        self._epoch_end("val", outputs)

    def test_epoch_end(self, outputs):
        self._epoch_end("test", outputs)

    def configure_optimizers(self):
        optimizer = self.optimizer(self.parameters())
        return {
            "optimizer": optimizer,
            "monitor": "val_loss",
            "interval": "epoch",
            "frequency": 1,
            "strict": True,
        }
