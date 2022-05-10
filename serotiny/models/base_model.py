import gc
import inspect
import logging
from typing import Sequence, Union
from omegaconf import ListConfig, DictConfig, OmegaConf

import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.utilities.parsing import get_init_args

Array = Union[torch.Tensor, np.array, Sequence[float]]
logger = logging.getLogger("lightning")
logger.propagate = False


def _is_primitive(value):
    if isinstance(value, (type(None), bool, str, int, float)):
        return True

    if isinstance(value, (tuple, list)):
        return all(_is_primitive(el) for el in value)

    if isinstance(value, dict):
        return all(_is_primitive(el) for el in value.values())

    return False


def _cast_omegaconf(value):
    if isinstance(value, (ListConfig, DictConfig)):
        return OmegaConf.to_container(value, resolve=True)
    return value


def _parse_init_args(frame):
    init_args = get_init_args(frame)
    if frame.f_back.f_code.co_name == "__init__":
        # if this was called from a subclass's init
        init_args.update(get_init_args(frame.f_back))

    init_args = {k: _cast_omegaconf(v) for k, v in init_args.items()}
    ignore = [arg for arg, v in init_args.items() if not _is_primitive(v)]
    for arg in ignore:
        del init_args[arg]
    return init_args


class BaseModel(pl.LightningModule):
    def __init__(self, **kwargs):
        super().__init__()

        frame = inspect.currentframe()
        init_args = _parse_init_args(frame)

        self.save_hyperparameters(init_args, logger=False)
        self.optimizer = init_args.get("optimizer", torch.optim.Adam)
        self.cache_outputs = init_args.get("cache_outputs", ("test",))
        self._cached_outputs = dict()

    def parse_batch(self, batch):
        raise NotImplementedError

    def forward(self, x, **kwargs):
        raise NotImplementedError

    def _step(self, stage, batch, batch_idx, logger):
        raise NotImplementedError

        # Here you should implement the logic for a step in the
        # training/validation/test process.
        # The stage (training/validation/test) is given by the variable `stage`.
        #
        # Example:
        #
        # x = self.parse_batch(batch)

        # if self.hparams.id_label is not None:
        #    if self.hparams.id_label in batch:
        #        ids = batch[self.hparams.id_label].detach().cpu()
        #        results.update({
        #            self.hparams.id_label: ids,
        #            "id": ids
        #        })

        # return results

    def training_step(self, batch, batch_idx):
        return self._step("train", batch, batch_idx, logger=True)

    def validation_step(self, batch, batch_idx):
        return self._step("val", batch, batch_idx, logger=True)

    def test_step(self, batch, batch_idx):
        return self._step("test", batch, batch_idx, logger=False)

    def predict_step(self, batch, batch_idx):
        return self._step("predict", batch, batch_idx, logger=False)

    def _epoch_end(self, split, outputs):
        if split in self.cache_outputs:
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
            "frequency": 1,
        }
