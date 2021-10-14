import torch
import torch.nn.functional as F

import matplotlib.pyplot as plt

from pytorch_lightning import Callback
from pytorch_lightning.loggers import TensorBoardLogger


class PlotXHat(Callback):
    def __init__(
        self,
        cell_id=None,
        cell_id_label="cell_id",
        logits=False,
        mode="3d",
    ):
        self.cell_id = cell_id
        self.cell_id_label = cell_id_label
        self.logits = logits
        self.mode = mode

    def on_validation_batch_end(
        self, trainer, model, outputs, batch, batch_idx, dataloader_idx
    ):
        if trainer.running_sanity_check:
            return

        if self.cell_id is None:
            self.cell_id = batch[self.cell_id_label][0]
        else:
            if self.cell_id not in batch[self.cell_id_label]:
                return

        idx = batch[self.cell_id_label].index(self.cell_id)

        logger = None
        for _logger in trainer.logger:
            if isinstance(_logger, TensorBoardLogger):
                logger = _logger
                break

        if logger is None:
            return

        x = model.parse_batch(batch)
        if isinstance(x, tuple):
            x, forward_kwargs = x
        else:
            forward_kwargs = dict()

        with torch.no_grad():
            xhat = model.forward(x, **forward_kwargs)[0]

            if self.logits:
                xhat = F.sigmoid(xhat)

            fig, axs = plt.subplots(1, 2, figsize=(10, 5))

            if self.mode == "3d":
                axs[0].imshow(x[idx].cpu().numpy().squeeze().mean(axis=0))
                axs[1].imshow(xhat[idx].cpu().numpy().squeeze().mean(axis=0))
            else:
                axs[0].imshow(x[idx].cpu().numpy().squeeze())
                axs[1].imshow(xhat[idx].cpu().numpy().squeeze())

            logger.experiment.add_figure(
                "x and xhat", fig, global_step=trainer.current_epoch
            )
