from typing import Optional, Sequence, Tuple, Union

import torch
from torch import device, Tensor
from pytorch_lightning import Callback, LightningModule, Trainer
from pathlib import Path
import pandas as pd
from ...metrics.plotting_utils import make_plot_encoding


class MLPVAELogging(Callback):  # pragma: no cover
    """"""

    def __init__(
        self,
        datamodule,
        resample_n: int = 10,
        values: list = [-1, 0, 1],
        conds_list: Optional[list] = None,
        save_dir: Optional[str] = None,
    ):
        """
        Args:
            resample_n: How many times to sample from
            the latent space before averaging results
            save_dir: Where to save plots
            Default: csv_logs folder
        """
        super().__init__()

        self.save_dir = save_dir
        self.resample_n = resample_n
        self.datamodule = datamodule
        self.conds_list = conds_list
        self.values = values
        if self.datamodule.__module__ == "serotiny.datamodules.gaussian":
            self.values = [0]

    def to_device(
        self, batch_x: Sequence, batch_y: Sequence, device: Union[str, device]
    ) -> Tuple[Tensor, Tensor]:

        # last input is for online eval
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        return batch_x, batch_y

    def on_test_epoch_end(self, trainer: Trainer, pl_module: LightningModule):

        with torch.no_grad():
            test_dataloader = self.datamodule.test_dataloader()
            test_iter = next(iter(test_dataloader))
            x_label, c_label = self.datamodule.x_label, self.datamodule.c_label

            x = test_iter[x_label].float()
            c = test_iter[c_label].float()

            x, c = self.to_device(x, c, pl_module.device)

            dir_path = Path(trainer.logger[1].save_dir)
            stats = pd.read_csv(dir_path / "stats_all.csv")

            enc_layers = pl_module.encoder.enc_layers
            dec_layers = pl_module.decoder.dec_layers

            if not self.save_dir:
                self.save_dir = dir_path

            if not self.conds_list:
                if self.datamodule.__module__ == "serotiny.datamodules.gaussian":
                    # Example for a 2D Gaussian
                    # conds_list = [[], [0], [0, 1]]
                    conds_list = []
                    for i in range(x.shape[-1] + 1):
                        conds_list.append([j for j in range(i)])

                elif (
                    self.datamodule.__module__
                    == "serotiny.datamodules.variance_spharm_coeffs"
                ):
                    # For 2 structure integer conditions this is
                    # say [0, 1]
                    num_classes = c.shape[1]
                    conds_list = []
                    for i in range(num_classes):
                        conds_list.append(i)
                    conds_list = [conds_list]

            for value in self.values:
                print(self.datamodule.__module__)
                print(conds_list)
                make_plot_encoding(
                    self.save_dir,
                    pl_module,
                    dec_layers,
                    enc_layers,
                    stats,
                    x,
                    c,
                    conds_list=conds_list,
                    datamodule=self.datamodule,
                    value=value,
                    beta=pl_module.beta,
                    resample_n=self.resample_n,
                    this_dataloader_color=None,
                    save=True,
                    mask=True,
                )
