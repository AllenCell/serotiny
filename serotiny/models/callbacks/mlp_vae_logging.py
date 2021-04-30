from typing import Optional

import torch
from pytorch_lightning import Callback, LightningModule, Trainer
from pathlib import Path
import pandas as pd
from serotiny.utils.viz_utils import make_plot_encoding
from serotiny.utils.model_utils import to_device


class MLPVAELogging(Callback):  # pragma: no cover
    """"""

    def __init__(
        self,
        datamodule,
        resample_n: int = 10,
        values: list = [-1, 0, 1],
        conds_list: Optional[list] = None,
    ):
        """
        Args:
            resample_n: How many times to sample from
            the latent space before averaging results
            values: What value to pass in as a condition to the decoder
            conds_list: Which columns in the condition to set to a value
            save_dir: Where to save plots
            Default: csv_logs folder
        """
        super().__init__()

        self.resample_n = resample_n
        self.datamodule = datamodule
        self.conds_list = conds_list
        self.values = values
        if self.datamodule.__module__ == "serotiny.datamodules.gaussian":
            self.values = [0]

    def on_test_epoch_end(self, trainer: Trainer, pl_module: LightningModule):

        with torch.no_grad():

            dir_path = Path(trainer.logger[1].save_dir)
            subdir = dir_path / "encoding_test"
            subdir.mkdir(parents=True, exist_ok=True)

            test_dataloader = self.datamodule.test_dataloader()
            test_iter = next(iter(test_dataloader))
            x_label, c_label = self.datamodule.x_label, self.datamodule.c_label

            x = test_iter[x_label].float()
            c = test_iter[c_label].float()

            x, c = to_device(x, c, pl_module.device)

            stats = pd.read_csv(dir_path / "stats_all.csv")

            enc_layers = pl_module.encoder.enc_layers
            dec_layers = pl_module.decoder.dec_layers

            if not self.conds_list:
                if self.datamodule.__module__ == "serotiny.datamodules.gaussian":
                    # Example for a 2D Gaussian
                    # conds_list = [[], [0], [0, 1]]
                    conds_list = []
                    for i in range(x.shape[-1] + 1):
                        conds_list.append([j for j in range(i)])
                    conds_list = [conds_list[-1]]

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
                make_plot_encoding(
                    subdir,
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
