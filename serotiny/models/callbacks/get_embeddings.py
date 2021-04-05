from typing import Optional, Sequence, Tuple, Union

import torch
from torch import device, Tensor
from pytorch_lightning import Callback, LightningModule, Trainer
from pathlib import Path
import pandas as pd
import numpy as np


class GetEmbeddings(Callback):
    """"""

    def __init__(
        self,
        resample_n: int,
        x_label: str,
        c_label: str,
        id_fields: list,
    ):
        """
        Args:
            config: Config file used by Matheus in cvapipe_analysis

            save_dir: Where to save plots
            Default: csv_logs folder

            condition: A condition to give the decoder
            Default None

            latent_walk_range: What range to do the latent walk
            Default = [-2, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0]

            cutoff_kld_per_dim: Cutoff to use for KLD per dim
            Default = 0.5
        """
        super().__init__()

        self.resample_n = resample_n
        self.x_label = x_label
        self.c_label = c_label
        self.id_fields = id_fields

    def to_device(
        self, batch_x: Sequence, batch_y: Sequence, device: Union[str, device]
    ) -> Tuple[Tensor, Tensor]:

        # last input is for online eval
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        return batch_x, batch_y

    def compute_embeddings(self, pl_module: LightningModule, input_x, cond_c):

        # Make empty list
        my_recon_list, my_z_means_list, my_log_var_list = [], [], []
        input_x, cond_c = self.to_device(input_x, cond_c, pl_module.device)
        # Run resample_n times for resampling
        for resample in range(self.resample_n):
            recon_batch, z_means, log_var, _, _, _, _, _ = pl_module(
                input_x.clone().float(), cond_c.clone().float()
            )
            my_recon_list.append(recon_batch)
            my_z_means_list.append(z_means)
            my_log_var_list.append(log_var)

        # Average over the N resamples
        recon_batch = torch.mean(torch.stack(my_recon_list), dim=0)
        z_means = torch.mean(torch.stack(my_z_means_list), dim=0)
        log_var = torch.mean(torch.stack(my_log_var_list), dim=0)

        return recon_batch, z_means, log_var

    def get_all_embeddings(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
    ):

        all_z_means = []
        cell_ids = []
        split = []
        for step, x in enumerate(trainer.train_dataloader):
            input_x = x[self.x_label]
            cond_c = x[self.c_label]
            cell_id = x["id"][self.id_fields[0]]

            recon_batch, z_means, log_var = self.compute_embeddings(
                pl_module, input_x, cond_c
            )
            all_z_means.append(z_means)
            cell_ids.append(cell_id)
            split.append(["train"] * z_means.shape[0])

        for step, x in enumerate(trainer.val_dataloaders[0]):
            input_x = x[self.x_label]
            cond_c = x[self.c_label]
            cell_id = x["id"][self.id_fields[0]]

            recon_batch, z_means, log_var = self.compute_embeddings(
                pl_module, input_x, cond_c
            )
            all_z_means.append(z_means)
            cell_ids.append(cell_id)
            split.append(["val"] * z_means.shape[0])

        for step, x in enumerate(trainer.test_dataloaders[0]):
            input_x = x[self.x_label]
            cond_c = x[self.c_label]
            cell_id = x["id"][self.id_fields[0]]

            recon_batch, z_means, log_var = self.compute_embeddings(
                pl_module, input_x, cond_c
            )
            all_z_means.append(z_means)
            cell_ids.append(cell_id)
            split.append(["test"] * z_means.shape[0])

        all_z_means = torch.cat(all_z_means, dim=0)
        cell_ids = torch.cat(cell_ids, dim=0)
        split = [item for sublist in split for item in sublist]
        all_z_means = all_z_means.detach().cpu().numpy()

        df1 = pd.DataFrame(
            all_z_means, columns=[f"mu_{i}" for i in range(all_z_means.shape[1])]
        )
        df2 = pd.DataFrame(cell_ids, columns=["CellId"])
        df3 = pd.DataFrame(split, columns=["split"])
        frames = [df1, df2, df3]
        result = pd.concat(frames, axis=1)

        return result

    def on_test_epoch_end(self, trainer: Trainer, pl_module: LightningModule):

        with torch.no_grad():
            dir_path = Path(trainer.logger[1].save_dir)

            result = self.get_all_embeddings(trainer, pl_module)

            path = dir_path / f"embeddings_all.csv"

            if path.exists():
                result.to_csv(path, mode="a", header=False, index=False)
            else:
                result.to_csv(path, header="column_names", index=False)
