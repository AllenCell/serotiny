import torch
from pytorch_lightning import Callback, LightningModule, Trainer
from pathlib import Path
from typing import Optional
from serotiny.utils.model_utils import (
    get_all_embeddings,
    get_ranked_dims,
    get_bins_for_each_cell,
    find_outliers,
)
from serotiny.utils.viz_utils import plot_bin_count_table


class GetEmbeddings(Callback):
    """"""

    def __init__(
        self,
        x_label: str,
        c_label: str,
        id_fields: list,
        latent_walk_range: Optional[list] = None,
    ):
        """
        Args:
            resample_n: How many times to sample from latent space and average

            x_label: x_label from datamodule

            c_label: c_label from datamodule

            id_fields: id_fields from datamodule
        """
        super().__init__()

        self.x_label = x_label
        self.c_label = c_label
        self.id_fields = id_fields
        self.latent_walk_range = latent_walk_range
        self.cutoff_kld_per_dim = 0
        if self.latent_walk_range is None:
            # self.latent_walk_range = [-2, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0]
            self.latent_walk_range = [
                -2,
                -1,
                -0.5,
                -0.25,
                -0.1,
                0,
                0.1,
                0.25,
                0.5,
                1,
                2,
            ]

    def on_test_epoch_end(self, trainer: Trainer, pl_module: LightningModule):

        with torch.no_grad():
            dir_path = Path(trainer.logger[1].save_dir)

            subdir = dir_path / "embeddings"
            subdir.mkdir(parents=True, exist_ok=True)

            result = get_all_embeddings(
                trainer.train_dataloader,
                trainer.val_dataloaders[0],
                trainer.test_dataloaders[0],
                pl_module,
                self.x_label,
                self.c_label,
                self.id_fields,
            )

            path = subdir / "embeddings_all.csv"

            if path.exists():
                result.to_csv(path, mode="a", header=False, index=False)
            else:
                result.to_csv(path, header="column_names", index=False)

            ranked_z_dim_list, mu_std_list, _ = get_ranked_dims(
                dir_path, self.cutoff_kld_per_dim, max_num_shapemodes=8
            )

            result_with_bins, all_dim_bin_counts = get_bins_for_each_cell(
                ranked_z_dim_list,
                result.loc[result.split == "test"],
                self.latent_walk_range,
            )
            result_with_bins_and_outliers = find_outliers(
                ranked_z_dim_list,
                result_with_bins,
                self.latent_walk_range,
            )

            path2 = subdir / "embeddings_test_with_bins_outliers.csv"

            result_with_bins_and_outliers.to_csv(
                path2, header="column_names", index=False
            )

            path3 = subdir / "all_dim_bin_counts.csv"

            all_dim_bin_counts.to_csv(path3, header="column_names", index=False)

            plot_bin_count_table(all_dim_bin_counts, subdir)
