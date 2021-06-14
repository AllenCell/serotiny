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

            result.to_csv(path, index=False)

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

            result_with_bins_and_outliers.to_csv(path2, index=False)

            path3 = subdir / "all_dim_bin_counts.csv"

            all_dim_bin_counts.to_csv(path3, index=False)

            plot_bin_count_table(all_dim_bin_counts, subdir)


def find_outliers(
    ranked_z_dim_list: list,
    test_embeddings: pd.DataFrame,
    bins: list,
):
    for dim in ranked_z_dim_list:
        mu_array = np.array(test_embeddings[[f"mu_{dim}"]]).astype(np.float32)
        mu_array -= mu_array.mean()
        mu_array /= mu_array.std()

        binw = 0.5 * np.diff(bins).mean()
        bin_edges = np.unique([(b - binw, b + binw) for b in bins])

        inds = np.digitize(mu_array, bin_edges)
        # Find outliers per dim and add to embeddings
        left_outliers = np.where(inds.flatten() == 0)
        right_outliers = np.where(inds.flatten() == len(bin_edges))
        outlier_col = np.zeros(test_embeddings.shape[0], dtype="object")

        if left_outliers[0].size > 0:
            outlier_col[left_outliers[0]] = "Left Outlier"
            inds[left_outliers[0]] = 0  # Map left outlier to bin 1
        if right_outliers[0].size > 0:
            outlier_col[right_outliers[0]] = "Right Outlier"
            inds[right_outliers[0]] = (
                len(bin_edges) - 1
            )  # Map right outlier to last bin

        outlier_col[np.where(outlier_col == 0)] = False
        test_embeddings[f"outliers_mu_{dim}"] = outlier_col

    return test_embeddings


def get_bins_for_each_cell(
    ranked_z_dim_list: list,
    test_embeddings: pd.DataFrame,
    bins: list,
):

    for dim in ranked_z_dim_list:
        mu_array = np.array(test_embeddings[[f"mu_{dim}"]]).astype(np.float32)
        mu_array -= mu_array.mean()
        mu_array /= mu_array.std()

        binw = 0.5 * np.diff(bins).mean()
        bin_edges = np.unique([(b - binw, b + binw) for b in bins])
        bin_edges[0] = -np.inf
        bin_edges[-1] = np.inf

        inds = np.digitize(mu_array, bin_edges)

        # Add bins per dim to embeddings
        bin_values = []
        for i in inds:
            bin_values.append((bin_edges[i - 1].item(), bin_edges[i].item()))
        test_embeddings[f"bins_mu_{dim}"] = bin_values

    bin_embeddings = test_embeddings[[i for i in test_embeddings.columns if "bin" in i]]

    ranked_z_dim_bin_counts = []
    for dim in ranked_z_dim_list:
        bin_counts = bin_embeddings.pivot_table(
            index=[f"bins_mu_{dim}"], aggfunc="size"
        )
        bin_counts = bin_counts.to_frame(f"bin_count_mu_{dim}")

        ranked_z_dim_bin_counts.append(bin_counts)

    all_dim_bin_counts = pd.concat(ranked_z_dim_bin_counts, axis=1)

    return test_embeddings, all_dim_bin_counts
