from typing import Optional, Sequence, Tuple, Union

import torch
from torch import device, Tensor
from pytorch_lightning import Callback, LightningModule, Trainer
from pathlib import Path
import pandas as pd
import numpy as np
from aicsshparam import shtools
from cvapipe_analysis.steps.pca_path_cells.utils import scan_pc_for_cells
from serotiny.utils.metric_utils import get_mesh_from_series
from aicscytoparam import cytoparam
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rc("xtick", labelsize=3)
matplotlib.rc("ytick", labelsize=3)
matplotlib.rcParams["xtick.major.size"] = 0.1
matplotlib.rcParams["xtick.major.width"] = 0.1
matplotlib.rcParams["xtick.minor.size"] = 0.1
matplotlib.rcParams["xtick.minor.width"] = 0.1
matplotlib.rcParams["ytick.major.size"] = 0.1
matplotlib.rcParams["ytick.major.width"] = 0.1
matplotlib.rcParams["ytick.minor.size"] = 0.1
matplotlib.rcParams["ytick.minor.width"] = 0.1


class GetClosestCellsToDims(Callback):
    """"""

    def __init__(
        self,
        path_in_stdv,
        spharm_coeffs_cols: list,
        metric: str,
        id_col: str,
        N_cells: int,
        c_shape: int,
        cutoff_kld_per_dim: Optional[float] = None,
    ):
        """
        Args:

            path_in_stdv: Where to save plots
            Example: np.array([-2.0, -1.5, -1.0, -0.5,  0.0,  0.5,  1.0,  1.5,  2.0])

            metric: A condition to give the decoder
            Example: "Euclidean"

            id_col: "CellId"

            N_cells: 3

            c_shape: shape of condition from datamodule
        """
        super().__init__()

        self.path_in_stdv = path_in_stdv
        self.metric = metric
        self.id_col = id_col
        self.N_cells = N_cells
        self.c_shape = c_shape
        self.dna_spharm_cols = spharm_coeffs_cols
        self.cutoff_kld_per_dim = cutoff_kld_per_dim
        if self.cutoff_kld_per_dim is None:
            self.cutoff_kld_per_dim = 0.5

    def to_device(
        self, batch_x: Sequence, batch_y: Sequence, device: Union[str, device]
    ) -> Tuple[Tensor, Tensor]:

        # last input is for online eval
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        return batch_x, batch_y

    def get_closest_cells(
        self,
        ranked_z_dim_list,
        mu_std_list,
        all_embeddings,
        dir_path,
    ):
        embeddings_most_important_dims = all_embeddings[
            [f"mu_{i}" for i in ranked_z_dim_list]
        ]

        dist_cols = embeddings_most_important_dims.columns

        df_list = []
        dims = []
        for index, dim in enumerate(ranked_z_dim_list):
            mu_std = mu_std_list[index]
            df_cells = scan_pc_for_cells(
                all_embeddings,
                pc=index + 1,  # This function assumes first index is 1
                path=np.array(self.path_in_stdv) * mu_std,
                dist_cols=dist_cols,
                metric=self.metric,
                id_col=self.id_col,
                N_cells=self.N_cells,
            )
            dims.append([dim] * df_cells.shape[0])
            df_list.append(df_cells)
        tmp = pd.concat(df_list)
        tmp = tmp.reset_index(drop=True)
        dims = [item for sublist in dims for item in sublist]
        df2 = pd.DataFrame(dims, columns=["ranked_dim"])
        result = pd.concat([tmp, df2], axis=1)

        path = dir_path / f"closest_real_cells_to_top_dims.csv"

        if path.exists():
            result.to_csv(path, mode="a", header=False, index=False)
        else:
            result.to_csv(path, header="column_names", index=False)

        return result

    def decode_latent_walk(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        closest_cells_df: pd.DataFrame,
        ranked_z_dim_list: list,
        mu_std_list: list,
        dir_path,
    ):
        batch_size = trainer.test_dataloaders[0].batch_size
        latent_dims = pl_module.encoder.enc_layers[-1]

        for index, z_dim in enumerate(ranked_z_dim_list):
            # Set subplots
            fig, ax_array = plt.subplots(
                3,
                len(self.path_in_stdv),
                squeeze=False,
                figsize=(15, 7),
            )
            subset_df = closest_cells_df.loc[closest_cells_df["ranked_dim"] == z_dim]
            mu_std = mu_std_list[index]

            for loc_index, location in enumerate(subset_df["location"].unique()):
                subset_sub_df = subset_df.loc[subset_df["location"] == location]
                subset_sub_df = subset_sub_df.reset_index(drop=True)

                z_inf = torch.zeros(batch_size, latent_dims)
                walk_cols = [i for i in subset_sub_df.columns if "mu" in i]

                this_cell_id = subset_sub_df.iloc[0]["CellId"]

                for walk in walk_cols:
                    # walk_cols is mu_{dim}, so walk[3:]
                    z_inf[:, int(walk[3:])] = torch.from_numpy(
                        np.array(subset_sub_df.iloc[0][walk])
                    )
                z_inf = z_inf.cuda(device=0)
                z_inf = z_inf.float()
                decoder = pl_module.decoder

                y_test = torch.zeros(batch_size, self.c_shape)

                z_inf, y_test = self.to_device(z_inf, y_test, pl_module.device)
                x_hat = decoder(z_inf, y_test)

                test_spharm = x_hat[0, :]
                test_spharm = test_spharm.detach().cpu().numpy()
                test_spharm = pd.DataFrame(test_spharm).T
                test_spharm.columns = self.dna_spharm_cols
                test_spharm_series = test_spharm.iloc[0]

                mesh = get_mesh_from_series(test_spharm_series, "dna", 32)
                img, origin = cytoparam.voxelize_meshes([mesh])

                for proj in [0, 1, 2]:
                    ax_array[proj, loc_index].set_title(
                        f"{self.path_in_stdv[loc_index]}" r"$\sigma$"
                    )
                    ax_array[proj, loc_index].imshow(img.max(proj), cmap="gray")

            [ax.axis("off") for ax in ax_array.flatten()]
            # Save figure
            ax_array.flatten()[0].get_figure().savefig(
                dir_path / f"dim_{z_dim}_rank_{index}_closest_real_cell.png"
            )
            # Close figure, otherwise clogs memory
            plt.close(fig)

    def on_test_epoch_end(self, trainer: Trainer, pl_module: LightningModule):

        with torch.no_grad():
            dir_path = Path(trainer.logger[1].save_dir)

            stats = pd.read_csv(dir_path / "stats_per_dim_test.csv")

            stats = (
                stats.loc[stats["test_kld_per_dim"] > self.cutoff_kld_per_dim]
                .sort_values(by=["test_kld_per_dim"])
                .reset_index(drop=True)
            )

            ranked_z_dim_list = [i for i in stats["dimension"][::-1]]
            mu_std_list = [i for i in stats["mu_std_per_dim"][::-1]]

            num_shapemodes = 8
            if len(ranked_z_dim_list) > num_shapemodes:
                ranked_z_dim_list = ranked_z_dim_list[:num_shapemodes]
                mu_std_list = mu_std_list[:num_shapemodes]

            all_embeddings = pd.read_csv(dir_path / "embeddings_all.csv")

            result = self.get_closest_cells(
                ranked_z_dim_list, mu_std_list, all_embeddings, dir_path
            )

            self.decode_latent_walk(
                trainer, pl_module, result, ranked_z_dim_list, mu_std_list, dir_path
            )
