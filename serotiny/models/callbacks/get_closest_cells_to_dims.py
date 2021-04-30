from typing import Optional

import torch
from pytorch_lightning import Callback, LightningModule, Trainer
from pathlib import Path
import pandas as pd
import matplotlib
from serotiny.utils.model_utils import get_closest_cells
from serotiny.utils.viz_utils import decode_latent_walk_closest_cells
from serotiny.utils.model_utils import get_ranked_dims

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
            self.cutoff_kld_per_dim = 0

    def on_test_epoch_end(self, trainer: Trainer, pl_module: LightningModule):

        with torch.no_grad():
            dir_path = Path(trainer.logger[1].save_dir)

            subdir = dir_path / "closest_cells"
            subdir.mkdir(parents=True, exist_ok=True)

            ranked_z_dim_list, mu_std_list, _ = get_ranked_dims(
                dir_path, self.cutoff_kld_per_dim, max_num_shapemodes=8
            )

            all_embeddings = pd.read_csv(dir_path / "embeddings/embeddings_all.csv")

            result = get_closest_cells(
                ranked_z_dim_list,
                mu_std_list,
                all_embeddings,
                subdir,
                self.path_in_stdv,
                self.metric,
                self.id_col,
                self.N_cells,
            )

            decode_latent_walk_closest_cells(
                trainer,
                pl_module,
                result,
                all_embeddings,
                ranked_z_dim_list,
                mu_std_list,
                subdir,
                self.path_in_stdv,
                self.c_shape,
                self.dna_spharm_cols,
            )
