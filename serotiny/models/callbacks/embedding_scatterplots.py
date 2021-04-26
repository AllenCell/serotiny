import torch
from typing import Optional, List
from pytorch_lightning import Callback, LightningModule, Trainer
from pathlib import Path
import pandas as pd
from sklearn.decomposition import PCA

from serotiny.utils.viz_utils import make_embedding_pairplots, make_pca_pairplots

class EmbeddingScatterPlots(Callback):
    """"""

    def __init__(
        self,
        fitted_pca: PCA,
        pca_df: pd.DataFrame,
        n_components: int,
        c_dim: int,
        cutoff_kld_per_dim: Optional[float] = None,
    ):
        """
        Args:

        """
        super().__init__()

        self.fitted_pca = fitted_pca
        self.n_components = n_components
        self.c_dim = c_dim
        self.pca_df = pca_df

        self.cutoff_kld_per_dim = cutoff_kld_per_dim
        if self.cutoff_kld_per_dim is None:
            self.cutoff_kld_per_dim = 0

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

            make_embedding_pairplots(
                all_embeddings.loc[all_embeddings.split == "test"],
                self.fitted_pca,
                self.n_components,
                ranked_z_dim_list=ranked_z_dim_list,
                model=pl_module,
                save_dir=dir_path,
                cond_size=self.c_dim
            )

            make_pca_pairplots(
                all_embeddings.loc[all_embeddings.split == "test"],
                self.fitted_pca,
                self.pca_df,
                self.n_components,
                ranked_z_dim_list=ranked_z_dim_list,
                model=pl_module,
                save_dir=dir_path,
                cond_size=self.c_dim
            )
