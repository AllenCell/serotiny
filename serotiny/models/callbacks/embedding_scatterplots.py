import torch
from typing import Optional
from pytorch_lightning import Callback, LightningModule, Trainer
from pathlib import Path
import pandas as pd
from sklearn.decomposition import PCA

from serotiny.utils.viz_utils import make_embedding_pairplots
from serotiny.utils.model_utils import get_ranked_dims


class EmbeddingScatterPlots(Callback):
    """"""

    def __init__(
        self,
        fitted_pca: PCA,
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

        self.cutoff_kld_per_dim = cutoff_kld_per_dim
        if self.cutoff_kld_per_dim is None:
            self.cutoff_kld_per_dim = 0

    def on_test_epoch_end(self, trainer: Trainer, pl_module: LightningModule):

        with torch.no_grad():
            dir_path = Path(trainer.logger[1].save_dir)

            ranked_z_dim_list, mu_std_list, _ = get_ranked_dims(
                dir_path, self.cutoff_kld_per_dim, max_num_shapemodes=8
            )

            subdir = dir_path / "embeddings"
            subdir.mkdir(parents=True, exist_ok=True)

            all_embeddings = pd.read_csv(subdir / "embeddings_all.csv")

            make_embedding_pairplots(
                all_embeddings.loc[all_embeddings.split == "test"],
                self.fitted_pca,
                self.n_components,
                ranked_z_dim_list=ranked_z_dim_list,
                model=pl_module,
                save_dir=subdir,
                cond_size=self.c_dim,
            )
