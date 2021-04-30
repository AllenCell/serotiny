from pathlib import Path

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from pytorch_lightning import Callback, LightningModule, Trainer

from serotiny.utils.model_utils import get_ranked_dims

class PCALatentCorrelation(Callback):
    def __init__(self, pca_df, n_pcs=60):
        self.pca_df = pca_df.set_index("CellId").sort_index()
        self.n_pcs = n_pcs


    def on_test_epoch_end(self, trainer: Trainer, pl_module: LightningModule):
        dir_path = Path(trainer.logger[1].save_dir)

        ranked_z_dim_list, mu_std_list, _ = get_ranked_dims(
            dir_path, 0, max_num_shapemodes=np.inf
        )


        if (dir_path / "correlations_pc_embeddings.csv").exists():
            corr = pd.read_csv(dir_path / "correlations_pc_embeddings.csv")
            if "Unnamed: 0" in corr:
                del corr["Unnamed: 0"]
        else:
            df = pd.concat([
                pd.read_csv(dir_path / "embeddings/embeddings_all.csv")
                .set_index("CellId")
                .sort_index(),
                self.pca_df
            ], axis=1)

            corr = df.corr()
            corr = corr[[f"mu_{col}" for col in ranked_z_dim_list]]
            corr = corr.loc[self.pca_df.columns[:self.n_pcs]]

            corr.to_csv(dir_path / "correlations_pc_embeddings.csv")

        f, (ax, cbar_ax) = plt.subplots(1, 2, figsize=(10, 10),
                                        gridspec_kw=dict(width_ratios=(0.9, 0.05)))
        sns.set_context("talk")
        g = sns.heatmap(corr.T, cmap="vlag", square=True, ax=ax, cbar_ax=cbar_ax,
                        xticklabels=True, yticklabels=True, cbar_kws=dict(shrink=0.05),
                        vmin=-1, vmax=1)
        ax.set_xticklabels(ax.get_xmajorticklabels(), fontsize = 13)
        ax.set_yticklabels(ax.get_ymajorticklabels(), fontsize = 13)
        f.savefig(dir_path / "correlation_embeddings_PC.png")
