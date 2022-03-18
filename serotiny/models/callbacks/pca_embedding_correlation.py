from pathlib import Path

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from pytorch_lightning import Callback, LightningModule, Trainer

from serotiny.utils.model_utils import get_ranked_dims

class PCALatentCorrelation(Callback):
    def __init__(self, pca_df, n_pcs=8, spearman=False):
        self.pca_df = pca_df.set_index("CellId").sort_index()
        self.n_pcs = n_pcs
        self.spearman = spearman


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
            print('hello')
            corr = df.corr()
            corr = corr[[f"mu_{col}" for col in ranked_z_dim_list]]
            corr = corr.loc[self.pca_df.columns[:self.n_pcs]]

            corr.to_csv(dir_path / "correlations_pc_embeddings.csv")
            print('saved')
            if self.spearman is True:
                corr_spearman = df.corr(method="spearman")
                corr_spearman = corr_spearman[[f"mu_{col}" for col in ranked_z_dim_list]]
                corr_spearman = corr_spearman.loc[self.pca_df.columns[:self.n_pcs]]

                corr_spearman.to_csv(dir_path / "correlations_pc_embeddings_spearman.csv")
        if self.spearman is True:
            method = ['pearson', 'spearman']
            df_list = [corr, corr_spearman]
        else:
            method = ['pearson']
            df_list = [corr]
        for j, df in enumerate(df_list):
            print(j)
            # f, ax = plt.subplots(1, 1, figsize=(10, 10))
            # cbar_ax = f.add_axes([1, 0.35, 0.04, 0.3])
            # sns.set_context("talk")
            # g = sns.heatmap(corr.T, cmap="vlag", square=True, ax=ax, cbar_ax=cbar_ax,
            #                 xticklabels=True, yticklabels=True)
            # ax.set_xticklabels(ax.get_xmajorticklabels(), fontsize = 13)
            # ax.set_yticklabels(ax.get_ymajorticklabels(), fontsize = 13)
            # plt.tight_layout()
            # f.savefig(dir_path / "correlation_embeddings_PC.png")
            df = df.iloc[:, :8]
            f, ax = plt.subplots(1, 1, figsize=(6, 6))
            cbar_ax = f.add_axes([1, 0.35, 0.04, 0.3])
            sns.set_context("talk")
            g = sns.heatmap(df.T, cmap="vlag", square=True, ax=ax, cbar_ax=cbar_ax,
                            xticklabels=True, yticklabels=True, vmin=-1, vmax=1)
            ax.set_xticklabels(ax.get_xmajorticklabels(), fontsize = 16)
            ax.set_yticklabels(ax.get_ymajorticklabels(), fontsize = 16)
            plt.tight_layout()
            f.savefig(dir_path / f"correlation_{method[j]}_embeddings_PC.png")
