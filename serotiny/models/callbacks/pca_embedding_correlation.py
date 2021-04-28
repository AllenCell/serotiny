from pathlib import Path

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from pytorch_lightning import Callback, LightningModule, Trainer

class PCALatentCorrelation(Callback):
    def __init__(self, pca_df):
        self.pca_df = pca_df.set_index("CellId").sort_index()

    def on_test_epoch_end(self, trainer: Trainer, pl_module: LightningModule):
        dir_path = Path(trainer.logger[1].save_dir)

        df = pd.concat([
            pd.read_csv(dir_path / "embeddings_all.csv")
                .set_index("CellId")
                .sort_index(),
            self.pca_df
        ], axis=1)

        corr = df.corr()
        corr = corr[[col for col in corr.columns if col not in self.pca_df.columns]]
        corr = corr.loc[self.pca_df.columns]

        corr.to_csv(dir_path / "correlations_pc_embeddings.csv")

        grid_kws = {"height_ratios": (.98, .02)}
        f, (ax, cbar_ax) = plt.subplots(2, figsize=(75, 30), gridspec_kw=grid_kws)
        sns.set_context("talk")
        g = sns.heatmap(corr.T, cmap="vlag", square=True, ax=ax, cbar_ax=cbar_ax,
                        cbar_kws={"orientation": "horizontal"},
                        xticklabels=True, yticklabels=True)
        ax.set_xticklabels(ax.get_xmajorticklabels(), fontsize = 16)
        ax.set_yticklabels(ax.get_ymajorticklabels(), fontsize = 16)
        plt.tight_layout()
        f.savefig(dir_path / "correlation_embeddings_PC.png")
