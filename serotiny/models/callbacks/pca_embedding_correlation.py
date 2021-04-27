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

        df = pd.concat(
            [
                pd.read_csv(dir_path / "embeddings/embeddings_all.csv")
                .set_index("CellId")
                .sort_index(),
                self.pca_df,
            ],
            axis=1,
        )

        corr = df.corr()
        corr = corr[[col for col in corr.columns if col not in self.pca_df.columns]]
        corr = corr.loc[self.pca_df.columns]

        f, ax = plt.subplots(figsize=(60, 20))
        sns.set_context("talk")
        sns.heatmap(
            corr.T, cmap="vlag", square=True, ax=ax, xticklabels=True, yticklabels=True
        )
        f.savefig(dir_path / "correlation_embeddings_PC.png")
