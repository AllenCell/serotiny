import torch
from pytorch_lightning import Callback, LightningModule, Trainer
from pathlib import Path
from serotiny.utils.model_utils import get_all_embeddings


class GetEmbeddings(Callback):
    """"""

    def __init__(
        self,
        x_label: str,
        c_label: str,
        id_fields: list,
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

    def on_test_epoch_end(self, trainer: Trainer, pl_module: LightningModule):

        with torch.no_grad():
            dir_path = Path(trainer.logger[1].save_dir)

            result = get_all_embeddings(
                trainer.train_dataloader,
                trainer.val_dataloaders[0],
                trainer.test_dataloaders[0],
                pl_module,
                self.x_label,
                self.c_label,
                self.id_fields,
            )

            path = dir_path / "embeddings_all.csv"

            if path.exists():
                result.to_csv(path, mode="a", header=False, index=False)
            else:
                result.to_csv(path, header="column_names", index=False)
