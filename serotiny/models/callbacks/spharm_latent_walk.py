from typing import Optional, Sequence, Tuple, Union

import torch
from torch import device, Tensor
from pytorch_lightning import Callback, LightningModule, Trainer
from pathlib import Path
import pandas as pd
from aicscytoparam import cytoparam
from ...metrics.utils import get_mesh_from_dataframe
import matplotlib
import matplotlib.pyplot as plt

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


class SpharmLatentWalk(Callback):
    """"""

    def __init__(
        self,
        save_dir: Optional[str] = None,
        condition: Optional[list] = None,
        latent_walk_range: Optional[list] = None,
        cutoff_kld_per_dim: Optional[float] = None,
    ):
        """
        Args:
            save_dir: Where to save plots
            Default: csv_logs folder

            condition: A condition to give the decoder
            Default None

            latent_walk_range: What range to do the latent walk
            Default = [-5, -2.5, 0, 2.5, 5]

            cutoff_kld_per_dim: Cutoff to use for KLD per dim
            Default = 0.5
        """
        super().__init__()

        self.save_dir = save_dir
        self.condition = condition
        self.cutoff_kld_per_dim = cutoff_kld_per_dim
        self.latent_walk_range = latent_walk_range
        dfg = pd.read_csv(
            "/allen/aics/modeling/ritvik/projects/serotiny/variance_spharm_coeffs.csv"
        )
        self.dna_spharm_cols = [col for col in dfg.columns if "dna_shcoeffs" in col]
        if self.latent_walk_range is None:
            self.latent_walk_range = [-5, -2.5, 0, 2.5, 5]

        if self.cutoff_kld_per_dim is None:
            self.cutoff_kld_per_dim = 0.5

    def to_device(
        self, batch_x: Sequence, batch_y: Sequence, device: Union[str, device]
    ) -> Tuple[Tensor, Tensor]:

        # last input is for online eval
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        return batch_x, batch_y

    def make_plots(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        ranked_z_dim_list: list,
        batch_size: int,
        latent_dims: int,
        c_shape: int,
    ):

        for rank, z_dim in enumerate(ranked_z_dim_list):
            for value in self.latent_walk_range:
                z_inf = torch.zeros(batch_size, latent_dims)
                z_inf[:, z_dim] = value
                z_inf = z_inf.cuda(device=0)
                z_inf = z_inf.float()
                decoder = pl_module.decoder

                y_test = torch.zeros(batch_size, c_shape)
                if self.condition:
                    y_test[:, 0] = self.condition

                z_inf, y_test = self.to_device(z_inf, y_test, pl_module.device)
                x_hat = decoder(z_inf, y_test)

                test_spharm = x_hat[0, :]
                test_spharm = test_spharm.detach().cpu().numpy()
                test_spharm = pd.DataFrame(test_spharm).T
                test_spharm.columns = self.dna_spharm_cols
                test_spharm_series = test_spharm.iloc[0]

                mesh = get_mesh_from_dataframe(test_spharm_series, "dna_shcoeffs_L", 32)
                img, origin = cytoparam.voxelize_meshes([mesh])

                for proj in [0, 1, 2]:
                    fig, ax = plt.subplots(figsize=(0.8, 0.8), dpi=80)
                    ax.set_xlim([0, 200])
                    ax.set_ylim([0, 200])
                    plt.imshow(img.max(proj), cmap="gray")
                    fig.savefig(
                        self.save_dir
                        / Path(
                            f"./latent_walk/z_dim_{z_dim}_rank_{rank}_"
                            + f"value_{value}_project_{proj}.png"
                        ),
                        dpi=200,
                    )

    def on_test_epoch_end(self, trainer: Trainer, pl_module: LightningModule):

        with torch.no_grad():
            dir_path = Path(trainer.logger[1].save_dir)
            stats = pd.read_csv(dir_path / "stats_per_dim_test.csv")

            if not self.save_dir:
                self.save_dir = dir_path

            if self.condition:
                stats = stats.loc[stats["condition"] == self.condition].reset_index(
                    drop=True
                )

            stats = (
                stats.loc[stats["test_kld_per_dim"] > self.cutoff_kld_per_dim]
                .sort_values(by=["test_kld_per_dim"])
                .reset_index(drop=True)
            )

            ranked_z_dim_list = [i for i in stats["dimension"][::-1]]

            batch_size = trainer.test_dataloaders[0].batch_size
            latent_dims = pl_module.encoder.enc_layers[-1]

            test_dataloader = trainer.test_dataloaders[0]
            test_iter = next(iter(test_dataloader))
            _, c_label, _ = [i for i in test_iter.keys()]
            c = test_iter[c_label]
            c_shape = c.shape

            self.make_plots(
                trainer, pl_module, ranked_z_dim_list, batch_size, latent_dims, c_shape
            )
