from typing import Optional

import torch
from pytorch_lightning import Callback, LightningModule, Trainer
from pathlib import Path
import pandas as pd
from cvapipe_analysis.tools.plotting import ShapeModePlotMaker


from serotiny.utils.mesh_utils import (
    get_meshes,
    find_plane_mesh_intersection,
    animate_contours,
)


# Configure z for inference
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


class SpharmLatentWalk(Callback):
    """"""

    def __init__(
        self,
        config: dict,
        spharm_coeffs_cols: list,
        latent_walk_range: Optional[list] = None,
        cutoff_kld_per_dim: Optional[float] = None,
        plot_limits: Optional[list] = None,
        subfolder: Optional[str] = None,
        ignore_mesh_and_contour_plots: Optional[bool] = None,
    ):
        """
        Args:
            config: Config file used by Matheus in cvapipe_analysis

            spharm_coeffs_cols: List of column names of the spharm coeffs
            Example: ["dna_spharm_L0M1",..]

            latent_walk_range: What range to do the latent walk
            Example = [-2, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0]

            cutoff_kld_per_dim: Cutoff to use for KLD per dim
            Example = 0.5

            plot_limits: Limits for plot

            subfolder: Subfolder name to save gifs
        """
        super().__init__()

        self.config = config
        self.cutoff_kld_per_dim = cutoff_kld_per_dim
        self.latent_walk_range = latent_walk_range
        self.plot_limits = plot_limits
        self.subfolder = subfolder
        self.dna_spharm_cols = spharm_coeffs_cols

        if self.latent_walk_range is None:
            # self.latent_walk_range = [-2, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0]
            self.latent_walk_range = [
                -2,
                -1,
                -0.5,
                -0.25,
                -0.1,
                0,
                0.1,
                0.25,
                0.5,
                1,
                2,
            ]

        if self.cutoff_kld_per_dim is None:
            self.cutoff_kld_per_dim = 0

        if self.plot_limits is None:
            self.plot_limits = [-150, 150, -80, 80]

        if self.subfolder is None:
            self.subfolder = "gifs"

        self.config["shapespace"]["aliases"] = ["NUC"]

        self.config["shapespace"][
            "map_points"
        ] = self.latent_walk_range  # This is map points in std dev units

        self.config["shapespace"]["plot"]["limits"] = self.plot_limits
        self.plot_maker = None
        self.ignore_mesh_and_contour_plots = ignore_mesh_and_contour_plots

    def on_test_epoch_end(self, trainer: Trainer, pl_module: LightningModule):

        with torch.no_grad():
            dir_path = Path(trainer.logger[1].save_dir)

            self.config["project"][
                "local_staging"
            ] = dir_path  # This is where plots get saved

            subdir = dir_path / self.subfolder
            subdir.mkdir(parents=True, exist_ok=True)

            from cvapipe_analysis.tools import controller

            control = controller.Controller(self.config)
            plot_maker = ShapeModePlotMaker(control, subfolder=self.subfolder)

            stats = pd.read_csv(dir_path / "stats_per_dim_test.csv")

            stats = (
                stats.loc[stats["test_kld_per_dim"] > self.cutoff_kld_per_dim]
                .sort_values(by=["test_kld_per_dim"])
                .reset_index(drop=True)
            )

            ranked_z_dim_list = [i for i in stats["dimension"][::-1]]
            mu_variance_list = [i for i in stats["mu_std_per_dim"][::-1]]
            mu_mean_list = [i for i in stats["mu_mean_per_dim"][::-1]]

            num_shapemodes = 8
            if len(ranked_z_dim_list) > num_shapemodes:
                ranked_z_dim_list = ranked_z_dim_list[:num_shapemodes]
                mu_variance_list = mu_variance_list[:num_shapemodes]

            batch_size = trainer.test_dataloaders[0].batch_size
            latent_dims = pl_module.encoder.enc_layers[-1]

            test_dataloader = trainer.test_dataloaders[0]
            test_iter = next(iter(test_dataloader))
            _, c_label, _, _ = [i for i in test_iter.keys()]
            c = test_iter[c_label]
            c_shape = c.shape[-1]

            meshes = get_meshes(
                pl_module,
                ranked_z_dim_list,
                mu_variance_list,
                mu_mean_list,
                batch_size,
                latent_dims,
                c_shape,
                self.subfolder,
                self.latent_walk_range,
                self.config,
                self.dna_spharm_cols,
            )

            if not self.ignore_mesh_and_contour_plots:
                for shapemode, mesh_dict in meshes.items():
                    projections = {}
                    projs = [[0, 1], [0, 2], [1, 2]]
                    for dim, proj in zip(["z", "y", "x"], projs):
                        projections[dim] = {}
                        for alias, mesh_list in mesh_dict.items():
                            print(dim, alias)
                            projections[dim][alias] = []
                            for mesh in mesh_list:
                                coords = find_plane_mesh_intersection(mesh, proj)
                                projections[dim][alias].append(coords)
                                print("done projecting single mesh")
                    # return contours
                    # projections = plot_maker.get_2d_contours(mesh_dict)
                    print(f"Done with all projections for shapemode {shapemode}")
                    for proj, contours in projections.items():
                        print(f"Beginning gif generation for proj {proj}")
                        animate_contours(
                            contours,
                            f"{shapemode}_{proj}",
                            self.config,
                            dir_path,
                            self.subfolder,
                        )
                    print("Done gif generation for shapemode {shapemode}")
                print("Combing all gifs into single plot")
                shapemodes = [str(i) for i in range(len(ranked_z_dim_list))]
                plot_maker.combine_and_save_animated_gifs(shapemodes)
