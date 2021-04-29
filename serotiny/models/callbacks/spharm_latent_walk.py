from typing import Optional

import torch
from pytorch_lightning import Callback, LightningModule, Trainer
from pathlib import Path
from cvapipe_analysis.tools.plotting import ShapeModePlotMaker
from cvapipe_analysis.tools import controller
from serotiny.utils.model_utils import get_ranked_dims


from serotiny.utils.mesh_utils import (
    get_meshes,
    # find_plane_mesh_intersection,
    # animate_contours,
)


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
            self.subfolder = "latent_walks"

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

            print(self.config["shapespace"])

            subdir = dir_path / self.subfolder
            subdir.mkdir(parents=True, exist_ok=True)

            control = controller.Controller(self.config)
            plot_maker = ShapeModePlotMaker(control, subfolder=self.subfolder)

            ranked_z_dim_list, mu_std_list, mu_mean_list = get_ranked_dims(
                dir_path, self.cutoff_kld_per_dim, max_num_shapemodes=8
            )

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
                mu_std_list,
                mu_mean_list,
                batch_size,
                latent_dims,
                c_shape,
                self.subfolder,
                self.latent_walk_range,
                self.config,
                self.dna_spharm_cols,
            )

            print(self.config["shapespace"])

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
                                coords = plot_maker.find_plane_mesh_intersection(
                                    mesh, proj
                                )
                                projections[dim][alias].append(coords)
                                print("done projecting single mesh")
                    # return contours
                    # projections = plot_maker.get_2d_contours(mesh_dict)

                    print(f"Done with all projections for shapemode {shapemode}")
                    # import ipdb
                    # ipdb.set_trace()
                    for proj, contours in projections.items():
                        print(f"Beginning gif generation for proj {proj}")
                        plot_maker.animate_contours(
                            contours,
                            f"dna_PC{shapemode + 1}_{proj}",
                            # self.config,
                            # dir_path,
                            # self.subfolder,
                        )
                    print("Done gif generation for shapemode {shapemode}")
                print("Combing all gifs into single plot")
                # shapemodes = [str(i) for i in range(len(ranked_z_dim_list))]
                plot_maker.combine_and_save_animated_gifs()
