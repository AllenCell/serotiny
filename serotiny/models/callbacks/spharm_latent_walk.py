from typing import Optional, Sequence, Tuple, Union

import torch
from torch import device, Tensor
from pytorch_lightning import Callback, LightningModule, Trainer
from pathlib import Path
import pandas as pd
import numpy as np
from aicsshparam import shtools
from aicscytoparam import cytoparam
from cvapipe_analysis.tools.plotting import ShapeModePlotMaker
from vtk.util.numpy_support import vtk_to_numpy as vtk2np
import matplotlib.pyplot as plt
from matplotlib import animation
from serotiny.metrics.utils import get_mesh_from_series
import vtk
import operator
from functools import reduce
import math

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
    ):
        """
        Args:
            config: Config file used by Matheus in cvapipe_analysis

            save_dir: Where to save plots
            Default: csv_logs folder

            condition: A condition to give the decoder
            Default None

            latent_walk_range: What range to do the latent walk
            Default = [-2, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0]

            cutoff_kld_per_dim: Cutoff to use for KLD per dim
            Default = 0.5
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
            self.cutoff_kld_per_dim = 0.5

        if self.plot_limits is None:
            self.plot_limits = [-150, 150, -80, 80]

        if self.subfolder is None:
            self.subfolder = "gifs"

        self.config["pca"]["aliases"] = ["NUC"]

        self.config["pca"][
            "map_points"
        ] = self.latent_walk_range  # This is map points in std dev units

        self.config["pca"]["plot"]["limits"] = self.plot_limits
        self.plot_maker = None

    def to_device(
        self, batch_x: Sequence, batch_y: Sequence, device: Union[str, device]
    ) -> Tuple[Tensor, Tensor]:

        # last input is for online eval
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        return batch_x, batch_y

    def get_meshes(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        ranked_z_dim_list: list,
        mu_variance_list: list,
        batch_size: int,
        latent_dims: int,
        c_shape: int,
    ):
        save_dir = self.config["project"]["local_staging"] / self.subfolder
        meshes = {}
        for rank, z_dim in enumerate(ranked_z_dim_list):
            # Set subplots
            fig, ax_array = plt.subplots(
                3,
                len(self.latent_walk_range),
                squeeze=False,
                figsize=(15, 7),
            )
            meshes[rank] = {}
            for alias in self.config["pca"]["aliases"]:
                meshes[rank][alias] = []
                for value_index, value in enumerate(self.latent_walk_range):
                    z_inf = torch.zeros(batch_size, latent_dims)
                    z_inf[:, z_dim] = value * mu_variance_list[rank]
                    z_inf = z_inf.cuda(device=0)
                    z_inf = z_inf.float()
                    decoder = pl_module.decoder

                    y_test = torch.zeros(batch_size, c_shape)

                    z_inf, y_test = self.to_device(z_inf, y_test, pl_module.device)
                    x_hat = decoder(z_inf, y_test)

                    test_spharm = x_hat[0, :]
                    test_spharm = test_spharm.detach().cpu().numpy()
                    test_spharm = pd.DataFrame(test_spharm).T
                    test_spharm.columns = self.dna_spharm_cols
                    test_spharm_series = test_spharm.iloc[0]

                    mesh = get_mesh_from_series(test_spharm_series, "dna", 32)
                    img, origin = cytoparam.voxelize_meshes([mesh])
                    meshes[rank][alias].append(mesh)
                    for proj in [0, 1, 2]:
                        ax_array[proj, value_index].set_title(f"{value}" r"$\sigma$")
                        ax_array[proj, value_index].imshow(img.max(proj), cmap="gray")
                    print(f"Done making mesg for dim {z_dim}")
            [ax.axis("off") for ax in ax_array.flatten()]
            # Save figure
            ax_array.flatten()[0].get_figure().savefig(
                save_dir / f"dim_{z_dim}_rank_{rank}.png"
            )
            # Close figure, otherwise clogs memory
            plt.close(fig)

        return meshes

    @staticmethod
    def find_plane_mesh_intersection(mesh, proj):

        # Find axis orthogonal to the projection of interest
        axis = [a for a in [0, 1, 2] if a not in proj][0]

        # Get all mesh points
        points = vtk2np(mesh.GetPoints().GetData())

        if not np.abs(points[:, axis]).sum():
            raise Exception("Only zeros found in the plane axis.")

        mid = np.mean(points[:, axis])

        # Set the plane a little off center to avoid undefined intersections
        # Without this the code hangs when the mesh has any edge aligned with the
        # projection plane
        mid += 0.75
        offset = 0.1 * np.ptp(points, axis=0).max()

        # Create a vtkPlaneSource
        plane = vtk.vtkPlaneSource()
        plane.SetXResolution(4)
        plane.SetYResolution(4)
        if axis == 0:
            plane.SetOrigin(
                mid, points[:, 1].min() - offset, points[:, 2].min() - offset
            )
            plane.SetPoint1(
                mid, points[:, 1].min() - offset, points[:, 2].max() + offset
            )
            plane.SetPoint2(
                mid, points[:, 1].max() + offset, points[:, 2].min() - offset
            )
        if axis == 1:
            plane.SetOrigin(
                points[:, 0].min() - offset, mid, points[:, 2].min() - offset
            )
            plane.SetPoint1(
                points[:, 0].min() - offset, mid, points[:, 2].max() + offset
            )
            plane.SetPoint2(
                points[:, 0].max() + offset, mid, points[:, 2].min() - offset
            )
        if axis == 2:
            plane.SetOrigin(
                points[:, 0].min() - offset, points[:, 1].min() - offset, mid
            )
            plane.SetPoint1(
                points[:, 0].min() - offset, points[:, 1].max() + offset, mid
            )
            plane.SetPoint2(
                points[:, 0].max() + offset, points[:, 1].min() - offset, mid
            )
        plane.Update()
        plane = plane.GetOutput()

        # Trangulate the plane
        triangulate = vtk.vtkTriangleFilter()
        triangulate.SetInputData(plane)
        triangulate.Update()
        plane = triangulate.GetOutput()

        # Calculate intersection
        intersection = vtk.vtkIntersectionPolyDataFilter()
        intersection.SetInputData(0, mesh)
        intersection.SetInputData(1, plane)
        intersection.Update()
        intersection = intersection.GetOutput()

        # Get coordinates of intersecting points
        points = vtk2np(intersection.GetPoints().GetData())

        # Sorting points clockwise
        # This has been discussed here:
        # https://stackoverflow.com/questions/51074984/sorting-according-to-clockwise-point-coordinates/51075469
        # but seems not to be very efficient. Better version is proposed here:
        # https://stackoverflow.com/questions/57566806/how-to-arrange-the-huge-list-of-2d-coordinates-in-a-clokwise-direction-in-python
        coords = points[:, proj]
        center = tuple(
            map(
                operator.truediv,
                reduce(lambda x, y: map(operator.add, x, y), coords),
                [len(coords)] * 2,
            )
        )
        coords = sorted(
            coords,
            key=lambda coord: (
                -135
                - math.degrees(
                    math.atan2(*tuple(map(operator.sub, coord, center))[::-1])
                )
            )
            % 360,
        )

        # Store sorted coordinates
        # points[:, proj] = coords
        return np.array(coords)

    @staticmethod
    def animate_contours(contours, prefix, config, save_dir, subfolder):
        nbins = len(config["pca"]["map_points"])
        hmin, hmax, vmin, vmax = config["pca"]["plot"]["limits"]
        offset = 0.05 * (hmax - hmin)

        fig, ax = plt.subplots(1, 1, figsize=(3, 3))
        plt.tight_layout()
        plt.close()
        ax.set_xlim(hmin - offset, hmax + offset)
        ax.set_ylim(vmin - offset, vmax + offset)
        ax.set_aspect("equal")

        lines = []
        for alias, _ in contours.items():
            for obj, value in config["data"]["segmentation"].items():
                if value["alias"] == alias:
                    break
            (line,) = ax.plot([], [], lw=2, color=value["color"])
            lines.append(line)

        def animate(i):
            for alias, line in zip(contours.keys(), lines):
                ct = contours[alias][i]
                mx = ct[:, 0]
                my = ct[:, 1]
                line.set_data(mx, my)
            return lines

        anim = animation.FuncAnimation(
            fig, animate, frames=nbins, interval=100, blit=True
        )
        fname = save_dir / f"{subfolder}/{prefix}.gif"
        anim.save(fname, writer="imagemagick", fps=nbins)
        plt.close("all")
        return

    def on_test_epoch_end(self, trainer: Trainer, pl_module: LightningModule):

        with torch.no_grad():
            dir_path = Path(trainer.logger[1].save_dir)

            self.config["project"][
                "local_staging"
            ] = dir_path  # This is where plots get saved

            subdir = dir_path / self.subfolder
            subdir.mkdir(parents=True, exist_ok=True)

            plot_maker = ShapeModePlotMaker(
                config=self.config, subfolder=self.subfolder
            )

            stats = pd.read_csv(dir_path / "stats_per_dim_test.csv")

            stats = (
                stats.loc[stats["test_kld_per_dim"] > self.cutoff_kld_per_dim]
                .sort_values(by=["test_kld_per_dim"])
                .reset_index(drop=True)
            )

            ranked_z_dim_list = [i for i in stats["dimension"][::-1]]
            mu_variance_list = [i for i in stats["mu_std_per_dim"][::-1]]

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

            meshes = self.get_meshes(
                trainer,
                pl_module,
                ranked_z_dim_list,
                mu_variance_list,
                batch_size,
                latent_dims,
                c_shape,
            )

            for shapemode, mesh_dict in meshes.items():
                projections = {}
                projs = [[0, 1], [0, 2], [1, 2]]
                for dim, proj in zip(["z", "y", "x"], projs):
                    projections[dim] = {}
                    for alias, mesh_list in mesh_dict.items():
                        print(dim, alias)
                        projections[dim][alias] = []
                        for mesh in mesh_list:
                            coords = self.find_plane_mesh_intersection(mesh, proj)
                            projections[dim][alias].append(coords)
                            print("done projecting single mesh")
                # return contours
                # projections = plot_maker.get_2d_contours(mesh_dict)
                print(f"Done with all projections for shapemode {shapemode}")
                for proj, contours in projections.items():
                    print(f"Beginning gif generation for proj {proj}")
                    self.animate_contours(
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
