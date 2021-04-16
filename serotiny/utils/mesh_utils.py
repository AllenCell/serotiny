from pytorch_lightning import LightningModule
from aicsshparam import shtools
from aicscytoparam import cytoparam
from vtk.util.numpy_support import vtk_to_numpy as vtk2np
import matplotlib.pyplot as plt
from matplotlib import animation
import vtk
import operator
from functools import reduce
import math
import torch
from .model_utils import to_device
import pandas as pd
import numpy as np


def get_meshes(
    pl_module: LightningModule,
    ranked_z_dim_list: list,
    mu_variance_list: list,
    batch_size: int,
    latent_dims: int,
    c_shape: int,
    subfolder: str,
    latent_walk_range: list,
    config: dict,
    dna_spharm_cols,
):
    save_dir = config["project"]["local_staging"] / subfolder
    meshes = {}
    for rank, z_dim in enumerate(ranked_z_dim_list):
        # Set subplots
        fig, ax_array = plt.subplots(
            3,
            len(latent_walk_range),
            squeeze=False,
            figsize=(15, 7),
        )
        meshes[rank] = {}
        for alias in config["shapespace"]["aliases"]:
            meshes[rank][alias] = []
            for value_index, value in enumerate(latent_walk_range):
                z_inf = torch.zeros(batch_size, latent_dims)
                z_inf[:, z_dim] = value * mu_variance_list[rank]
                z_inf = z_inf.cuda(device=0)
                z_inf = z_inf.float()
                decoder = pl_module.decoder

                y_test = torch.zeros(batch_size, c_shape)

                z_inf, y_test = to_device(z_inf, y_test, pl_module.device)
                x_hat = decoder(z_inf, y_test)

                test_spharm = x_hat[0, :]
                test_spharm = test_spharm.detach().cpu().numpy()
                test_spharm = pd.DataFrame(test_spharm).T
                test_spharm.columns = dna_spharm_cols
                test_spharm_series = test_spharm.iloc[0]

                mesh = get_mesh_from_series(test_spharm_series, "dna", 32)
                img, origin = cytoparam.voxelize_meshes([mesh])
                meshes[rank][alias].append(mesh)
                for proj in [0, 1, 2]:
                    ax_array[proj, value_index].set_title(f"{value}" r"$\sigma$")
                    ax_array[proj, value_index].imshow(img.max(proj), cmap="gray")
                print(f"Done making mesh for dim {z_dim}")
        [ax.axis("off") for ax in ax_array.flatten()]
        # Save figure
        ax_array.flatten()[0].get_figure().savefig(
            save_dir / f"dim_{z_dim}_rank_{rank}.png"
        )
        # Close figure, otherwise clogs memory
        plt.close(fig)

    return meshes


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
        plane.SetOrigin(mid, points[:, 1].min() - offset, points[:, 2].min() - offset)
        plane.SetPoint1(mid, points[:, 1].min() - offset, points[:, 2].max() + offset)
        plane.SetPoint2(mid, points[:, 1].max() + offset, points[:, 2].min() - offset)
    if axis == 1:
        plane.SetOrigin(points[:, 0].min() - offset, mid, points[:, 2].min() - offset)
        plane.SetPoint1(points[:, 0].min() - offset, mid, points[:, 2].max() + offset)
        plane.SetPoint2(points[:, 0].max() + offset, mid, points[:, 2].min() - offset)
    if axis == 2:
        plane.SetOrigin(points[:, 0].min() - offset, points[:, 1].min() - offset, mid)
        plane.SetPoint1(points[:, 0].min() - offset, points[:, 1].max() + offset, mid)
        plane.SetPoint2(points[:, 0].max() + offset, points[:, 1].min() - offset, mid)
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
            - math.degrees(math.atan2(*tuple(map(operator.sub, coord, center))[::-1]))
        )
        % 360,
    )

    # Store sorted coordinates
    # points[:, proj] = coords
    return np.array(coords)


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

    anim = animation.FuncAnimation(fig, animate, frames=nbins, interval=100, blit=True)
    fname = save_dir / f"{subfolder}/{prefix}.gif"
    anim.save(fname, writer="imagemagick", fps=nbins)
    plt.close("all")
    return


def get_shcoeff_matrix_from_dataframe(row: pd.Series, prefix: str, lmax: int):

    """
    Reshape spherical harmonics expansion (SHE) coefficients
    into a coefficients matrix of shape 2 x lmax x lmax, where
    lmax is the degree of the expansion.
    Parameters
    --------------------
    row: pd.Series
        Series that contains the SHE coefficients.
    prefix: str
        String to identify the keys of the series that contain
        the SHE coefficients.
    lmax: int
        Degree of the expansion
    Returns
    -------
        coeffs: np.array
            Array of shape 2 x lmax x lmax that contains the
            SHE coefficients.
    """

    # Empty matrix to store the SHE coefficients
    coeffs = np.zeros((2, lmax, lmax), dtype=np.float32)

    for l in range(lmax):
        for m in range(l + 1):
            try:
                # Cosine SHE coefficients
                coeffs[0, l, m] = row[
                    [f for f in row.keys() if f"{prefix}{l}M{m}C" in f]
                ]
                # Sine SHE coefficients
                coeffs[1, l, m] = row[
                    [f for f in row.keys() if f"{prefix}{l}M{m}S" in f]
                ]
            # If a given (l,m) pair is not found, it is
            # assumed to be zero
            except:
                pass

    # Error if no coefficients were found.
    if not np.abs(coeffs).sum():
        raise Exception(f"No coefficients found. Please check prefix: {prefix}")

    return coeffs


def get_mesh_from_dataframe(row: pd.Series, prefix: str, lmax: int):

    """
    Reconstruct the 3D triangle mesh corresponding to SHE
    coefficients stored in a pandas Series format.
    Parameters
    --------------------
    row: pd.Series
        Series that contains the SHE coefficients.
    prefix: str
        String to identify the keys of the series that contain
        the SHE coefficients.
    lmax: int
        Degree of the expansion
    Returns
    -------
        mesh: vtk.vtkPolyData
            Triangle mesh.
    """

    # Reshape SHE coefficients
    coeffs = get_shcoeff_matrix_from_dataframe(row=row, prefix=prefix, lmax=lmax)

    # Use aicsshparam to convert SHE coefficients into
    # triangle mesh
    mesh, _ = shtools.get_reconstruction_from_coeffs(coeffs)

    return mesh


def get_mesh_from_series(row, alias, lmax):
    coeffs = np.zeros((2, lmax, lmax), dtype=np.float32)
    for l in range(lmax):
        for m in range(l + 1):
            try:
                # Cosine SHE coefficients
                coeffs[0, l, m] = row[
                    [f for f in row.keys() if f"{alias}_shcoeffs_L{l}M{m}C" in f]
                ]
                # Sine SHE coefficients
                coeffs[1, l, m] = row[
                    [f for f in row.keys() if f"{alias}_shcoeffs_L{l}M{m}S" in f]
                ]
            # If a given (l,m) pair is not found, it is assumed to be zero
            except:
                pass
    mesh, _ = shtools.get_reconstruction_from_coeffs(coeffs)
    return mesh
