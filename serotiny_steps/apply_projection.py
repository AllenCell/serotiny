import os
import logging

from pathlib import Path
from typing import Optional

import fire

from aics_dask_utils import DistributedHandler

from serotiny.io.dataframe import read_dataframe
from serotiny.io.image import project_2d, png_loader

###############################################################################

log = logging.getLogger(__name__)

###############################################################################


def find_projection_path(projection):
    axis = projection["axis"]
    channels = projection["channels"]
    method = projection["method"]
    masks = projection.get("masks", {})

    # find the right projection
    channel_str = ".".join(channels)
    projection_path = f"{axis}.{method}.{channel_str}"
    if masks:
        mask_paths = [f"{channel}-{mask}" for channel, mask in masks.items()]
        mask_path = ".".join(mask_paths)
        projection_path += f".{mask_path}"

    return projection_path


def apply_projection(
    dataset_path: str,
    output_path: str,
    projection: dict,
    path_3d_column: str,
    chosen_projection: str,
    proj_all: bool,
    executor_address: Optional[str] = None,
    source_path: Optional[str] = None,
):
    """
    Compute projection of 3D images

    Parameters
    -----------
    dataset_path: str
        Path to dataset containing 3D images

    output_path: str
        Path to save projection images

    projection: dict
        Dictionary config containing details like
        channels, method, masks, axis

    path_3d_column: str
        Column containing 3D image paths

    chosen_projection: str
        Add the new column of proejcted images
        to the dataset with this key

    proj_all: bool
        Whether to do all axis max projection or not

    executor_address: Optional[str] = None
        Address for dask distributed compute

    source_path: Optional[str] = None
        If source path, append source path to
        3D image paths
    """

    print(f"applying projection {projection}")

    dataset = read_dataframe(dataset_path, [])
    axis = projection["axis"]

    def find_dimensions(png_path):
        data = png_loader(png_path)
        return tuple(data.shape[-2:])

    channels = projection["channels"]
    method = projection["method"]
    masks = projection.get("masks", {})
    out_path = Path(projection["output"])

    # find the right projection
    projection_path = find_projection_path(projection)

    out_images = []
    projections = []

    for path in dataset[path_3d_column]:
        # get the 3d image path
        if source_path is not None:
            path = os.path.join(source_path, os.path.basename(path))

        path_3d = Path(path)
        full_name = path_3d.name

        # get the root image name
        image, tiff = os.path.splitext(full_name)
        image, ome = os.path.splitext(image)

        # save the 2d path with the same name as the 3d path
        projection_dir = out_path / projection_path
        projection_dir.mkdir(parents=True, exist_ok=True)
        path_2d = projection_dir / f"{image}.png"
        out_images.append(str(path_2d))

        if not path_2d.exists():
            # calculate the chosen projection and save
            projections.append((path_3d, path_2d))

    # add the new column of projected images to the dataset
    dataset[chosen_projection] = out_images

    # if we have any projections to compute use the distributed handler
    if projections:
        with DistributedHandler(executor_address) as handler:
            handler.batched_map(
                project_2d,
                [paths[0] for paths in projections],
                [axis for _ in range(len(projections))],
                [method for _ in range(len(projections))],
                [paths[1] for paths in projections],
                [channels for _ in range(len(projections))],
                [masks for _ in range(len(projections))],
                [proj_all for _ in range(len(projections))],
            )

    dataset.to_csv(output_path, index=False)


if __name__ == "__main__":
    # example command:
    # python -m serotiny.steps.apply_projection \
    #     --dataset_path "data/manifest_merged.csv" \
    #     --output_path "data/projection.csv" \
    #     --projection "{'channels': ['membrane', 'structure', 'dna'],
    # 'masks':
    # {'membrane': 'membrane_segmentation', 'dna': 'nucleus_segmentation'},
    # 'axis': 'Y', 'method': 'max',
    # 'output':
    # '/allen/aics/modeling/spanglry/data/mitotic-classifier/projections/'}" \
    #     --path_3d_column "CellImage3DPath" \
    #     --chosen_projection "Chosen2DProjectionPath" \
    #     --chosen_class "ChosenMitoticClass" \
    #     --label "Draft mitotic state resolved"

    fire.Fire(apply_projection)
