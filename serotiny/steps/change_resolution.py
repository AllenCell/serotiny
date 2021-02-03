#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import fire
import logging

from pathlib import Path
from typing import Optional

from aics_dask_utils import DistributedHandler

from ..library.csv import load_csv
from ..library.image import change_resolution

###############################################################################

log = logging.getLogger(__name__)

###############################################################################


def find_resolution_path(resolution):
    """
    Create folder name from resolution parameter
    Parameters
    ----------
    resolution: Union[scalar, list]
       Resolution scaling factor or desired ZYX dimensions (list of 3)
    Returns
    -------
    resolution_path: str
       String containing folder name based on resolution parameter
    """
    if isinstance(resolution, list):
        if len(resolution) != 3:
            raise Exception(f"Resolution must be three long (Z Y X) not {len(resolution)}")
        resolution_path = f"ResolutionZ{resolution[0]}Y{resolution[1]}X{resolution[2]}"
    else:
        resolution_path = f"Resolution_scale{resolution}"

    return resolution_path


def apply_resolution(
    manifest_in,
    path_3d_column: str,
    manifest_out,
    path_3d_resized_column: str,
    path_out,
    resolution,
    executor_address: Optional[str] = None,
):

    """
    Changes the resolution of a set of 3D OME TIFF files
    Parameters
    ----------
    manifest_in: Union[str, Path]
        The path to the manifest csv file with original images
    path_3d_column: str
        The name of the column in manifest_in that lists the paths to the input OME TIFF files
    manifest_out: Union[str, Path]
        The path to the manifest csv file with resampled images
    path_3d_resized_column: str
        The name of the column in manifest_out that lists the paths to the output OME TIFF files
    path_out: Union[str, Path]
        The path to the root folder where the output OME TIFF files is stored
    resolution: Union[scalar, list]
        Resolution scaling factor or desired ZYX dimensions (list of 3)
    executor_address: Optional[str] = None
            Executor address
    Returns
    -------
    data_new.shape: Tuple
        Tuple that contains the image dimensions of output image
    """

    # Create folder name from resolution parameter
    resolution_path = find_resolution_path(resolution)
    # print(f"applying resolution change {resolution_path}")

    # Load up data manifest
    dataset = load_csv(manifest_in, [])

    # Preparing variables for resizing of images
    out_images = []
    images_to_resize = []
    resolution_dir = Path(path_out) / resolution_path
    resolution_dir.mkdir(parents=True, exist_ok=True)

    for path in dataset[path_3d_column]:
        # get the 3d image path
        path_3d = Path(path)
        full_name = path_3d.name

        # save the resized images with the same name as the 3d path
        path_res = resolution_dir / full_name
        out_images.append(str(path_res))

        if not path_res.exists():
            # change the resolution and save
            images_to_resize.append((path_3d, path_res))

    # add the new column of projected images to the dataset
    dataset[path_3d_resized_column] = out_images

    # if we have any images_to_resize to compute use the distributed handler
    if images_to_resize:
        with DistributedHandler(executor_address) as handler:
            handler.batched_map(
                change_resolution,
                [paths[0] for paths in images_to_resize],
                [paths[1] for paths in images_to_resize],
                [resolution for _ in range(len(images_to_resize))],
            )

    dataset.to_csv(manifest_out, index=False)

if __name__ == "__main__":
    # example command:
    # python -m  serotiny.steps.change_resolution \
    # --manifest_in /allen/aics/modeling/theok/Projects/Data/idle/mcw/smalltestdataset/manifest.csv \
    # --path_3d_column CellImage3DPath \
    # --manifest_out /allen/aics/modeling/theok/Projects/Data/idle/mcw/smalltestdataset/out/manifest.csv \
    # --path_3d_resized_column CellSampledImage3DPath \
    # --path_out /allen/aics/modeling/theok/Projects/Data/idle/mcw/smalltestdataset/out/ \
    # --resolution [10,20,50]

    fire.Fire(apply_resolution)
