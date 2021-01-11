#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import logging

from pathlib import Path
from typing import Dict, List, Optional, Union

import pandas as pd

from aics_dask_utils import DistributedHandler
from datastep import Step, log_run_params

from ...constants import DatasetFields
from ..load_data import LoadData, REQUIRED_FIELDS
from ...library.csv import load_csv
from ...library.image import project_2d, png_loader

###############################################################################

log = logging.getLogger(__name__)

###############################################################################


class Project2D(Step):
    def __init__(
        self,
        direct_upstream_tasks: List["Step"] = [LoadData],
        config: Optional[Union[str, Path, Dict[str, str]]] = None,
    ):

        self.possible_labels = [
            DatasetFields.DraftMitoticStateResolved,
            DatasetFields.DraftMitoticStateCoarse,
            DatasetFields.ExpertMitoticStateResolved,
            DatasetFields.ExpertMitoticStateCoarse,
        ]

        super().__init__(direct_upstream_tasks=direct_upstream_tasks, config=config)

    @log_run_params
    def run(
        self,
        dataset: Union[str, Path, pd.DataFrame],
        projection,
        label: str,
        executor_address: Optional[str] = None,
        **kwargs,
    ):

        dataset = load_csv(dataset, [])

        axis = projection["axis"]
        chosen = DatasetFields.Chosen2DProjectionPath

        def find_dimensions(png_path):
            data = png_loader(png_path)
            return tuple(data.shape[-2:])

        # if axis is 'all', use precalcuated all projection
        if axis == ["all"]:
            print("using actk all axes projections")
            dataset[chosen] = dataset[DatasetFields.CellImage2DAllProjectionsPath]
        else:
            channels = projection["channels"]
            method = projection["method"]
            masks = projection.get("masks", {})

            # if args match other precalculated projection, use that directly
            if (
                axis == "Z"
                and method == "max"
                and channels == ["membrane", "structure", "dna"]
                and masks
                == {"membrane": "membrane_segmentation", "dna": "nucleus_segmentation"}
            ):
                print("using actk Z projections")
                dataset[chosen] = dataset[DatasetFields.CellImage2DYXProjectionPath]
            else:
                # find the right projection
                out_path = Path(projection["output"])
                channel_str = ".".join(channels)
                projection_path = f"{axis}.{method}.{channel_str}"
                if masks:
                    mask_paths = [
                        f"{channel}-{mask}" for channel, mask in masks.items()
                    ]
                    mask_path = ".".join(mask_paths)
                    projection_path += f".{mask_path}"

                out_images = []
                projections = []

                for path in dataset[DatasetFields.CellImage3DPath]:
                    # get the 3d image path
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
                        # project_2d(path_3d, axis, method, path_2d, channels)
                        projections.append((path_3d, path_2d))

                # add the new column of projected images to the dataset
                dataset[chosen] = out_images

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
                        )

        dimensions = find_dimensions(dataset[chosen][0])

        # choose mitotic label
        if label in self.possible_labels:
            chosen = DatasetFields.ChosenMitoticClass
            dataset[chosen] = dataset[label]
        else:
            raise Exception(
                (
                    f"mitotic classes available are {self.possible_labels}, "
                    f"not {label}"
                )
            )

        projected_fields = REQUIRED_FIELDS["actk_manifest"] + [
            DatasetFields.ChosenMitoticClass,
            DatasetFields.Chosen2DProjectionPath,
        ]

        self.manifest = dataset[projected_fields]

        manifest_save_path = self.step_local_staging_dir / "manifest.csv"
        self.manifest.to_csv(manifest_save_path, index=False)

        return {"manifest": manifest_save_path, "dimensions": dimensions}
