#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pathlib import Path
from typing import Optional
import logging

import fire
import pandas as pd

from actk.steps.diagnostic_sheets import DiagnosticSheets
from .apply_projection import apply_projection

###############################################################################

log = logging.getLogger(__name__)

###############################################################################


def diagnostic_sheets(
    dataset_path: str,
    output_path: str,
    projection: dict,
    path_3d_column: str, # ="CellImage3DPath",
    chosen_projection: str, # ="Chosen2DProjectionPath",
    chosen_class: str, # ="ChosenMitoticClass",
    label: str, # ="Draft mitotic state resolved",
    source_path: str, # =None,
    metadata: str, # ="ChosenMitoticClass",
    proj_all: bool, # =True,
    num_images: int, # =200,
    executor_address: Optional[str] = None,
    overwrite: bool = False,
):

    # make a visualization manifest
    dataset_path = Path(dataset_path)
    manifest_visualization_path = output_path / Path("manifest_visualization.csv")

    # Sample input dataset
    dataset = pd.read_csv(dataset_path)
    sampled_dataset = dataset.sample(n=num_images)

    # Update projection config output path for apply_projection function
    projection["output"] = output_path

    # Get all projection images
    # Only difference here is proj_all is True
    apply_projection(
        dataset_path=sampled_dataset,
        output_path=manifest_visualization_path,
        projection=projection,
        path_3d_column=path_3d_column,
        chosen_projection=chosen_projection,
        chosen_class=chosen_class,
        label=label,
        source_path=source_path,
        proj_all=proj_all,
        executor_address=executor_address,
    )

    # Set local staging dir for datastep step
    workflow_config = {"project_local_staging_dir": output_path}

    # Call diagnsotic sheet step
    diagnostic_sheets_class = DiagnosticSheets(config=workflow_config)

    visualization_dataframe = pd.read_csv(manifest_visualization_path)

    # Make sure no / in the class labels because code reads it as a split directory
    # and spits out errors
    if metadata == "ChosenMitoticClass":
        visualization_dataframe[metadata] = visualization_dataframe[metadata].map(
            {
                "M4/M5": "M4_M5",
                "M1/M2": "M1_M2",
                "M6/M7": "M6_M7",
                "M0": "M0",
                "M3": "M3",
            }
        )

    diagnostic_sheets_class.run(
        dataset=visualization_dataframe,
        distributed_executor_address=executor_address,
        metadata=metadata,
        overwrite=overwrite,
    )


if __name__ == "__main__":
    # example command:
    # python -m serotiny.steps.diagnostic_sheets \
    #     --dataset_path "./results/manifest_merged.csv" \
    #     --projection "{'channels': ['membrane', 'structure', 'dna'],
    # 'masks':
    # {'membrane': 'membrane_segmentation', 'dna': 'nucleus_segmentation'},
    # 'axis': 'Y', 'method': 'max'}" \
    #     --path_3d_column "CellImage3DPath" \
    #     --chosen_projection "Chosen2DProjectionPath" \
    #     --chosen_class "ChosenMitoticClass" \
    #     --label "Draft mitotic state resolved"

    fire.Fire(diagnostic_sheets)
