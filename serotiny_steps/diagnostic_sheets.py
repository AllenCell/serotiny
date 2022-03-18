from pathlib import Path
from typing import Optional

import fire
import pandas as pd

from actk.steps.diagnostic_sheets import DiagnosticSheets
from .apply_projection import apply_projection


def diagnostic_sheets(
    dataset_path: str,
    output_path: str,
    projection: dict,
    path_3d_column: str,
    chosen_projection: str,
    metadata: str,
    proj_all: bool,
    num_images: int,
    executor_address: Optional[str] = None,
    source_path: Optional[str] = None,
    overwrite: bool = False,
):
    """
    Call actk.steps.diagnostic_sheets to generate diagnostic sheets
    for a dataset of 3D images. Calls apply_projection

    Parameters
    -----------
    dataset_path: str
        Path to input dataset containing 3D images

    output_path: str
        Path to save diagnostic sheet manifest

    projection: dict
        Config containing projection details

    path_3d_column: str
        Column name containing 3D image paths
        Example: "CellImage3DPath"

    chosen_projection: str
        Column name to append projected image paths to
        original dataset
        Example: "Chosen2DProjectionPath"

    metadata: str
        Column containing class label information
        Example: "ChosenMitoticClass"

    proj_all: bool
        Whether to do all axis max projection or not

    num_images: int
        Number of images to sample to make diagnostic sheet
        Example: 200

    executor_address: Optional[str] = None
        Address for dask distributed compute

    source_path: Optional[str] = None
        If source path, append source path to
        3D image paths

    """

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

    fire.Fire(diagnostic_sheets)
