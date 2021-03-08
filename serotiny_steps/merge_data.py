#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
import fire

from serotiny.csv import load_csv

###############################################################################

log = logging.getLogger(__name__)

###############################################################################


def merge_data(dataset_paths, manifest_path, required_fields=None, merge_datasets=None):
    """
    Load a list of dataset csv's, merge them, then write back out to csv.

    dataset_paths - list of paths to dataset csvs.
    manifest_path - where to write the result.
    required_fields - dictionary of dataset paths to a list
    of required fields for that dataset.
    merge_datasets - list of options to merge successive datasets
        into a single dataset. As this is
        defined by an operation between two datasets,
        there is always (number of datasets - 1)
        elements in this list.
    """

    if not required_fields:
        required_fields = {}

    datasets = [load_csv(path, required_fields.get(path, {})) for path in dataset_paths]

    manifest = datasets[0]

    if merge_datasets is not None:
        # merge operations join datasets
        assert len(merge_datasets) == len(dataset_paths) - 1

        for next_dataset, merge_keys in zip(datasets[1:], merge_datasets):
            existing_cols = set(manifest.columns.tolist())
            manifest = manifest.merge(
                next_dataset[
                    [
                        col
                        for col in next_dataset.columns
                        if (col not in existing_cols) or (col in merge_keys["on"])
                    ]
                ],
                **merge_keys
            )
    else:
        assert len(dataset_paths) == 1

    # Save manifest to CSV
    manifest.to_csv(manifest_path, index=False)


if __name__ == "__main__":
    # example command:
    # > python -m serotiny.steps.merge_data \
    #     --dataset_paths "['data/draft_plus_human_mito_annotations.csv',
    #                       'data/manifest.csv']" \
    #     --manifest_path "data/manifest_merged.csv" \
    #     --merge_datasets "[{'on': ['CellId', 'FOVId', 'CellIndex']}]"

    fire.Fire(merge_data)
