#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import logging
from pathlib import Path
import fire

from sklearn.model_selection import train_test_split

import pandas as pd

from serotiny.csv import load_csv
from serotiny.data import append_one_hot

###############################################################################

log = logging.getLogger(__name__)

###############################################################################


def split_data(dataset_path: str, output_path: str, ratios=None, required_fields=None):

    """
    Split the incoming data into N sets of output data, randomly
    sampled according to the provided ratios. The output files will
    be named after the keys for each ratio.
    """

    if required_fields is None:
        required_fields = {}

    dataset = load_csv(dataset_path, required_fields)

    if ratios is None:
        ratios = {"train": 0.6, "test": 0.2, "valid": 0.2}

    indexes = {}
    remaining_portion = 1.0
    remaining_data = dataset.index

    for key, portion in list(ratios.items())[:-1]:
        this_portion = portion / remaining_portion
        remaining_portion -= portion
        remaining_data, indexes[key] = train_test_split(
            dataset.loc[remaining_data, :].index, test_size=this_portion
        )

    indexes[list(ratios.keys())[-1]] = remaining_data

    # index split datasets
    datasets = {
        key: pd.DataFrame(dataset.loc[index, :].reset_index(drop=True))
        for key, index in indexes.items()
    }

    # save a dataloader for each dataset
    dataset_paths = {}
    for split, dataset in datasets.items():
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        save_path = Path(output_path) / f"{split}.csv"
        dataset.to_csv(save_path, index=False)
        dataset_paths[split] = str(save_path)


if __name__ == "__main__":
    # example command:
    # python -m serotiny.steps.split_data \
    #     --dataset_path "data/filtered.csv" \
    #     --output_path "data/splits/" \
    #     --class_column "ChosenMitoticClass" \
    #     --id_column "CellId" \
    #     --ratios "{'train': 0.7, 'test': 0.2, 'valid': 0.1}"

    fire.Fire(split_data)
