#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import fire
import logging

from pathlib import Path
from typing import Dict, List, Optional, Union
from sklearn.model_selection import train_test_split

import pandas as pd

from ..library.csv import load_csv
from ..library.data import append_one_hot

###############################################################################

log = logging.getLogger(__name__)

###############################################################################


def split_data(
        dataset_path: str,
        output_path: str,
        class_column: str,
        id_column,
        ratios=None,
        required_fields=None):

    """
    """

    if required_fields is None:
        required_fields = {}

    dataset = load_csv(dataset_path, required_fields)
    dataset.dropna(inplace=True)
    dataset, one_hot_len = append_one_hot(
        dataset, class_column, id_column)

    if ratios is None:
        ratios = {
            'train': 0.6,
            'test': 0.2,
            'valid': 0.2}

    indexes = {}
    remaining_portion = 1.0
    remaining_data = dataset.index

    for key, portion in list(ratios.items())[:-1]:
        this_portion = portion / remaining_portion
        remaining_portion -= portion
        remaining_data, indexes[key] = train_test_split(
            dataset.loc[remaining_data, :].index, test_size=this_portion)

    indexes[list(ratios.keys())[-1]] = remaining_data

    # # split dataset into train, test and validtion subsets
    # # TODO: make split ratio a parameter (currently 0.2)
    # indexes = {}
    # index_train_valid, indexes["test"] = train_test_split(
    #     dataset.index, test_size=0.2
    # )
    # indexes["train"], indexes["valid"] = train_test_split(
    #     dataset.loc[index_train_valid, :].index
    # )

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

    output = {
        "dataset_paths": dataset_paths,
        "one_hot_len": one_hot_len}

    print(output)
    return output


if __name__ == '__main__':
    # example command:
    # python -m serotiny.steps.split_data \
    #     --dataset_path "data/filtered.csv" \
    #     --output_path "data/splits/" \
    #     --class_column "ChosenMitoticClass" \
    #     --id_column "CellId" \
    #     --ratios "{'train': 0.7, 'test': 0.2, 'valid': 0.1}"

    fire.Fire(split_data)
