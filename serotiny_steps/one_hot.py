#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import logging
import fire

from serotiny.csv import load_csv
from serotiny.data import append_one_hot

###############################################################################

log = logging.getLogger(__name__)

###############################################################################


def one_hot(
    dataset_path: str,
    output_path: str,
    class_column: str,
    chosen_class: str,
    id_column,
    required_fields=None,
):

    """
    Choose column as class and assign one-hot encoding
    """

    if required_fields is None:
        required_fields = {}

    dataset = load_csv(dataset_path, required_fields)

    dataset[class_column] = dataset[chosen_class]
    dataset, one_hot_len = append_one_hot(dataset, class_column, id_column)

    dataset.to_csv(output_path, index=False)


if __name__ == "__main__":
    # example command:
    # python -m serotiny.steps.one_hot \
    #     --dataset_path "data/filtered.csv" \
    #     --output_path "data/splits/" \
    #     --class_column "ChosenMitoticClass" \
    #     --id_column "CellId" \
    #     --ratios "{'train': 0.7, 'test': 0.2, 'valid': 0.1}"

    fire.Fire(one_hot)
