#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

import logging
import fire

from ..library.csv import load_csv

###############################################################################

log = logging.getLogger(__name__)

###############################################################################


def filter_data(
        dataset_path,
        output_path,
        filter_options,
        required_fields=None
):
    if required_fields is None:
        required_fields = {}

    dataset = load_csv(
        dataset_path,
        required_fields.get(dataset_path, {}))

    output = dataset
    operation = filter_options.get('operation', 'values')

    if operation == 'values':
        column = filter_options['column']
        output = dataset[dataset[column].isin(filter_options['values'])]

    elif operation == 'choose_n_each':
        column = filter_options['column']
        number = filter_options.get('number', 1)
        index = filter_options.get('index', 'id')

        dataset.set_index(index)
        values = dataset[column].unique()

        subsets = [
            dataset[dataset[column] == value].head(number)
            for value in values]

        output = pd.concat(subsets)

    output.to_csv(output_path, index=False)


if __name__ == '__main__':
    fire.Fire(filter_data)