import logging
from typing import Optional, List
import pandas as pd

import fire

from serotiny.io.dataframe import load_csv

###############################################################################

log = logging.getLogger(__name__)

###############################################################################


def filter_data(
    dataset_path: str,
    output_path: str,
    filter_options: dict,
    seed: int,
    required_fields: Optional[List] = None,
):
    """
    Filter dataframe down to speicifed columns

    Parameters
    ------------
    dataset_path: str
        Path to dataframe

    output_path: str
        Output path to save filtered dataset

    filter_options: dict
        Dict containing some filtering options

    seed: int
        Example: 42

    required_fields:
        List of required fields in the input dataset
    """
    if required_fields is None:
        required_fields = {}

    dataset = load_csv(dataset_path, required_fields.get(dataset_path, {}))

    # trick to shuffle dataset rows
    dataset = dataset.sample(frac=1, random_state=seed)

    output = dataset
    operation = filter_options.get("operation", "values")

    if operation == "values":
        column = filter_options["column"]
        output = dataset[dataset[column].isin(filter_options["values"])]

    elif operation == "choose_n_each":
        column = filter_options["column"]
        number = filter_options.get("number", 1)
        index = filter_options.get("index", "id")

        dataset.set_index(index)
        values = dataset[column].unique()

        subsets = [dataset[dataset[column] == value].head(number) for value in values]

        output = pd.concat(subsets)
    elif operation == "sample":
        column = filter_options["column"]
        number = filter_options.get("number", 1)
        index = filter_options.get("index", "id")

        dataset.set_index(index)
        values = dataset[column].dropna().unique()

        subsets = []
        for value in values:
            class_rows = dataset[dataset[column] == value]
            subsets.append(
                class_rows.sample(
                    number,
                    random_state=seed,
                    # only sample with replacement if there
                    # aren't enough data points in this class
                    replace=(len(class_rows) < number),
                )
            )

        output = pd.concat(subsets)

    output.to_csv(output_path, index=False)


if __name__ == "__main__":
    fire.Fire(filter_data)
