import os
from pathlib import Path
import fire

import pandas as pd


def reset_index(dataset, index):
    pd.DataFrame(dataset.loc[index, :].reset_index(drop=True))


def write_csvs(datasets, output_path):
    for key, dataset in datasets.items():
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        save_path = Path(output_path) / f"{key}.csv"
        dataset.to_csv(save_path, index=False)


def partition_data(
    dataset_path: str,
    output_path: str,
    partition_size=100,
    partition_prefix="partition",
    required_fields=None,
):
    """
    Split the incoming data into N sets of output data, where
    each set has `partition_size` elements, generating filenames
    based on `partition_prefix`.
    """

    # import here to optimize CLIs / Fire usage
    from serotiny.io.dataframe import read_dataframe

    if required_fields is None:
        required_fields = {}

    dataset = read_dataframe(dataset_path, required_fields)
    partition_count = len(dataset) // partition_size
    remaining = len(dataset) % partition_size
    if remaining > 0:
        partition_count += 1

    cursor = 0
    partitions = {}
    for partition_index in range(partition_count):
        seek = cursor + partition_size
        if seek > len(dataset):
            seek = cursor + remaining
        partition_rows = dataset[cursor:seek]
        partition_key = f"{partition_prefix}_{partition_index}"
        partitions[partition_key] = partition_rows
        cursor += partition_size

    write_csvs(partitions, output_path)


if __name__ == "__main__":
    # example command:
    # python -m serotiny_steps.partition_data \
    #     --dataset_path "data/filtered.csv" \
    #     --output_path "data/partitions/" \
    #     --partition_size 50

    fire.Fire(partition_data)
