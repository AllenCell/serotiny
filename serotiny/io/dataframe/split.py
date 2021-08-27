import pandas as pd
from sklearn.model_selection import train_test_split

DEFAULT_RATIOS = {"train": 0.6, "test": 0.2, "valid": 0.2}


def split_dataset(dataset, ratios=None):
    if ratios is None:
        ratios = DEFAULT_RATIOS

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

    return datasets
