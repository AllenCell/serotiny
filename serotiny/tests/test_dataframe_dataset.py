import numpy as np
import pandas as pd

from ..library.data import DataframeDataset, one_hot_encoding, append_one_hot


def test_one_hot_encoder():
    data = {"A": [1, 2, 3], "B": [10, 20, 30]}
    dataframe = pd.DataFrame(data=data)

    one_hot = one_hot_encoding(dataframe, "A")

    assert np.sum(one_hot) == 3.0
    assert len(one_hot) == 3

    dataframe, one_hot_len = append_one_hot(dataframe, "B", "A")

    assert np.sum(dataframe[0]) == 1.0


def test_dataframe_dataset():
    data = {"A": [1, 2, 3], "B": [10, 20, 30]}
    dataframe = pd.DataFrame(data=data)

    loaders = {"A": lambda row: row["A"] + 5, "B": lambda row: row["B"] + 100}
    dataframe_dataset = DataframeDataset(dataframe, loaders)

    assert dataframe_dataset[0]["A"] == data["A"][0] + 5


if __name__ == "__main__":
    test_one_hot_encoder()
    test_dataframe_dataset()
