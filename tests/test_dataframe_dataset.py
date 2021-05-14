import pytest
import numpy as np
import pandas as pd

from serotiny.io.dataframe import DataframeDataset, one_hot_encoding, append_one_hot


@pytest.mark.parametrize(
    "data, expected", [({"A": [1, 2, 3], "B": [10, 20, 30]}, (3.0, 3, 1.0))]
)
def test_one_hot_encoder(data, expected):
    """
    Test `append_one_hot` function
    """
    dataframe = pd.DataFrame(data=data)

    one_hot = one_hot_encoding(dataframe, "A")

    assert np.sum(one_hot) == expected[0]
    assert len(one_hot) == expected[1]

    dataframe, one_hot_len = append_one_hot(dataframe, "B", "A")

    assert np.sum(dataframe[0]) == expected[2]
    assert one_hot_len == expected[1]


def test_dataframe_dataset():
    """
    Test `DataframeDataset` class
    """
    data = {"A": [1, 2, 3], "B": [10, 20, 30]}
    dataframe = pd.DataFrame(data=data)

    loaders = {"A": lambda row: row["A"] + 5, "B": lambda row: row["B"] + 100}
    dataframe_dataset = DataframeDataset(dataframe, loaders)

    assert dataframe_dataset[0]["A"] == data["A"][0] + 5


if __name__ == "__main__":
    test_one_hot_encoder()

    test_dataframe_dataset()
