import numpy as np
import pandas as pd

from ..library.data import DataframeDataset, one_hot_encoding, append_one_hot

import pytest

@pytest.mark.parametrize(
    "data, expected",
    [
        ({"A": [1, 2, 3], "B": [10, 20, 30]}, (3.0, 3, 1.0))
    ]
)
def test_one_hot_encoder(data, expected):
    dataframe = pd.DataFrame(data=data)

    one_hot = one_hot_encoding(dataframe, "A")

    assert np.sum(one_hot) == expected[0]
    assert len(one_hot) == expected[1]

    dataframe, one_hot_len = append_one_hot(dataframe, "B", "A")

    assert np.sum(dataframe[0]) == expected[2]


def test_dataframe_dataset():
    data = {"A": [1, 2, 3], "B": [10, 20, 30]}
    dataframe = pd.DataFrame(data=data)

    loaders = {"A": lambda row: row["A"] + 5, "B": lambda row: row["B"] + 100}
    dataframe_dataset = DataframeDataset(dataframe, loaders)

    assert dataframe_dataset[0]["A"] == data["A"][0] + 5



# # The best practice would be to parametrize your tests, and include tests for any
# # exceptions that would occur
# @pytest.mark.parametrize(
#     "start_val, next_val, expected_values",
#     [
#         # (start_val, next_val, expected_values)
#         (5, 20, (20, 5)),
#         (10, 40, (40, 10)),
#         (1, 2, (2, 1)),
#         pytest.param(
#             "hello",
#             None,
#             None,
#             marks=pytest.mark.raises(
#                 exception=ValueError
#             ),  # Init value isn't an integer
#         ),
#         pytest.param(
#             1,
#             "hello",
#             None,
#             marks=pytest.mark.raises(
#                 exception=ValueError
#             ),  # Update value isn't an integer
#         ),
#     ],
# )
# def test_parameterized_value_change_with_exceptions(
#     start_val, next_val, expected_values
# ):
#     example = Example(start_val)
#     example.update_value(next_val)
#     assert expected_values == example.values



if __name__ == "__main__":
    test_one_hot_encoder()
    test_dataframe_dataset()
