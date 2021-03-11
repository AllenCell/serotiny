from typing import Sequence, Union
from pathlib import Path
import pandas as pd

from actk.utils import dataset_utils


def load_csv(dataset: Union[str, Path, pd.DataFrame], required: Sequence[str]):
    """
    Read dataframe from either a path or an existing pd.DataFrame, checking
    the fields given by `required` are present
    """

    # Handle dataset provided as string or path
    if isinstance(dataset, (str, Path)):
        dataset = Path(dataset).expanduser().resolve(strict=True)

        # Read dataset
        dataset = pd.read_csv(dataset)

    # Check the dataset for the required columns
    dataset_utils.check_required_fields(
        dataset=dataset,
        required_fields=required,
    )

    return dataset
