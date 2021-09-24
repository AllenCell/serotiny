from typing import Sequence, Dict
from pathlib import Path

from inspect import Parameter, signature
from makefun import wraps

import logging
import joblib

import pandas as pd
import serotiny.data.dataframe.transforms as df_transforms
from serotiny.io.dataframe import read_dataframe

logger = logging.getLogger(__name__)


class TransformDataframeCLI:
    def __init__(self, suffix="", output_path=None):
        self._output_path = output_path
        self._suffix = suffix
        self._result = None

        for transform in [
            df_transforms.split_dataframe,
            df_transforms.filter_rows,
            df_transforms.filter_columns,
            df_transforms.sample_n_each,
            df_transforms.append_one_hot,
            df_transforms.append_labels_to_integers,
            df_transforms.append_class_weights,
        ]:
            self._add_transform(transform)

    def _add_transform(self, func):
        @wraps(func)
        def wrapper(**kwargs):
            if self._result is not None:
                kwargs["dataframe"] = self._result
            else:
                kwargs["dataframe"] = read_dataframe(kwargs["dataframe"])

            import pdb; pdb.set_trace()
            self._result = func(**kwargs)
            return self

        setattr(self, func.__name__, wrapper)

    def __str__(self):
        """
        Hijack __str__ to store the final result of a pipeline,
        stored in self._result
        """

        if self._result is None:
            raise ValueError("No result stored on pipeline")

        if self._output_path is None:
            raise ValueError("No output directory")

        self._output_path = Path(self._output_path)
        if isinstance(self._result, pd.DataFrame):
            if self._output_path.suffix == "":
                self._output_path = self._output_path.with_suffix(self._suffix)

            if self._output_path.suffix == ".parquet":
                self._result.to_parquet(self._output_path)
            else:
                if self._output_path.suffix != ".csv":
                    logger.warning(
                        f"Unrecognized suffix {self._output_path.suffix}. "
                        f"Assuming .csv"
                    )

                self._result.to_csv(self._output_path)

        elif isinstance(self._result, Dict[str, pd.DataFrame]):
            for name, df in self._result.items():
                name = name + self._suffix
                name = Path(name)
                if name.suffix == ".csv":
                    df.to_csv(self._output_path / name)
                elif name.suffix == ".parquet":
                    df.to_parquet(self._output_path / name)
                else:
                    raise ValueError(f"Unexpected suffix: '{name.suffix}'")
        else:
            if self._output_path.is_dir():
                self._output_path = self._output_path / "result.joblib"
            else:
                if self._output_path.suffix == "":
                    self._output_path = self._output_path.with_suffix(self._suffix)

        return ""
