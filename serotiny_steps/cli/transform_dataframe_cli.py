from typing import Union, Dict, Sequence, Optional, T
from pathlib import Path
from inspect import Parameter, signature
from makefun import wraps

import logging

import pandas as pd
import serotiny.data.dataframe.transforms as df_transforms
from serotiny.io.dataframe import read_dataframe
from ._pipeline_cli import PipelineCLI
from .image_cli.apply_transforms import transform_images
from .omegaconf_decorator import omegaconf_decorator

PathLike = Union[str, Path]
OneOrMany = Union[T, Sequence[T]]


logger = logging.getLogger(__name__)

def _store_one_df(result, output_path):
    if output_path.suffix == ".parquet":
        result.to_parquet(output_path)
    elif output_path.suffix == ".csv":
        result.to_csv(output_path)
    else:
        raise TypeError("output path must be either .parquet or .csv")


def _store_many_dfs(result, output_path):
    for name, df in result.items():
        assert isinstance(name, str)
        assert isinstance(df, pd.DataFrame)

        name = Path(name)
        if name.suffix == ".csv":
            df.to_csv(output_path / name)
        elif name.suffix == ".parquet":
            df.to_parquet(output_path / name)
        elif name.suffix == "":
            df.to_csv(output_path / name.with_suffix(".csv"))
        else:
            raise TypeError(f"Unexpected suffix: '{name.suffix}'")


def _read_one_or_many(
    input_manifests: OneOrMany[PathLike],
    merge_col: Optional[str] = None
):
    input_manifest = None

    if isinstance(input_manifests, (str, Path)):
        input_manifests = [input_manifests]

    for manifest in input_manifests:
        manifest = Path(manifest)
        if not manifest.exists():
            raise FileNotFoundError(f"Given input path {manifest} does not exist")
        manifest = read_dataframe(manifest)

        if input_manifest is None:
            input_manifest = manifest
            continue

        if merge_col is not None:
            input_manifest = input_manifest.merge(manifest, on=merge_col, how="outer")
        else:
            input_manifest = input_manifest.concat(manifest)

    return input_manifest


def _dataframe_from_disk(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        func_sig = signature(func)
        func_arg_names = list(func_sig.parameters.keys())

        base_args = {}
        for arg_ix, arg_value in enumerate(args):
            base_args[func_arg_names[arg_ix]] = arg_value

        base_args.update(kwargs)

        if "dataframe" in base_args:
            if not isinstance(base_args["dataframe"], pd.DataFrame):
                if base_args["dataframe"] not in (..., "..."):
                    base_args["dataframe"] = read_dataframe(base_args["dataframe"])

        return func(**base_args)

    return wrapper


class TransformDataframeCLI(PipelineCLI):
    def __init__(self, input_manifests=None, output_path=None):
        super().__init__(
            output_path=output_path,
            transforms=[
                _dataframe_from_disk(df_transforms.split_dataframe),
                _dataframe_from_disk(df_transforms.filter_rows),
                _dataframe_from_disk(df_transforms.filter_columns),
                _dataframe_from_disk(df_transforms.sample_n_each),
                _dataframe_from_disk(df_transforms.append_one_hot),
                _dataframe_from_disk(df_transforms.append_labels_to_integers),
                _dataframe_from_disk(df_transforms.append_class_weights),
                omegaconf_decorator(transform_images, "transforms_to_apply")
            ],
            store_methods={
                pd.DataFrame: _store_one_df,
                dict: _store_many_dfs,
            }
        )

        self._result = (None if input_manifests is None else
                        _read_one_or_many(input_manifests))
