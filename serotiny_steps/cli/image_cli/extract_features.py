import os
from pathlib import Path
from typing import Union, Dict, Optional, Sequence
from functools import partial
import json
import pandas as pd
import numpy as np


def _do_imports():
    global image_loader
    global load_config, load_multiple
    global _parse_batch

    from serotiny.io.image import image_loader
    from serotiny.utils import load_config, load_multiple
    from .batch_work import _parse_batch


class NumpyJSONEncoder(json.JSONEncoder):
    """
    Helper class to enable serializing numpy objects
    """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NumpyJSONEncoder, self).default(obj)


def _extract_features(input_path, features_to_extract, reader=None):
    img, channel_map = image_loader(input_path, return_channels=True, reader=reader)
    results = dict()

    for feature, feature_extractor in features_to_extract.items():
        result = feature_extractor(img, channel_map)
        if isinstance(result, dict):
            results.update(result)
        else:
            results[feature] = result

    return results


def _extract_from_row(
    row: Union[pd.Series, Dict],
    path_col: str,
    index_col: str,
    features_to_extract: Dict,
    reader=None,
):
    features = _extract_features(row[path_col], features_to_extract, reader=reader)
    features[index_col] = row[index_col]
    return features


def extract_features(
    input_path: Union[str, Path],
    features_to_extract: Dict,
    reader=None,
):
    input_path = Path(input_path)

    features = _extract_features(input_path, features_to_extract, reader=reader)

    return json.dumps(features, cls=NumpyJSONEncoder)


def extract_features_batch(
    input_manifest: Union[pd.DataFrame, Union[str, Path]],
    output_path: Union[str, Path],
    features_to_extract: Dict,
    path_col: str,
    index_col: str,
    include_cols: Sequence[str] = [],
    image_reader: Optional[str] = None,
    return_merged: bool = True,
    backend: str = "multiprocessing",
    write_every_n_rows: int = 100,
    n_workers: int = 1,
    verbose: bool = False,
    skip_if_exists: bool = False,
    debug: bool = False,
):
    """
    Extract features from an image in a dataframe row, using extractors given by
    config in `features_to_extract`

    Parameters
    ----------
    input_manifest: Union[pd.DataFrame, Union[str, Path]]
        Path to the input manifest, or a pd.DataFrame

    output_path: Union[str, Path]
        Path where the output dataframe will be stored

    features_to_extract: Dict
        Config dictionary specifying what features to extract

    path_col: str
        Column that contains the paths to the input images

    index_col: str
        Column to serve as index. Included in the output manifest
        (useful for merging)

    include_cols: Sequence[str] = []
        List of columns to include in the output manifest, aside from the
        column containing paths to the output images

    image_reader: Optional[str] = None
        aicsimageio reader to use. If None, it is automatically determined,
        but it entails additional io which makes things slower

    return_merged: bool = True
        If True, return the result manifest merged with the input.
        Otherwise, return only the result.

    backend: str = "multiprocessing"
        Backend to use for parallel computation. Possible values are
        "multiprocessing", "slurm"

    write_every_n_rows: int = 100
        Write results to temporary file every n rows

    n_workers: int = 1
        Number of parallel workers to use for parallel computation

    verbose: bool = False
        Flag to tell whether to produce command line output

    skip_if_exists: bool = False
        Whether to skip images if they're found on disk. Useful to avoid
        redoing computations when something fails

    debug: bool = False
        Whether to return errors in csv instead of raising and breaking

    """

    _do_imports()

    # make helper function with pre attributed params except df row
    extract_features_ = partial(
        _extract_from_row,
        path_col=path_col,
        index_col=index_col,
        features_to_extract=load_multiple(features_to_extract),
        reader=image_reader,
    )

    _parse_batch(
        input_manifest=input_manifest,
        func_per_row=extract_features_,
        output_path=output_path,
        index_col=index_col,
        include_cols=include_cols,
        write_every_n_rows=write_every_n_rows,
        return_merged=return_merged,
        backend=backend,
        n_workers=n_workers,
        verbose=verbose,
        skip_if_exists=skip_if_exists,
        debug=debug,
    )
