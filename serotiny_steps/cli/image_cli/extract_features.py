from pathlib import Path
from typing import Union, Dict
from functools import partial
import json
import pandas as pd
import numpy as np
import multiprocessing_on_dill as mp
from tqdm import tqdm



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


def _extract_features(input_path, features_to_extract):
    # imports here to optimize CLIs / Fire usage
    from serotiny.io.image import image_loader
    from serotiny.utils import load_config
    img, channel_map = image_loader(input_path, return_channels=True)
    results = dict()

    for feature, config in features_to_extract.items():
        feature_extractor = load_config(config)
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
):
    features = _extract_features(row[path_col], features_to_extract)
    features[index_col] = row[index_col]
    return features


def extract_features(
    input_path: Union[str, Path],
    features_to_extract: Dict,
):
    input_path = Path(input_path)

    features = _extract_features(input_path, features_to_extract)

    return json.dumps(features, cls=NumpyJSONEncoder)


def extract_features_batch(
    input_manifest: Union[pd.DataFrame, Union[str, Path]],
    output_path: Union[str, Path],
    path_col: str,
    index_col: str,
    features_to_extract: Dict,
    return_merged: bool = True,
    n_workers: int = 1,
    verbose: bool = False,
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

    path_col: str
        Column that contains the paths to the input images

    index_col: str
        Column to serve as index. Included in the output manifest
        (useful for merging)

    features_to_extract: Dict
        Config dictionary specifying what features to extract

    return_merged: bool = True
        If True, return the result manifest merged with the input.
        Otherwise, return only the result.

    n_workers: int = 1
        Number of multiprocessing workers to use for parallel computation

    verbose: bool = False
        Flag to tell whether to produce command line output
    """

    if not isinstance(input_manifest, pd.DataFrame):
        # import here to optimize CLIs / Fire usage
        from serotiny.io.dataframe import read_dataframe
        input_manifest = read_dataframe(input_manifest)

    output_path = Path(output_path)
    if not output_path.parent.exists():
        output_path.parent.mkdir(parents=True)

    # make helper function with pre attributed params except df row
    extract_features_ = partial(
        _extract_from_row,
        path_col=path_col,
        index_col=index_col,
        features_to_extract=features_to_extract,
    )
    iter_rows = (row for _, row in input_manifest.iterrows())

    if n_workers > 1:
        with mp.Pool(n_workers) as pool:
            jobs = pool.imap_unordered(extract_features_, iter_rows)
            if verbose:
                jobs = tqdm(jobs, total=len(input_manifest))
            success = pd.DataFrame.from_records(list(jobs))
    else:
        if verbose:
            iter_rows = tqdm(iter_rows, total=len(input_manifest))
        success = pd.DataFrame.from_records(
            [extract_features_(row) for row in iter_rows]
        )

    if return_merged:
        success = success.merge(input_manifest, on=index_col)

    success.to_csv(output_path)
