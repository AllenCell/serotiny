import re
from itertools import chain
from upath import UPath as Path

import numpy as np
import pandas as pd
from omegaconf import OmegaConf, ListConfig

from monai.data import Dataset, PersistentDataset
from monai.transforms import Compose
from serotiny.dataframe import read_dataframe


def get_canonical_split_name(split):
    for canon in ("train", "val", "test", "predict"):
        if split.startswith(canon) or canon.startswith(split):
            return canon
    raise ValueError


def get_dataset(dataframe, transform, cache_dir=None):
    data = list(dataframe.to_dict("records"))
    if cache_dir is not None:
        return PersistentDataset(data, transform=transform, cache_dir=cache_dir)
    return Dataset(data, transform=transform)


def make_single_dataframe_splits(
    dataframe_path,
    transforms,
    split_column,
    columns=None,
    just_inference=False,
    split_map=None,
    cache_dir=None,
):
    dataframe = read_dataframe(dataframe_path, columns)
    dataframe[split_column] = dataframe[split_column].astype(np.dtype("O"))

    if not just_inference:
        assert dataframe.dtypes[split_column] == np.dtype("O")

    if split_map is not None:
        dataframe[split_column] = dataframe[split_column].replace(split_map)

    split_names = dataframe[split_column].unique().tolist()
    if not just_inference:
        assert set(split_names).issubset(
            {"train", "training", "valid", "val", "validation", "test", "testing"}
        )

    if split_column != "split":
        dataframe["split"] = dataframe[split_column].apply(get_canonical_split_name)

    datasets = {}
    if not just_inference:
        for split in ("train", "val", "test"):
            if cache_dir is not None:
                _split_cache = Path(cache_dir) / split
                _split_cache.mkdir(exist_ok=True, parents=True)
            else:
                _split_cache = None
            datasets[split] = get_dataset(
                dataframe.loc[dataframe["split"].str.startswith(split)],
                transforms[split],
                _split_cache,
            )

    datasets["predict"] = get_dataset(dataframe, transform=transforms["predict"])
    return datasets


def make_multiple_dataframe_splits(
    split_path, transforms, columns=None, just_inference=False, cache_dir=None
):
    split_path = Path(split_path)
    datasets = {}
    predict_df = []

    for fpath in chain(split_path.glob("*.csv"), split_path.glob("*.parquet")):
        split = re.findall(r"(.*)\.(?:csv|parquet)", fpath.name)[0]
        split = get_canonical_split_name(split)
        dataframe = read_dataframe(fpath, required_columns=columns)
        dataframe["split"] = split

        if cache_dir is not None:
            _split_cache = Path(cache_dir) / split
            _split_cache.mkdir(exist_ok=True, parents=True)
        else:
            _split_cache = None

        if not just_inference:
            datasets[split] = get_dataset(dataframe, transforms[split], _split_cache)
        predict_df.append(dataframe.copy())

    predict_df = pd.concat(predict_df)
    datasets["predict"] = get_dataset(predict_df, transform=transforms["predict"])

    return datasets


def _dict_depth(d):
    return (
        max((_dict_depth(v) if OmegaConf.is_config(v) else 0) for v in d.values()) + 1
    )


def parse_transforms(transforms):
    depth = _dict_depth(transforms)
    if depth == 1:
        transforms = {
            split: transforms for split in ["train", "val", "test", "predict"]
        }
    elif depth != 2:
        raise ValueError(f"Transforms dict should have depth 1 or 2. Got {depth}")

    for k, v in transforms.items():
        transforms[get_canonical_split_name(k)] = v

    for k in transforms:
        if isinstance(transforms[k], str):
            assert transforms[k] in transforms
            transforms[k] = transforms[transforms[k]]

    for split in ("train", "val", "test"):
        if split not in transforms:
            raise ValueError(f"'{split}' missing from transforms dict.")

    if "predict" not in transforms:
        transforms["predict"] = transforms["test"]

    for k in transforms:
        if isinstance(transforms[k], (list, ListConfig)):
            transforms[k] = Compose(transforms[k])

    return transforms
