from typing import Union, Dict, Sequence
from pathlib import Path
import traceback

from functools import partial
import json
import pandas as pd
import multiprocessing_on_dill as mp
from omegaconf.dictconfig import DictConfig

from tqdm import tqdm

from serotiny.io.image import tiff_writer, image_loader
from serotiny.io.dataframe import read_dataframe
from serotiny.utils import load_config


def _unpack_image_channels(img):
    if isinstance(img, list):
        return img
    return [img[ch] for ch in range(img.shape[0])]


def apply_transforms(
    row: Union[Dict, pd.Series],
    transforms_to_apply,
):
    """
    Apply transforms defined by `transforms_to_apply` to images in
    paths specified in columns of `row`

    Parameters
    ----------
    row: Union[Dict, pd.Series]
        The dataframe row (or equivalent dict or namedtuple) containing
        the necessary fields

    transforms_to_apply: Dict
        The dictionary specifying the transforms to apply
    """

    result_imgs = dict()

    # the `transforms_to_apply` dict contains a field "steps",
    # which is the list of steps needed to obtain the output image.
    # each element of "steps" is a dict containing instructions for
    # a sequence of transforms
    for step in transforms_to_apply:

        # `name` will be used to refer to the result of this step,
        # when it's stored in the `result_imgs` dict.
        name = step["name"]

        # a step may have as starting point either a file on disk,
        # or the output of a previous step. if a step's "input" field
        # should be a dictionary, where each key is a string - indicating
        # either a previous step's result or a column in the dataframe -
        # and each value is the list of channels to use for that input.
        # if an input is being read from disk, channel labels can be used.
        # if an input is being read as the result of a previous step, channel
        # indices must be used
        if isinstance(step["input"], str):
            if step["input"] in result_imgs:
                #imgs = _unpack_image_channels(result_imgs[step["input"]])
                imgs = [result_imgs[step["input"]]]
            else:
                img = image_loader(row[step["input"]], output_dtype="float32")
                #imgs = _unpack_image_channels(img)
                imgs = [img]
        elif isinstance(step["input"], (dict, DictConfig)):
            imgs = []
            for input_img, channels in step["input"].items():
                if input_img in result_imgs:
                    _img = result_imgs[input_img]
                elif input_img in row:
                    _img, channel_map = image_loader(row[input_img],
                                                     output_dtype="float32",
                                                     return_channels=True)
                else:
                    raise ValueError(f"Given input not found: {input_img}")

                if channels is None:
                    #imgs += _unpack_image_channels(_img)
                    imgs += [_img]
                else:
                    channels_type = set(map(type, channels)).pop()
                    if channels_type == int:
                        channel_map = list(range(_img.shape[0]))
                    imgs += [_img[[channel_map[ch] for ch in channels]]]
        else:
            raise TypeError(f"Unexpected type for `input`: {step['input']}")

        # after the starting point for this step has been loaded,
        # we iterate over the transforms listed in this step's
        # "transforms" field. each element of this list is a dict
        # specifying a specific transform. these dicts are parsed
        # using our dynamic_import utils
        transforms_configs = step["transforms"]
        for transform in transforms_configs:
            # if an "^individual_args" key is present, as that means
            # there are values we need to retrieve from this row
            # as additional arguments to the transform
            if "^individual_args" in transform:
                for arg, column in transform["^individual_args"].items():
                    try:
                        transform[arg] = json.loads(row[column])
                    except json.JSONDecodeError:
                        transform[arg] = row[column]

            # load the transform using our dynamic import logic
            transform = load_config(
                {k: v for k, v in transform.items() if k != "^individual_args"}
            )

            # the result of the transform is always stored in a list,
            # even if it's a single image
            if len(imgs) == 1:
                imgs = transform(imgs[0])
            else:
                # if the input `imgs` at this point consists of
                # more than a single image, there is an additional
                # argument "unpack" that can be specified for this
                # step, which tells us to use the * operator here
                unpack = step.get("unpack", False)
                imgs = transform(*imgs) if unpack else transform(imgs)

        result_imgs[name] = imgs

    # finally, the key of `result_imgs` which contains the output
    # image is the name of the last step in the list
    output_key = transforms_to_apply[-1]["name"]
    return result_imgs[output_key]


def _transform_from_row(
    row: Union[pd.Series, Dict],
    transforms_to_apply: Dict,
    output_channel_names: Sequence[str],
    output_path: Union[str, Path],
    index_col: str,
    include_cols: Sequence[str] = [],
    debug: bool = False,
):
    """
    Transform an image in a dataframe row, using transforms given by
    config in `transforms_to_apply`

    Parameters:
    ---
    row: Union[pd.Series, Dict]
        The DataFrame row from which the image path is taken

    transforms_to_apply: Dict
        Config dictionary specifying what transforms to apply

    output_path: Union[str, Path]
        Path to the folder where the outputs will be stored

    output_channel_names: Sequence[str]
        List of the names to be assigned to the channels in the output
        image (in order)

    index_col: str
        Column to serve as index. Used for output filenames

    include_cols: Sequence[str]
        List of columns to include in the output manifest, aside from the
        column containing paths to the output images

    debug: bool = False
        Whether to return errors in csv instead of raising and breaking
    """

    output_path = output_path / f"{row[index_col]}.tiff"

    result = {col: row[col] for col in include_cols}
    result[index_col] = row[index_col]
    result["img_path"] = output_path

    try:
        img = apply_transforms(row, transforms_to_apply)
        tiff_writer(img, output_path, channel_names=output_channel_names)
        result["errors"] = ""
    except Exception as e:
        if debug:
            result["errors"] = traceback.format_exc()
        else:
            raise e

    return result


def transform_images(
    input_manifest: Union[pd.DataFrame, Union[str, Path]],
    output_path: Union[str, Path],
    output_channel_names: Sequence[str],
    transforms_to_apply: Sequence,
    index_col: str,
    include_cols: Sequence[str] = [],
    n_workers: int = 1,
    verbose: bool = False,
    debug: bool = False,
):
    """
    Transform images given in a manifest, using transforms given by
    config in `transforms_to_apply`

    Parameters
    ----------
    input_manifest: Union[pd.DataFrame, Union[str, Path]]
        Path to the input manifest, or a pd.DataFrame

    output_path: Union[str, Path]
        Path to the folder where the outputs will be stored

    output_channel_names: Sequence[str]
        List of the names to be assigned to the channels in the output
        image (in order)

    transforms_to_apply: Sequence
        Config dictionary specifying what transforms to apply

    index_col: str
        Column to serve as index. Used for output filenames

    include_cols: Sequence[str] = []
        List of columns to include in the output manifest, aside from the
        column containing paths to the output images

    n_workers: int = 1
        Number of multiprocessing workers to use for parallel computation

    verbose: bool = False
        Flag to tell whether to produce command line output
    """

    if not isinstance(input_manifest, pd.DataFrame):
        input_manifest = read_dataframe(input_manifest)

    output_path = Path(output_path)
    if not output_path.exists():
        output_path.mkdir(parents=True)

    # make helper function with pre attributed params except df row
    apply_transform = partial(
        _transform_from_row,
        transforms_to_apply=transforms_to_apply,
        output_channel_names=output_channel_names,
        output_path=output_path,
        index_col=index_col,
        include_cols=include_cols,
        debug=debug
    )
    iter_rows = (row for _, row in input_manifest.iterrows())

    if n_workers > 1:
        with mp.Pool(n_workers) as pool:
            jobs = pool.imap_unordered(apply_transform, iter_rows)
            if verbose:
                jobs = tqdm(jobs, total=len(input_manifest))
            success = pd.DataFrame.from_records(list(jobs))
    else:
        if verbose:
            iter_rows = tqdm(iter_rows, total=len(input_manifest))
        success = pd.DataFrame.from_records([apply_transform(row) for row in iter_rows])

    success.to_csv(output_path / "manifest.csv")
