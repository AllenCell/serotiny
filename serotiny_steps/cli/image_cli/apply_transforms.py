from typing import Union, Dict, Sequence, Optional
from pathlib import Path

from collections import defaultdict
from functools import partial
import json

from copy import deepcopy
import pandas as pd
from omegaconf.dictconfig import DictConfig
from omegaconf.listconfig import ListConfig


def _do_imports():
    # import here to optimize CLIs / Fire usage
    global torch
    global image_loader
    global tiff_writer
    global load_config
    global _parse_batch

    import torch

    from serotiny.io.image import image_loader, tiff_writer
    from serotiny.utils import load_config
    from .batch_work import _parse_batch


def _unpack_image_channels(img):
    if isinstance(img, list):
        return img
    return [img[ch] for ch in range(img.shape[0])]


def _preload_transforms(transforms_to_apply):
    transforms_to_apply = deepcopy(transforms_to_apply)
    _new_transforms = defaultdict(dict)
    for name, transform in transforms_to_apply.items():
        _new_transforms[name]["input"] = transform["input"]

        if isinstance(transform["steps"], (list, ListConfig)):
            steps_configs = transform["steps"]
        elif isinstance(transform["steps"], (dict, DictConfig)):
            steps_configs = transform["steps"].values()
        else:
            raise TypeError(f"Unexpected type for `steps` field: "
                            f"{type(transform['steps'])}")

        _steps = []
        for step in steps_configs:

            # if an "^individual_args" key is present, as that means
            # there are values we need to retrieve from each row
            # as additional arguments to the transform
            individual_args = step.pop("^individual_args", {})

            # in case this step is a class, the individual_args
            # need to be used in its __init__, so we turn it
            # into a ^bind, and signal that it must be inited
            # before it is called
            must_init = False
            if len(individual_args) > 0:
                if "^init" in step:
                    step["^bind"] = step["^init"]
                    del step["^init"]
                    must_init = True
                for arg in individual_args:
                    if arg in step:
                        del step[arg]


            # retrieve another special key "^unpack", if it's present.
            # its use is explained below
            unpack = step.pop("^unpack", False)


            # load the step with our dynamic import functionality
            step = load_config(step)
            _steps.append((step, individual_args, unpack, must_init))
        _new_transforms[name]["steps"] = _steps
    return _new_transforms


def apply_transforms(
    row: Union[Dict, pd.Series],
    transforms_to_apply,
    reader=None
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


    transforms_to_apply = deepcopy(transforms_to_apply)
    result_imgs = dict()

    # the `transforms_to_apply` dict contains a field "transforms",
    # which is the sequence of transforms needed to obtain the output image.
    # each element of "transforms" is a dict containing instructions for
    # a sequence of steps

    # `name` will be used to refer to the result of this transform,
    # when it's stored in the `result_imgs` dict.
    for name, transform in transforms_to_apply.items():
        # a step may have as starting point either a file on disk,
        # or the output of a previous step. if a step's "input" field
        # should be a dictionary, where each key is a string - indicating
        # either a previous step's result or a column in the dataframe -
        # and each value is the list of channels to use for that input.
        # if an input is being read from disk, channel labels can be used.
        # if an input is being read as the result of a previous step, channel
        # indices must be used
        if isinstance(transform["input"], str):
            if transform["input"] in result_imgs:
                imgs = [result_imgs[transform["input"]]]
            else:
                img = image_loader(row[transform["input"]], output_dtype="float32", reader=reader)
                imgs = [img]
        elif isinstance(transform["input"], (dict, DictConfig)):
            imgs = []
            for input_img, channels in transform["input"].items():
                if input_img in result_imgs:
                    _img = result_imgs[input_img]
                elif input_img in row:
                    _img, channel_map = image_loader(row[input_img],
                                                     output_dtype="float32",
                                                     return_channels=True)
                else:
                    raise ValueError(f"Given input not found: {input_img}")

                if channels is None:
                    imgs += [_img]
                else:
                    channels_type = set(map(type, channels)).pop()
                    if channels_type == int:
                        channel_map = list(range(_img.shape[0]))
                    imgs += [_img[[channel_map[ch] for ch in channels]]]
        else:
            raise TypeError(f"Unexpected type for `input`: {transform['input']}")

        # after the starting point for this transform has been loaded,
        # we iterate over the steps listed in this transform's
        # "steps" field. each element of "steps" is a dict
        # specifying a specific transform. these dicts are parsed
        # using our dynamic_import utils. at this point, the transforms
        # have been preloaded in the main process and passed to each worker

        for (step, individual_args, unpack, must_init) in transform["steps"]:
            kwargs = dict()
            for arg, column in individual_args.items():
                try:
                    kwargs[arg] = json.loads(row[column])
                except json.JSONDecodeError:
                    kwargs[arg] = row[column]

            if must_init:
                # if we're here, it means this step is a class that
                # requires individual_args to be inited, and so it
                # must be inited here before it's called ahead
                step = step(**kwargs)
                kwargs = dict()

            # because of the way we collect the inputs,
            # `imgs` can be a list with a single image, if this is
            # the first step in this transform
            if isinstance(imgs, list) and len(imgs) == 1:
                imgs = step(imgs[0], **kwargs)
            else:
                # if the input `imgs` at this point consists of
                # more than a single image, there is an additional
                # argument "^unpack" that can be specified for this
                # step, which tells us to use the * operator here
                imgs = step(*imgs, **kwargs) if unpack else step(imgs, **kwargs)

        result_imgs[name] = imgs

    return result_imgs


def _transform_from_row(
    row: Union[pd.Series, Dict],
    transforms_to_apply: Dict,
    outputs: Dict[str, Sequence[str]],
    output_path: Union[str, Path],
    index_col: str,
    reader: Optional[str] = None,
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

    outputs: Dict[str, Sequence[str]]
        Dictionary containing the keys for each output to save to disk,
        and the corresponding channel names for that output. The channel
        names are assumed to be in the correct order for each output

    output_path: Union[str, Path]
        Path to the folder where the outputs will be stored

    index_col: str
        Column to serve as index. Used for output filenames

    reader: Optional[str] = None
        aicsimageio reader to use. If None, it is automatically determined,
        but it entails additional io which makes things slower
    """

    result = dict()
    result_imgs = apply_transforms(row, transforms_to_apply, reader=reader)
    for output, channel_names in outputs.items():
        _output_path = output_path / f"{row[index_col]}_{output}.tiff"
        img = result_imgs[output]
        if isinstance(img, torch.Tensor):
            img = img.numpy()
        tiff_writer(img, _output_path, channel_names=channel_names)
        result[output] = str(_output_path)
    return result


def transform_images(
    input_manifest: Union[pd.DataFrame, Union[str, Path]],
    output_path: Union[str, Path],
    transforms_to_apply: Sequence,
    outputs: Dict[str, Sequence[str]],
    index_col: str,
    include_cols: Sequence[str] = [],
    image_reader: Optional[str] = None,
    backend: str = "multiprocessing",
    return_merged: bool = False,
    write_every_n_rows: int = 100,
    n_workers: int = 1,
    verbose: bool = False,
    skip_if_exists: bool = False,
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

    transforms_to_apply: Sequence
        Config dictionary specifying what transforms to apply

    outputs: Dict[str, Sequence[str]]
        Dictionary containing the keys for each output to save to disk,
        and the corresponding channel names for that output. The channel
        names are assumed to be in the correct order for each output

    index_col: str
        Column to serve as index. Used for output filenames

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
        Number of multiprocessing workers to use for parallel computation

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
    apply_transform = partial(
        _transform_from_row,
        transforms_to_apply=_preload_transforms(transforms_to_apply),
        outputs=outputs,
        output_path=Path(output_path),
        index_col=index_col,
        reader=image_reader,
    )

    _parse_batch(
        input_manifest=input_manifest,
        func_per_row=apply_transform,
        output_path=output_path,
        index_col=index_col,
        include_cols=include_cols,
        return_merged=return_merged,
        backend=backend,
        write_every_n_rows=write_every_n_rows,
        n_workers=n_workers,
        verbose=verbose,
        skip_if_exists=skip_if_exists,
        debug=debug,
    )
