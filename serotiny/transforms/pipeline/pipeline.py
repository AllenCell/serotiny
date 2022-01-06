from typing import Union, Dict, Sequence, Optional
from pathlib import Path

import collections
from collections import defaultdict
from functools import partial
import json

from copy import deepcopy
import numpy as np
import pandas as pd
import torch
from omegaconf.dictconfig import DictConfig
from omegaconf.listconfig import ListConfig

from serotiny.io.image import image_loader, tiff_writer
from serotiny.imports import load_config
from ._backend import _backend


def _flatten(d, parent_key='', sep='.'):
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if len(parent_key) else str(k)
        if isinstance(v, collections.abc.MutableMapping):
            items.extend(_flatten(v, new_key, sep=sep).items())
        elif isinstance(v, (list, tuple)):
            items.extend(_flatten(enumerate(v), new_key, sep=sep).items())
        elif isinstance(v, np.array) and len(v.squeeze().shape) == 1:
            items.extend(_flatten(enumerate(v.squeeze()), new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

def _parse_result_access(result, access):
    """
    Helper function to access (potentially nested)
    attributes of `result`. Potentially useful when
    using third-party functions for feature extraction
    and we don't want to keep the whole output. Only
    supports a very simplistic way of accessing attributes:
    by using square brackets (even for object attributes)
    """
    if isinstance(access, int):
        return result[access]

    if isinstance(access, str):
        if "]" not in access:
            return result[access]

        access = access.replace("]", "").split("[")
        while len(access):
            ix = access.pop(0)

            if isinstance(result, collections.abc.MutableMapping):
                if ix in result:
                    result = result[ix]
                elif int(ix) in result:
                    result = result[int(ix)]
                else:
                    raise ValueError(f"Key '{ix}' not found")
            elif isinstance(result, (list, tuple)):
                result = result[int(ix)]
            else:
                result = getattr(result, ix)

        return result

    raise TypeError(f"Can't parse `access` of type {type(access)}")


def _preload_pipeline(pipeline):
    pipeline = deepcopy(pipeline)
    _new_transforms = defaultdict(dict)
    for name, transform in pipeline.items():
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

            # if a key starts with a '+' it means its value
            # is meant to be obtained "live", either from previously
            # calculated results in the pipeline, or from a value
            # in the dataframe row. these will then be passed as kwargs
            kwargs = {}
            orig_keys = list(step.keys())
            for key in orig_keys:
                 if key[0] == "+":
                     kwargs[key[1:]] = step.pop(key)

            # in case this step is a class, the kwargs
            # need to be used in its __init__, so we turn it
            # into a ^bind, and signal that it must be inited
            # before it is called
            must_init = False
            if len(kwargs) > 0:
                if "^init" in step:
                    step["^bind"] = step["^init"]
                    del step["^init"]
                    must_init = True

            # retrieve another special key "^unpack", if it's present.
            # its use is explained below
            unpack = step.pop("^unpack", False)

            # load the step with our dynamic import functionality
            step = load_config(step)
            _steps.append((step, kwargs, unpack, must_init))
        _new_transforms[name]["steps"] = _steps
    return _new_transforms


def apply_pipeline_once(
    row: Union[Dict, pd.Series],
    pipeline,
    reader=None
):
    """
    Apply pipeline defined by `pipeline` to images in
    paths specified in columns of `row`

    Parameters
    ----------
    row: Union[Dict, pd.Series]
        The dataframe row (or equivalent dict or namedtuple) containing
        the necessary fields

    pipeline: Dict
        The dictionary specifying the transforms to apply
    """

    pipeline = deepcopy(pipeline)
    results = dict()

    # the `pipeline` dict contains a field "transforms",
    # which is the sequence of transforms needed to obtain the output image.
    # each element of "transforms" is a dict containing instructions for
    # a sequence of steps

    # `name` will be used to refer to the result of this transform,
    # when it's stored in the `results` dict.
    for name, transform in pipeline.items():
        # a step may have as starting point either a file on disk,
        # or the output of a previous step. if a step's "input" field
        # should be a dictionary, where each key is a string - indicating
        # either a previous step's result or a column in the dataframe -
        # and each value is the list of channels to use for that input.
        # if an input is being read from disk, channel labels can be used.
        # if an input is being read as the result of a previous step, channel
        # indices must be used

        if isinstance(transform["input"], str):
            _inputs = [(transform["input"], None)]
        elif isinstance(transform["input"], (dict, DictConfig)):
            _inputs = transform["input"].items()
        else:
            raise TypeError(f"Unexpected type for `input`: {transform['input']}")

        inputs = []
        for _input, channels in _inputs:
            if _input in results:
                _input = results[_input]
            elif _input in row:
                _input = row[_input]
                if isinstance(_input, str) and Path(_input).is_file():
                    _input, channel_map = image_loader(_input,
                                                       output_dtype="float32",
                                                       reader=reader,
                                                       return_channels=True)
            else:
                raise ValueError(f"Given input not found: {_input}")

            if channels is None:
                inputs += [_input]
            elif isinstance(channels, int):
                inputs += [_input[channels]]
            elif isinstance(channels, str):
                inputs += [_inputs[channel_map[channels]]]

            else:
                channels_type = set(map(type, channels)).pop()
                if channels_type == int:
                    channel_map = list(range(_input.shape[0]))
                inputs += [_input[[channel_map[ch] for ch in channels]]]

        # after the starting point for this transform has been loaded,
        # we iterate over the steps listed in this transform's
        # "steps" field. each element of "steps" is a dict
        # specifying a specific transform. these dicts are parsed
        # using our dynamic_import utils. at this point, the transforms
        # have been preloaded in the main process and passed to each worker
        for (step, kwargs, unpack, must_init) in transform["steps"]:
            kwargs = dict()
            for arg, column in kwargs.items():
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
            # `inputs` can be a list with a single image, if this is
            # the first step in this transform
            if isinstance(inputs, list) and len(inputs) == 1:
                inputs = step(inputs[0], **kwargs)
            else:
                # if the input `inputs` at this point consists of
                # more than a single image, there is an additional
                # argument "^unpack" that can be specified for this
                # step, which tells us to use the * operator here
                inputs = step(*inputs, **kwargs) if unpack else step(inputs, **kwargs)

        results[name] = inputs

    return results


def _apply_once_and_store_results(
    row: Union[pd.Series, Dict],
    pipeline: Dict,
    outputs: Dict[str, Sequence[str]],
    output_path: Union[str, Path],
    index_col: str,
    reader: Optional[str] = None,
):
    """
    Transform an image in a dataframe row, using transforms given by
    config in `pipeline`

    Parameters:
    ---
    row: Union[pd.Series, Dict]
        The DataFrame row from which the image path is taken

    pipeline: Dict
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
    pipeline_results = apply_pipeline_once(row, pipeline, reader=reader)
    for output, args in outputs.items():
        _result = pipeline_results[output]
        if isinstance(_result, torch.Tensor):
            _result = _result.numpy()

        if isinstance(_result, np.array) and len(_result.shape) >= 3:
            channel_names = args
            _result_path = output_path / f"{row[index_col]}_{output}.tiff"
            tiff_writer(_result, _result_path, channel_names=channel_names)
            result[output] = str(_result_path)
        else:
            if args is not None:
                if isinstance(args, str):
                    _result = _parse_result_access(_result, args)
                else:
                    _result = [
                        _parse_result_access(_result, arg)
                        for arg in args
                    ]
            result[output] = _result

    return _flatten(result)


def apply_pipeline_batch(
    input_manifest: Union[pd.DataFrame, Union[str, Path]],
    output_path: Union[str, Path],
    pipeline: Sequence,
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
    config in `pipeline`

    Parameters
    ----------
    input_manifest: Union[pd.DataFrame, Union[str, Path]]
        Path to the input manifest, or a pd.DataFrame

    output_path: Union[str, Path]
        Path to the folder where the outputs will be stored

    pipeline: Sequence
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

    # make helper function with pre attributed params except df row
    apply_pipeline = partial(
        _apply_once_and_store_results,
        pipeline=_preload_pipeline(pipeline),
        outputs=outputs,
        output_path=Path(output_path),
        index_col=index_col,
        reader=image_reader,
    )

    return _backend(
        input_manifest=input_manifest,
        func_per_row=apply_pipeline,
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
