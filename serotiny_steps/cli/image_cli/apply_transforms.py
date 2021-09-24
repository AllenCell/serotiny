from typing import Union, Dict, Sequence, Optional
from pathlib import Path
import traceback

import json
import pandas as pd
import multiprocessing_on_dill as mp

from tqdm import tqdm

from serotiny.io.image import tiff_writer, image_loader
from serotiny.utils import load_config


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
    for step in transforms_to_apply["steps"]:

        # `name` will be used to refer to the result of this step,
        # when it's stored in the `result_imgs` dict.
        name = step["name"]

        # a step may have as starting point either a file on disk,
        # or the output of a previous step. in the first case,
        # a field "path_col" indicates which dataframe column to use.
        # in the second case, a field "input_imgs" tells us which
        # key of `result_imgs` to use.
        if "path_col" in step:  # load image from disk
            path_col = step["path_col"]
            channels = step["channels"]
            img, channel_map = image_loader(row[path_col], return_channels=True)
            imgs = [img[[channel_map[ch] for ch in channels]]]
        elif "input_imgs" in step:  # use result of previous step
            imgs = []
            for inpt in step["input_imgs"]:
                imgs += [img for img in result_imgs[inpt]]

        # after the starting point for this step has been loaded,
        # we iterate over the transforms listed in this step's
        # "transforms" field. each element of this list is a dict
        # specifying a specific transform. these dicts are parsed
        # using our dynamic_import utils
        transforms_configs = step["transforms"]
        for transform in transforms_configs:
            # if an "individual_args" key is present, as that means
            # there are values we need to retrieve from this row
            # as additional arguments to the transform
            if "individual_args" in transform:
                for arg, column in transform["individual_args"].items():
                    transform[arg] = json.loads(row[column])

            # load the transform using our dynamic import logic
            transform = load_config(
                {k: v for k, v in transform.items() if k != "individual_args"}
            )

            # the result of the transform is always stored in a list,
            # even if it's a single image
            if len(imgs) == 1:
                imgs = [transform(imgs[0])]
            else:
                # if the input `imgs` at this point consists of
                # more than a single image, there is an additional
                # argument "unzip" that can be specified for this
                # step, which tells us to use the * operator here
                unzip = step.get("unzip", False)
                imgs = [transform(*imgs)] if unzip else [transform(imgs)]

        result_imgs[name] = imgs

    # finally, the key of `result_imgs` which contains the output
    # image is specified in the "output" field in the config
    output_key = transforms_to_apply["output"]
    return result_imgs[output_key][0]


def transform_from_row(
    row: Union[pd.Series, Dict],
    transforms_to_apply: Dict,
    output_channel_names: Sequence[str],
    output_path: Union[str, Path],
    index_col: str = "CellId",
    include_cols: Sequence[str] = [],
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
        result["errors"] = traceback.format_exc()

    return result


def transform_batch(
    *input_manifests: Sequence[Union[str, Path]],
    output_path: Union[str, Path],
    output_channel_names: Sequence[str],
    transforms_to_apply: Dict,
    merge_col: Optional[str] = None,
    n_workers: int = 1,
    index_col: str = "CellId",
    include_cols: Sequence[str] = [],
    verbose: bool = False,
):
    """
    Extract features from an image in a dataframe row, using extractors given by
    config in `transforms_to_apply`

    Parameters
    ----------
    input_manifest: Sequence[Union[str, Path]]
        Path(s) to the input manifest(s)

    output_path: Union[str, Path]
        Path to the folder where the outputs will be stored

    output_channel_names: Sequence[str]
        List of the names to be assigned to the channels in the output
        image (in order)

    transforms_to_apply: Dict
        Config dictionary specifying what transforms to apply

    merge_col: Optional[str] = None
        Column on which to merge, in case there multiple input manifests are
        provided

    index_col: Optional[str] = "CellId"
        Column to serve as index

    n_workers: int = 1
        Number of multiprocessing workers to use for parallel computation

    verbose: bool = False
        Flag to tell whether to produce command line output

    """

    input_manifest = None
    if len(input_manifests) > 1:
        assert merge_col is not None

    for manifest in input_manifests:
        manifest = Path(manifest)
        if not manifest.exists():
            raise FileNotFoundError(f"Given input path {manifest} does not exist")
        manifest = pd.read_csv(manifest)

        if input_manifest is not None:
            input_manifest = input_manifest.merge(manifest, on=merge_col, how="outer")
        else:
            input_manifest = manifest

    output_path = Path(output_path)
    if not output_path.exists():
        output_path.mkdir(parents=True)

    # make helper function with pre attributed params except df row
    apply_transf = lambda row: transform_from_row(
        row,
        transforms_to_apply,
        output_channel_names,
        output_path,
        index_col=index_col,
        include_cols=include_cols,
    )
    iter_rows = (row for _, row in input_manifest.iterrows())

    if n_workers > 1:
        with mp.Pool(n_workers) as pool:
            jobs = pool.imap_unordered(apply_transf, iter_rows)
            if verbose:
                jobs = tqdm(jobs, total=len(input_manifest))
            success = pd.DataFrame.from_records(list(jobs))
    else:
        if verbose:
            iter_rows = tqdm(iter_rows, total=len(input_manifest))
        success = pd.DataFrame.from_records([apply_transf(row) for row in iter_rows])

    success.to_csv(output_path / "manifest.csv")
