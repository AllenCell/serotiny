from typing import Union, Dict, Sequence, Optional
from pathlib import Path
import traceback

from functools import partial

import multiprocessing_on_dill as mp
import pandas as pd

from tqdm import tqdm

def _write_rows(tmp_manifest, rows_to_write):
    tmp = pd.DataFrame.from_records(rows_to_write)
    tmp.to_csv(
        tmp_manifest,
        mode=("a" if tmp_manifest.exists() else "w"),
        header=(not tmp_manifest.exists()),
        index=False
    )
    rows_to_write.clear()


def _parse_row(
    row: Union[pd.Series, Dict],
    func,
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

    outputs: Dict[str, Sequence[str]]
        Dictionary containing the keys for each output to save to disk,
        and the corresponding channel names for that output. The channel
        names are assumed to be in the correct order for each output

    output_path: Union[str, Path]
        Path to the folder where the outputs will be stored

    index_col: str
        Column to serve as index. Used for output filenames

    include_cols: Sequence[str]
        List of columns to include in the output manifest, aside from the
        column containing paths to the output images

    debug: bool = False
        Whether to return errors in csv instead of raising and breaking
    """

    result = {col: row[col] for col in include_cols}
    result[index_col] = row[index_col]
    result["errors"] = ""

    try:
        result.update(func(row))
    except Exception as e:
        if debug:
            result["errors"] = traceback.format_exc()
        else:
            raise e

    return result


def _parse_batch(
    input_manifest: Union[pd.DataFrame, Union[str, Path]],
    func_per_row,
    output_path: Union[str, Path],
    index_col: str,
    include_cols: Sequence[str] = [],
    write_every_n_rows: int = 100,
    return_merged: bool = False,
    backend: str = "multiprocessing",
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

    index_col: str
        Column to serve as index. Used for output filenames

    write_every_n_rows: int = 100
        Write results to temporary file every n rows

    include_cols: Sequence[str] = []
        List of columns to include in the output manifest, aside from the
        column containing paths to the output images

    return_merged: bool = True
        If True, return the result manifest merged with the input.
        Otherwise, return only the result.

    backend: str = "multiprocessing"
        Backend to use for parallel computation. Possible values are
        "multiprocessing", "slurm"

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

    if not isinstance(input_manifest, pd.DataFrame):
        # import here to optimize CLIs / Fire usage
        from serotiny.io.dataframe import read_dataframe
        input_manifest = read_dataframe(input_manifest)

    output_path = Path(output_path)
    if not output_path.exists():
        if output_path.suffix == "":
            output_path.mkdir(parents=True)
        else:
            output_path.parent.mkdir(parents=True)

    to_skip = set()

    if output_path.suffix == "":
        tmp_manifest_path = (output_path / "manifest.tmp.csv")
    else:
        tmp_manifest_path = output_path.with_suffix(".tmp.csv")

    if skip_if_exists:
        if tmp_manifest_path.exists():
            tmp_manifest = pd.read_csv(tmp_manifest_path)
            tmp_manifest = tmp_manifest.loc[
                pd.isnull(tmp_manifest["errors"]) |
                (tmp_manifest["errors"] == "")
            ]
            to_skip = set(tmp_manifest[index_col].unique().tolist())

    # make helper function with pre attributed params except df row
    _func_per_row = partial(
        _parse_row,
        func=func_per_row,
        index_col=index_col,
        include_cols=include_cols,
        debug=debug,
    )

    iter_rows = (dict(**row) for _, row in input_manifest.iterrows()
                 if row[index_col] not in to_skip)

    pool = None
    if n_workers > 1:
        if backend == "multiprocessing":
            pool = mp.Pool(n_workers)
            jobs = pool.imap_unordered(_func_per_row, iter_rows)
            if verbose:
                jobs = tqdm(jobs, total=len(input_manifest)-len(to_skip))

        else:
            raise NotImplementedError(f"Parallel backend {backend} not implemented.")
    else:
        if verbose:
            jobs = tqdm(iter_rows, total=len(input_manifest)-len(to_skip))
        jobs = map(_func_per_row, jobs)

    rows_to_write = []
    for result in jobs:
        rows_to_write.append(result)
        if len(rows_to_write) == write_every_n_rows:
            _write_rows(tmp_manifest_path, rows_to_write)
    if len(rows_to_write) > 0:
        _write_rows(tmp_manifest_path, rows_to_write)

    if hasattr(pool, "close"):
        pool.close()

    if output_path.suffix == "":
        output_path = output_path / "manifest.csv"
    tmp_manifest_path.rename(output_path)

    if return_merged:
        success = pd.read_csv(output_path)
        success = success.merge(input_manifest, on=index_col)
        success.to_csv(output_path, index=False)
