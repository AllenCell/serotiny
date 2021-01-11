#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
from pathlib import Path
from typing import Dict, List, Optional, Union
from sklearn.model_selection import train_test_split

import pandas as pd

from datastep import Step, log_run_params

from ..project_2d import Project2D
from ...constants import DatasetFields
from ...library.csv import load_csv
from ...library.data import append_one_hot

###############################################################################

log = logging.getLogger(__name__)

###############################################################################


REQUIRED_DATASET_FIELDS = [
    DatasetFields.Chosen2DProjectionPath,
    DatasetFields.ChosenMitoticClass,
]


class TrainTestSplit(Step):
    def __init__(
        self,
        direct_upstream_tasks: List["Step"] = [Project2D],
        config: Optional[Union[str, Path, Dict[str, str]]] = None,
    ):
        super().__init__(direct_upstream_tasks=direct_upstream_tasks, config=config)

    @log_run_params
    def run(
        self,
        dataset: Union[str, Path, pd.DataFrame],
        **kwargs,
    ):
        """
        Run a pure function.

        Protected Parameters
        --------------------
        distributed_executor_address: Optional[str]
            An optional executor address to pass to some computation engine.
        clean: bool
            Should the local staging directory be cleaned prior to this run.
            Default: False (Do not clean)
        debug: bool
            A debug flag for the developer to use to manipulate how much data runs,
            how it is processed, etc.
            Default: False (Do not debug)

        Parameters
        ----------

        Returns
        -------
        result: Any
            A pickable object or value that is the result of any processing you do.
        """
        dataset = load_csv(dataset, REQUIRED_DATASET_FIELDS)
        dataset.dropna(inplace=True)
        dataset, one_hot_len = append_one_hot(
            dataset, DatasetFields.ChosenMitoticClass, "CellId"
        )

        # split dataset into train, test and validtion subsets
        # TODO: make split ratio a parameter (currently 0.2)
        indexes = {}
        index_train_valid, indexes["test"] = train_test_split(
            dataset.index, test_size=0.2
        )
        indexes["train"], indexes["valid"] = train_test_split(
            dataset.loc[index_train_valid, :].index
        )

        # index split datasets
        datasets = {
            key: pd.DataFrame(dataset.loc[index, :].reset_index(drop=True))
            for key, index in indexes.items()
        }

        dataset_paths = {}

        # save a dataloader for each dataset
        for split, dataset in datasets.items():
            save_path = self.step_local_staging_dir / f"dataset_{split}.csv"
            dataset.to_csv(save_path, index=False)
            dataset_paths[split] = str(save_path)

        output = {"dataset_paths": dataset_paths, "one_hot_len": one_hot_len}

        return output
