import logging
from typing import Union, Dict
from pathlib import Path

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch

from torch_geometric.loader import GraphSAINTRandomWalkSampler, RandomNodeSampler

# from torch_geometric.data import GraphSAINTRandomWalkSampler
from torch_geometric.utils import degree

from sklearn.model_selection import train_test_split
from serotiny.io.dataframe.utils import filter_columns
from serotiny.datamodules.datasets import WholeGraph

log = logging.getLogger(__name__)


class SingleGraphDatamodule(pl.LightningDataModule):
    """
    A pytorch lightning datamodule that handles the logic for iterating over a
    folder of files

    Parameters
    -----------
    batch_size: int
        batch size for dataloader
    num_workers: int
        Number of worker processes for dataloader
    manifest: Optional[Union[Path, str]] = None
        (optional) Path to a manifest file to be merged with the folder, or to
        be the core of this dataset, if no path is provided
    loader_dict: Dict
        Dictionary of loader specifications for each given key. When the value
        is callable, that is the assumed loader. When the value is a tuple, it
        is assumed to be of the form (loader class name, loader class args) and
        will be used to instantiate the loader
    split_col: Optional[str] = None
        Name of a column in the dataset which can be used to create train, val, test
        splits.
    columns: Optional[Sequence[str]] = None
        List of columns to load from the dataset, in case it's a parquet file.
        If None, load everything.
    pin_memory: bool = True
        Set to true when using GPU, for better performance
    drop_last: bool = False
        Whether to drop the last batch (in case the given batch size is the only
        supported)
    subset_train: float = 1.0
    """

    def __init__(
        self,
        batch_size: int,
        num_workers: int,
        num_cores: int,
        save_dir: str,
        validated_manifest: Union[Path, str],
        node_loader: Dict,
        task_dict: dict,
        model_type: str,
        dataloader_type: str,
        subset: bool,
        normalize: bool,
    ):

        super().__init__()

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.save_dir = save_dir
        self.model_type = model_type
        self.node_loader = node_loader
        self.subset = subset
        self.normalize = normalize

        # To delete later, this is only to get length

        dataset = WholeGraph(
            root=save_dir,
            node_loader=node_loader,
            num_cores=num_cores,
            single_graph=True,
            task_dict=task_dict,
            val_manifest=validated_manifest,
            train_test_split=True,
            subset=subset,
            normalize=normalize,
        )
        data = dataset[0]
        dataset_df = dataset.df

        # Do this again in case we load the same dataset
        # but want to use different node columns
        self.node_cols = filter_columns(
            dataset_df.columns.to_list(), **self.node_loader
        )

        # Make sure not all 0 in some columns
        df2 = dataset_df[self.node_cols].copy()
        df1 = df2.loc[:, (df2 != 0).all()]
        self.node_cols = df1.columns

        if self.normalize:
            dataset_df[self.node_cols] = dataset_df[self.node_cols].apply(
                lambda x: (x - x.min()) / (x.max() - x.min())
            )

        if (validated_manifest is not None) & (validated_manifest != "None"):
            main_manifest = dataset.mask_df
            val_manifest = pd.read_csv(validated_manifest)

            # train test split based on tracks
            all_val_tracks = val_manifest["track_id"].unique()
            np.random.shuffle(all_val_tracks)
            train_tracks = all_val_tracks[: int(0.6 * len(all_val_tracks))]
            val_test_tracks = np.setdiff1d(all_val_tracks, train_tracks)
            np.random.shuffle(val_test_tracks)
            test_tracks = val_test_tracks[: int(0.5 * len(val_test_tracks))]
            val_tracks = np.setdiff1d(val_test_tracks, test_tracks)

            train_df = val_manifest.loc[val_manifest["track_id"].isin(train_tracks)]
            val_df = val_manifest.loc[val_manifest["track_id"].isin(val_tracks)]
            test_df = val_manifest.loc[val_manifest["track_id"].isin(test_tracks)]

            train_df["train_mask"] = True
            val_df["val_mask"] = True
            test_df["test_mask"] = True

            val_manifest_with_mask = pd.concat([train_df, val_df, test_df], axis=0)
            val_manifest_with_mask = val_manifest_with_mask[
                ["CellId", "train_mask", "val_mask", "test_mask"]
            ]
            val_manifest_with_mask = val_manifest_with_mask.replace({np.NaN: False})

            main_manifest = main_manifest.drop(
                columns=["train_mask", "val_mask", "test_mask"]
            )
            main_manifest = main_manifest.merge(
                val_manifest_with_mask, how="left", left_on="CellId", right_on="CellId"
            )
            main_manifest = main_manifest.replace({np.NaN: False})

            merged_cols = []
            for col in self.node_cols:
                if "pc" in col:
                    merged_cols.append(col)
                else:
                    merged_cols.append(col + "_x")

            if self.normalize:
                main_manifest[merged_cols] = main_manifest[merged_cols].apply(
                    lambda x: (x - x.min()) / (x.max() - x.min())
                )
            # adding data.x again in case we are using the same dataset
            # but want to change the node cols
            data.x = torch.tensor(
                np.array(main_manifest[merged_cols], dtype=np.float64),
                dtype=torch.float,
            )

            data.train_mask = torch.tensor(
                main_manifest["train_mask"].values, dtype=torch.bool
            )
            data.val_mask = torch.tensor(
                main_manifest["val_mask"].values, dtype=torch.bool
            )
            data.test_mask = torch.tensor(
                main_manifest["test_mask"].values, dtype=torch.bool
            )

            save_path = Path(save_dir + "/data")
            csv_path_mask = save_path / "input_to_graph_with_track_id_split.csv"
            self.mask_df_with_splits = main_manifest
            if not csv_path_mask.is_file():
                main_manifest.to_csv(csv_path_mask)
        else:
            main_manifest = dataset_df.copy()

            data.x = torch.tensor(
                main_manifest[self.node_cols].values, dtype=torch.float,
            )

            size = data.x.shape[0]
            X_train_val, X_test = train_test_split(
                np.array([i for i in range(size)]), test_size=0.15, random_state=42
            )
            X_train, X_val = train_test_split(
                X_train_val, test_size=0.15, random_state=42
            )

            assert len(set(X_train).intersection(set(X_val))) == 0
            assert len(set(X_train).intersection(set(X_test))) == 0
            assert len(set(X_test).intersection(set(X_val))) == 0

            train_mask = np.array([False for i in range(size)])
            train_mask[X_train] = True

            val_mask = np.array([False for i in range(size)])
            val_mask[X_val] = True

            test_mask = np.array([False for i in range(size)])
            test_mask[X_test] = True

            data.train_mask = torch.tensor(train_mask, dtype=torch.bool)
            data.val_mask = torch.tensor(val_mask, dtype=torch.bool)
            data.test_mask = torch.tensor(test_mask, dtype=torch.bool)

        # Do this again in case we want to load new columns
        target_label = task_dict["target_label"]

        if task_dict["task"] == "classification":
            num_bins = int(task_dict["num_bins"])

            main_manifest[target_label + "_int"] = pd.qcut(
                main_manifest[target_label],
                q=[i / (num_bins) for i in range(num_bins)] + [1],
                labels=False,
            )
            main_manifest[target_label + "_int"] = main_manifest[
                target_label + "_int"
            ].fillna(num_bins)
            targets = torch.tensor(
                main_manifest[target_label + "_int"], dtype=torch.long
            )
        elif task_dict["task"] == "regression":
            main_manifest[target_label] = main_manifest[target_label].fillna(10)
            targets = torch.tensor(main_manifest[target_label], dtype=torch.float)

        data.y = targets
        main_manifest['train_mask'] = train_mask
        main_manifest['val_mask'] = val_mask
        main_manifest['test_mask'] = test_mask
        data.cellid = main_manifest['CellId'].values
        weights = torch.tensor(
            [
                len(data.y) / i
                for i in np.unique(np.array(data.y), return_counts=True)[1]
            ]
        )
        data.weights = weights
        row, col = data.edge_index
        data.edge_weight = 1.0 / degree(col, data.num_nodes)[col]

        self.dataset = dataset
        self.dataframe = main_manifest
        self.length = len(dataset)
        self.data = data
        self.dataloader_type = dataloader_type
        self.weights = torch.tensor(
            [
                len(data.y) / i
                for i in np.unique(np.array(data.y), return_counts=True)[1]
            ]
        )

        print()
        print(f"Dataset: {dataset}:")
        print("======================")
        print(f"Number of graphs: {len(dataset)}")
        print(f"Number of features: {data.x.shape[1]}")
        print(f"Number of classes: {dataset.num_classes}")

        print()
        print(data)
        print("===========================================================")

        # Gather some statistics about the graph.
        print(f"Number of nodes: {data.num_nodes}")
        print(f"Number of edges: {data.num_edges}")
        print(f"Average node degree: {data.num_edges / data.num_nodes:.2f}")
        # print(f"Contains isolated nodes: {data.has_isolated_nodes()}")
        # print(f"Contains self-loops: {data.has_self_loops()}")
        print(f"Is undirected: {data.is_undirected()}")

        # self.dataloader = self.make_dataloader()

    def make_dataloader(self):
        save_path = Path(self.save_dir + f"/{self.model_type}")
        save_path.mkdir(parents=True, exist_ok=True)
        if self.dataloader_type == "graph_saint":
            return GraphSAINTRandomWalkSampler(
                self.data,
                batch_size=self.batch_size,
                walk_length=2,
                num_steps=20,
                sample_coverage=100,
                save_dir=save_path,
                num_workers=self.num_workers,
            )
        else:
            return RandomNodeSampler(self.data, num_parts=1)

    def train_dataloader(self):
        return self.make_dataloader()

    def val_dataloader(self):
        return self.make_dataloader()

    def test_dataloader(self):
        return self.make_dataloader()

