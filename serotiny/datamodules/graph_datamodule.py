import logging
from typing import Union, Dict
from pathlib import Path

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import ast
import multiprocessing
from functools import partial
import tqdm
import torch

from torch_geometric.loader import GraphSAINTRandomWalkSampler

# from torch_geometric.data import GraphSAINTRandomWalkSampler
from torch_geometric.utils import degree

from torch_geometric.data import Data, InMemoryDataset
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from serotiny.io.dataframe.utils import filter_columns

from aicsfiles import FileManagementSystem

log = logging.getLogger(__name__)


def parallelize_dataframe(df, func, num_cores, subset=None, groupby_val=None, **kwargs):

    if subset:
        df = df.loc[df[groupby_val].isin([i for i in range(2)])]
        num_cores = 1

    num_partitions = num_cores  # number of partitions to split dataframe
    splits = np.array_split(df[groupby_val].unique(), num_partitions)
    func_partial = partial(
        func, df, groupby_val, *[value for key, value in kwargs.items()]
    )

    pool = multiprocessing.Pool(num_cores)

    vals = list(tqdm.tqdm(pool.imap(func_partial, splits), total=len(splits)))

    for i, val in enumerate(vals):
        if i == 0:
            output_lists = [[] for _ in range(len(val))]
        for j, out in enumerate(val):
            output_lists[j].append(np.array(out))

    for ind in range(len(val)):
        output_lists[ind] = np.concatenate(output_lists[ind], axis=0)

    pool.close()
    pool.join()
    return output_lists


def histedges_equalN(x, nbin):
    npt = len(x)
    return np.interp(np.linspace(0, npt, nbin + 1), np.arange(npt), np.sort(x))


def compute_bins(df, key, num_bins, dont_update=None):
    vol = np.array(df[[key]])
    vol = np.around(vol, 1)
    df[[key]] = vol
    arr = histedges_equalN(vol.squeeze(), num_bins)
    n, bins, patches = plt.hist(vol.squeeze(), arr)
    hist = np.histogram(vol, bins=bins)
    num_bins = len(hist[0])
    inds = np.fmin(np.digitize(vol, hist[1]), num_bins)
    bin_edges = hist[1]

    # Add bins per dim to embeddings
    bin_values = []
    for i in inds:
        bin_values.append((bin_edges[i - 1].item(), bin_edges[i].item()))
    if not dont_update:
        df[f"bins_{key}"] = bin_values

    return df


class GraphDatamodule(pl.LightningDataModule):
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
        semi_supervised: bool,
        task_dict: dict,
        model_type: str,
        subset_train: float = 1.0,
        fms: bool = False,
    ):

        super().__init__()

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.save_dir = save_dir
        self.model_type = model_type
        self.node_loader = node_loader

        assert subset_train <= 1

        # To delete later, this is only to get length

        dataset = WholeGraph(
            root=save_dir,
            node_loader=node_loader,
            num_cores=num_cores,
        )

        data = dataset[0]  # Get the first graph object.
        row, col = data.edge_index
        data.edge_weight = 1.0 / degree(col, data.num_nodes)[col]

        dataset_df = dataset.df
        val_manifest = pd.read_csv(validated_manifest)

        # Do this again in case we load the same dataset
        # but want to use different node columns
        self.node_cols = filter_columns(
            dataset_df.columns.to_list(), **self.node_loader
        )
        dataset_df[self.node_cols] = dataset_df[self.node_cols].apply(
            lambda x: (x - x.min()) / (x.max() - x.min())
        )

        if semi_supervised:
            size = val_manifest.shape[0]
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

            val_manifest["test_mask"] = test_mask
            val_manifest["val_mask"] = val_mask
            val_manifest["train_mask"] = train_mask

            main_manifest = dataset_df.merge(
                val_manifest, how="left", left_on="CellId", right_on="CellId"
            )

            main_manifest["test_mask"] = main_manifest["test_mask"].fillna(False)
            main_manifest["train_mask"] = main_manifest["train_mask"].fillna(False)
            main_manifest["val_mask"] = main_manifest["val_mask"].fillna(False)

            data.train_mask = torch.tensor(
                main_manifest["train_mask"].values, dtype=torch.bool
            )
            data.val_mask = torch.tensor(
                main_manifest["val_mask"].values, dtype=torch.bool
            )
            data.test_mask = torch.tensor(
                main_manifest["test_mask"].values, dtype=torch.bool
            )
            # adding data.x again in case we are using the same dataset
            # but want to change the node cols
            data.x = torch.tensor(
                main_manifest[[j + "_x" for j in self.node_cols]].values,
                dtype=torch.float,
            )
        else:
            main_manifest = dataset_df.copy()
            data.x = torch.tensor(
                main_manifest[self.node_cols].values,
                dtype=torch.float,
            )

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
            targets = torch.tensor(main_manifest[target_label], dtype=torch.float)

        data.y = targets
        weights = torch.tensor(
            [
                len(data.y) / i
                for i in np.unique(np.array(data.y), return_counts=True)[1]
            ]
        )
        data.weights = weights

        self.dataset = dataset
        self.length = len(dataset)
        self.data = data
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
        return GraphSAINTRandomWalkSampler(
            self.data,
            batch_size=self.batch_size,
            walk_length=2,
            num_steps=5,
            sample_coverage=100,
            save_dir=save_path,
            num_workers=self.num_workers,
        )

    def train_dataloader(self):
        return self.make_dataloader()

    def val_dataloader(self):
        return self.make_dataloader()

    def test_dataloader(self):
        return self.make_dataloader()


def get_edges(
    dataframe,
    groupby_val,
    node_cols,
    flat_list,
    flat_x_centroids,
    flat_y_centroids,
    flat_z_centroids,
    groupby_split,
):

    all_edges, all_edge_attr = [], []
    dataframe = dataframe.set_index("CellId")
    for split in groupby_split:
        df3 = dataframe.loc[dataframe[groupby_val] == split]
        edges = []
        edge_attr = []
        for index, row in df3.iterrows():
            first_node = list(row[node_cols])
            try:
                neighbor_ids = ast.literal_eval(row["neighbors"])
            except:
                neighbor_ids = None

            if neighbor_ids is not None:

                first_node_index = flat_list.index(first_node)

                for this_id in neighbor_ids:
                    if this_id in df3.index:
                        this_node = list(dataframe.loc[this_id][node_cols])
                        this_node_index = flat_list.index(this_node)
                        coords_1 = np.array(
                            [
                                flat_x_centroids[first_node_index],
                                flat_y_centroids[first_node_index],
                                flat_z_centroids[first_node_index],
                            ]
                        )
                        coords_2 = np.array(
                            [
                                flat_x_centroids[this_node_index],
                                flat_y_centroids[this_node_index],
                                flat_z_centroids[this_node_index],
                            ]
                        )
                        dist = np.linalg.norm(coords_1 - coords_2)
                        edg1 = [first_node_index, this_node_index]

                        if edg1[0] != edg1[1]:
                            edges.append(edg1)
                            edge_attr.append(dist)

        edges = np.asarray(edges)
        edge_attr = np.asarray(edge_attr)

        all_edges.append(edges)
        all_edge_attr.append(edge_attr)

    all_edges = [i for i in all_edges if i.shape[0] != 0]
    all_edge_attr = [i for i in all_edge_attr if i.shape[0] != 0]

    all_edges = np.concatenate(all_edges, axis=0)
    all_edge_attr = np.concatenate(all_edge_attr, axis=0)

    return all_edges, all_edge_attr


def get_track_edges(
    dataframe,
    groupby_val,
    node_cols,
    flat_list,
    flat_x_centroids,
    flat_y_centroids,
    flat_z_centroids,
    groupby_split,
):

    all_edges, all_edge_attr = [], []
    dataframe = dataframe.set_index("CellId")
    for split in groupby_split:
        df3 = dataframe.loc[dataframe[groupby_val] == split]
        if df3.shape[0] > 30:
            lags = 30
        else:
            lags = df3.shape[0]
        for lag in range(lags):
            df3 = dataframe.loc[dataframe[groupby_val] == split]
            df4 = df3.iloc[lag + 1 :].copy()[node_cols]
            df3 = df3.iloc[: -(lag + 1)][node_cols]
            track_edges = []
            track_edge_attr = []
            for i, (x, y) in enumerate(zip(df3.values, df4.values)):
                first_node = list(x)

                first_node_index = flat_list.index(first_node)

                this_node = list(y)
                this_node_index = flat_list.index(this_node)
                coords_1 = np.array(
                    [
                        flat_x_centroids[first_node_index],
                        flat_y_centroids[first_node_index],
                        flat_z_centroids[first_node_index],
                    ]
                )
                coords_2 = np.array(
                    [
                        flat_x_centroids[this_node_index],
                        flat_y_centroids[this_node_index],
                        flat_z_centroids[this_node_index],
                    ]
                )
                dist = np.linalg.norm(coords_1 - coords_2)
                edg1 = [first_node_index, this_node_index]
                if edg1[0] != edg1[1]:
                    track_edges.append(edg1)
                    track_edge_attr.append(dist)
            track_edges = np.asarray(track_edges)
            track_edge_attr = np.asarray(track_edge_attr)

            all_edges.append(track_edges)
            all_edge_attr.append(track_edge_attr)

    all_edges = [i for i in all_edges if i.shape[0] != 0]
    all_edge_attr = [i for i in all_edge_attr if i.shape[0] != 0]

    all_edges = np.concatenate(all_edges, axis=0)
    all_edge_attr = np.concatenate(all_edge_attr, axis=0)

    return all_edges, all_edge_attr


def get_nodes(dataframe, groupby_val, node_cols, groupby_split):
    all_nodes, all_nodes_list = [], []
    all_nodes_x_position, all_nodes_y_position, all_nodes_z_position = [], [], []
    for split in groupby_split:
        df3 = dataframe.loc[dataframe[groupby_val] == split]
        nodes = []
        nodes_x_position, nodes_y_position, nodes_z_position = [], [], []
        for index, row in df3.iterrows():
            try:
                neighbor_ids = ast.literal_eval(row["neighbors"])
            except:
                neighbor_ids = None
            if neighbor_ids is not None:
                node = row[node_cols]
                nodes_x_position.append(row["centroid_x"])
                nodes_y_position.append(row["centroid_y"])
                nodes_z_position.append(row["centroid_z"])
                nodes.append(node)

        nodes_np = np.asarray(nodes)
        all_nodes_x_position.append(nodes_x_position)
        all_nodes_y_position.append(nodes_y_position)
        all_nodes_z_position.append(nodes_z_position)
        all_nodes.append(nodes_np)
        all_nodes_list.append(nodes)

    all_nodes = np.concatenate(all_nodes, axis=0)

    flat_y_centroids = [item for sublist in all_nodes_y_position for item in sublist]
    flat_x_centroids = [item for sublist in all_nodes_x_position for item in sublist]
    flat_z_centroids = [item for sublist in all_nodes_z_position for item in sublist]
    return all_nodes, flat_x_centroids, flat_y_centroids, flat_z_centroids


class WholeGraph(InMemoryDataset):
    url = "792fc26c03054f20a272d87209c924f1"

    def __init__(
        self,
        root,
        node_loader,
        num_cores,
        transform=None,
        pre_transform=None,
        pre_filter=None,
    ):

        self.node_loader = node_loader
        self.num_cores = num_cores

        super(WholeGraph, self).__init__(
            root, transform, pre_transform, pre_filter
        )  # Load processed data
        self.data, self.slices = torch.load(self.processed_paths[0])
        self.root = root
        self.transform = transform

    @property
    def raw_file_names(self):
        return ["nothing.csv"]

    @property
    def processed_file_names(self):
        return ["data.pt"]

    def download(self):
        fms = FileManagementSystem()
        record = fms.find_one_by_id(self.url)
        self.df = pd.read_csv(record.path)

        self.node_cols = filter_columns(self.df.columns.to_list(), **self.node_loader)

        self.df = self.df.loc[self.df["neighbors"].astype(str) != "nan"]

        norm_cols = ["centroid_x", "centroid_y", "centroid_z"] + self.node_cols
        self.df[norm_cols] = self.df[norm_cols].apply(
            lambda x: (x - x.min()) / (x.max() - x.min())
        )
        self.df = self.df.loc[self.df["is_outlier"].isin([False])]

        # df2 = self.df[self.node_cols].copy()
        # df1 = df2.loc[:, (df2 != 0).all()]
        # self.node_cols = df1.columns

        save_path = Path(self.root + "/data")
        save_path.mkdir(parents=True, exist_ok=True)
        csv_path = save_path / "input_to_graph.csv"
        if not csv_path.is_file():
            self.df.to_csv(csv_path)

    def process(self):

        node_cols = self.node_cols

        union_cols = list(
            set(
                [
                    "CellId",
                    "neighbors",
                    "centroid_x",
                    "centroid_y",
                    "centroid_z",
                    "T_index",
                    "track_id",
                ]
            ).union(set(node_cols))
        )

        df_col = self.df[union_cols].copy()

        print("Computing nodes...")
        all_nodes_list = parallelize_dataframe(
            df_col,
            get_nodes,
            num_cores=self.num_cores,
            subset=False,
            groupby_val="T_index",
            node_cols=node_cols,
        )

        nodes_list = all_nodes_list[0].tolist()

        flat_x_centroids = all_nodes_list[1].tolist()
        flat_y_centroids = all_nodes_list[2].tolist()
        flat_z_centroids = all_nodes_list[3].tolist()

        print("Computing spatial edges...")
        spatial_edges_list = parallelize_dataframe(
            df_col,
            get_edges,
            num_cores=self.num_cores,
            subset=False,
            groupby_val="T_index",
            node_cols=node_cols,
            flat_list=nodes_list,
            flat_x_centroids=flat_x_centroids,
            flat_y_centroids=flat_y_centroids,
            flat_z_centroids=flat_z_centroids,
        )

        print("Computing track edges...")
        track_edges_list = parallelize_dataframe(
            df_col,
            get_track_edges,
            num_cores=self.num_cores,
            subset=True,
            groupby_val="track_id",
            node_cols=node_cols,
            flat_list=nodes_list,
            flat_x_centroids=flat_x_centroids,
            flat_y_centroids=flat_y_centroids,
            flat_z_centroids=flat_z_centroids,
        )

        all_edges = np.concatenate([spatial_edges_list[0], track_edges_list[0]], axis=0)
        all_edge_attributes = np.concatenate(
            [spatial_edges_list[1], track_edges_list[1]], axis=0
        )

        all_nodes = torch.tensor(
            np.array(all_nodes_list[0]).astype(float), dtype=torch.float
        )
        all_edges = torch.tensor(np.array(all_edges).astype(float), dtype=torch.long)
        all_edge_attributes = torch.tensor(
            np.array(all_edge_attributes).astype(float), dtype=torch.float
        )

        all_edge_attributes = all_edge_attributes.unsqueeze(dim=1)

        data = Data(
            x=all_nodes,
            edge_index=all_edges.t().contiguous(),
            edge_attr=all_edge_attributes,
        )

        size = data.x.shape[0]
        X_train_val, X_test = train_test_split(
            np.array([i for i in range(size)]), test_size=0.15, random_state=42
        )
        X_train, X_val = train_test_split(X_train_val, test_size=0.15, random_state=42)

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

        self.data, self.slices = self.collate([data])
        torch.save((self.data, self.slices), self.processed_paths[0])
