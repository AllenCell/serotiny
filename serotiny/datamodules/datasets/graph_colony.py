import logging
from pathlib import Path

import numpy as np
import pandas as pd
import ast
import multiprocessing
from functools import partial
import tqdm
import torch

from torch_geometric.utils import degree

from torch_geometric.data import Data, InMemoryDataset
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from serotiny.io.dataframe.utils import filter_columns

from aicsfiles import FileManagementSystem
from serotiny_steps.fit_pca import _fit_pca as fit_pca

from pathlib import Path
import joblib

log = logging.getLogger(__name__)


class WholeGraph(InMemoryDataset):
    # url = "792fc26c03054f20a272d87209c924f1"
    def __init__(
        self,
        root,
        node_loader,
        num_cores,
        single_graph,
        task_dict,
        url,
        val_manifest=None,
        train_test_split=None,
        subset=None,
        transform=None,
        pre_transform=None,
        pre_filter=None,
        normalize=None,
    ):

        self.node_loader = node_loader
        self.num_cores = num_cores
        self.single_graph = single_graph
        self.task_dict = task_dict
        self.val_manifest = val_manifest
        self.train_test_split = train_test_split
        self.subset = subset
        self.normalize = normalize
        self.url = url

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
        try:
            fms = FileManagementSystem()
            record = fms.find_one_by_id(self.url)
            self.df = pd.read_csv(record.path)
        except:
            self.df = pd.read_csv(self.url)

        self.node_cols = filter_columns(self.df.columns.to_list(), **self.node_loader)

        self.df = self.df.loc[self.df["neighbors"].astype(str) != "nan"].reset_index(drop=True)

        # self.df = self.df.loc[self.df["is_outlier"].isin([False])]

        pca_df = Path("./") / "pca_df.csv"
        fitted_pca = Path("./") / "fitted_pca.joblib"

        # Compute PCA on spharm

        # spharm_cols_filter = dict(startswith="NUC_", contains="shcoeff")

        # cols = filter_columns(self.df.columns, **spharm_cols_filter)

        # df2 = self.df[cols].copy()
        # df1 = df2.loc[:, (df2 != 0).all()]
        # spharm_cols = df1.columns

        # fit_pca(self.df[spharm_cols], fitted_pca, 8, spharm_cols_filter)

        # fitted_pca = joblib.load(fitted_pca)
        # pca_df = pd.DataFrame(fitted_pca.transform(self.df[spharm_cols]))
        # pca_df.columns = ["pc_" + str(i) for i in pca_df.columns]
        # pca_df["CellId"] = self.df["CellId"]

        # self.df = self.df.merge(pca_df, on="CellId")

        if (self.val_manifest is not None) & (self.val_manifest != "None"):
            self.val_manifest = pd.read_csv(self.val_manifest)
            size = self.val_manifest.shape[0]

            if self.train_test_split:
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

                self.val_manifest["test_mask"] = test_mask
                self.val_manifest["val_mask"] = val_mask
                self.val_manifest["train_mask"] = train_mask
            else:
                mask = np.array([True for i in range(size)])
                self.val_manifest["mask"] = mask
            cols_to_use = list(self.val_manifest.columns.difference(self.df.columns))
            cols_to_use.append('CellId')
            main_manifest = self.df.merge(
                self.val_manifest[cols_to_use], how="left", left_on="CellId", right_on="CellId"
            )

            if self.train_test_split:
                main_manifest["test_mask"] = main_manifest["test_mask"].fillna(False)
                main_manifest["train_mask"] = main_manifest["train_mask"].fillna(False)
                main_manifest["val_mask"] = main_manifest["val_mask"].fillna(False)
            else:
                main_manifest["mask"] = main_manifest["mask"].fillna(False)
        else:
            main_manifest = self.df

        self.target_label = self.task_dict["target_label"]
        if self.task_dict["task"] == "classification":
            num_bins = int(self.task_dict["num_bins"])

            main_manifest[self.target_label + "_int"] = pd.qcut(
                main_manifest[self.target_label],
                q=[i / (num_bins) for i in range(num_bins)] + [1],
                labels=False,
            )
            main_manifest[self.target_label + "_int"] = main_manifest[
                self.target_label + "_int"
            ].fillna(num_bins)
        elif self.task_dict["task"] == "regression":
            main_manifest[self.target_label] = main_manifest[self.target_label].fillna(
                10
            )
            # main_manifest[[self.target_label]] = main_manifest[[self.target_label]].apply(
            #     lambda x: (x - x.min()) / (x.max() - x.min())
            # )

        if self.normalize:
            norm_cols = ["centroid_x", "centroid_y", "centroid_z"] + self.node_cols
        else:
            norm_cols = ["centroid_x", "centroid_y", "centroid_z"]
        self.df[norm_cols] = self.df[norm_cols].apply(
            lambda x: (x - x.min()) / (x.max() - x.min())
        )

        self.mask_df = main_manifest
        # norm_cols = ["T_index_x"]
        # self.mask_df[norm_cols] = self.mask_df[norm_cols].apply(
        #     lambda x: (x - x.min()) / (x.max() - x.min())
        # )

        save_path = Path(self.root + "/data")
        save_path.mkdir(parents=True, exist_ok=True)
        csv_path = save_path / "input_to_graph.csv"
        if not csv_path.is_file():
            self.df.to_csv(csv_path)

        csv_path_mask = save_path / "input_to_graph_with_masks.csv"
        if not csv_path_mask.is_file():
            self.mask_df.to_csv(csv_path_mask)

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

        if self.single_graph:

            print("Computing nodes...")
            all_nodes_list = parallelize_dataframe(
                df_col,
                get_nodes,
                num_cores=self.num_cores,
                subset=False,
                groupby_val="T_index",
                node_cols=node_cols,
            )
            # all_nodes = np.concatenate(all_nodes, axis=0)
            all_nodes_multiple_graphs = all_nodes_list[0]
            all_nodes_single_graph = np.concatenate(all_nodes_multiple_graphs, axis=0)
            nodes_list = all_nodes_single_graph.tolist()

            flat_x_centroids = all_nodes_list[1].tolist()
            flat_y_centroids = all_nodes_list[2].tolist()
            flat_z_centroids = all_nodes_list[3].tolist()

            print("Computing spatial edges...")
            spatial_edges_list = parallelize_dataframe(
                df_col,
                get_edges,
                num_cores=self.num_cores,
                subset=self.subset,
                groupby_val="T_index",
                node_cols=node_cols,
                flat_list=nodes_list,
                flat_x_centroids=flat_x_centroids,
                flat_y_centroids=flat_y_centroids,
                flat_z_centroids=flat_z_centroids,
            )

            all_edges_multiple_graphs = spatial_edges_list[0]
            all_edge_attr_multiple_graphs = spatial_edges_list[1]
            all_edges_single_graph = np.concatenate(all_edges_multiple_graphs, axis=0)
            all_edge_attr_single_graph = np.concatenate(
                all_edge_attr_multiple_graphs, axis=0
            )

            print("Computing track edges...")
            track_edges_list = parallelize_dataframe(
                df_col,
                get_track_edges,
                num_cores=self.num_cores,
                subset=self.subset,
                groupby_val="track_id",
                node_cols=node_cols,
                flat_list=nodes_list,
                flat_x_centroids=flat_x_centroids,
                flat_y_centroids=flat_y_centroids,
                flat_z_centroids=flat_z_centroids,
            )

            all_track_edges_multiple_graphs = track_edges_list[0]
            all_track_edge_attr_multiple_graphs = track_edges_list[1]
            all_track_edges_single_graph = np.concatenate(
                all_track_edges_multiple_graphs, axis=0
            )
            all_track_edge_attr_single_graph = np.concatenate(
                all_track_edge_attr_multiple_graphs, axis=0
            )

            # all_edges = np.concatenate(
            #     [all_edges_single_graph, all_track_edges_single_graph], axis=0
            # )
            # all_edge_attributes = np.concatenate(
            #     [all_edge_attr_single_graph, all_track_edge_attr_single_graph], axis=0
            # )

            all_edges = np.concatenate([all_edges_single_graph], axis=0)
            all_edge_attributes = np.concatenate(
                [all_edge_attr_single_graph], axis=0
            )

            all_nodes = torch.tensor(
                np.array(all_nodes_single_graph).astype(float), dtype=torch.float
            )
            all_edges = torch.tensor(
                np.array(all_edges).astype(float), dtype=torch.long
            )
            all_edge_attributes = torch.tensor(
                np.array(all_edge_attributes).astype(float), dtype=torch.float
            )

            all_edge_attributes = all_edge_attributes.unsqueeze(dim=1)

            data = Data(
                x=all_nodes,
                edge_index=all_edges.t().contiguous(),
                edge_attr=all_edge_attributes,
            )

            data_list = [data]
        else:
            print("Computing nodes and edges...")
            nodes_edges_list = parallelize_dataframe(
                df_col,
                get_nodes_and_edges,
                num_cores=self.num_cores,
                subset=self.subset,
                groupby_val="T_index",
                node_cols=node_cols,
            )
            all_nodes = nodes_edges_list[0]
            all_edges = nodes_edges_list[1]
            all_edge_attr = nodes_edges_list[2]
            all_splits = nodes_edges_list[3]
            combo_list = [all_nodes, all_edges, all_edge_attr]

            for j, list2 in enumerate(combo_list):
                zipped_lists = zip(all_splits, list2)
                sorted_pairs = sorted(zipped_lists)
                tuples = zip(*sorted_pairs)
                tmp_splits, combo_list[j] = [list(tuple) for tuple in tuples]

            all_nodes = combo_list[0]
            all_edges = combo_list[1]
            all_edge_attr = combo_list[2]
            all_splits = tmp_splits

            data_list = []

            for graph in range(len(all_nodes)):
                nodes = all_nodes[graph]
                edges = all_edges[graph]
                edge_attr = all_edge_attr[graph]
                split = all_splits[graph]

                nodes = torch.tensor(nodes, dtype=torch.float)
                edges = torch.tensor(edges, dtype=torch.long)
                edge_attr = torch.tensor(edge_attr, dtype=torch.float)

                data = Data(
                    x=nodes, edge_index=edges.t().contiguous(), edge_attr=edge_attr
                )

                try:
                    _, col = data.edge_index
                    data.edge_weight = (
                        1.0 / degree(col, data.num_nodes)[col]
                    )  # Norm by in-degree.
                except:
                    pass

                if self.val_manifest is not None:
                    main_manifest = self.mask_df

                    # df_sub = main_manifest.loc[main_manifest["T_index" + "_x"] == split]
                    df_sub = main_manifest.loc[main_manifest["T_index"] == split]
                    df_sub = df_sub.reset_index()
                    data.mask = torch.tensor(df_sub["mask"].values, dtype=torch.bool)

                if self.task_dict["task"] == "classification":
                    targets = torch.tensor(
                        df_sub[self.target_label + "_int"], dtype=torch.long
                    )
                elif self.task_dict["task"] == "regression":
                    targets = torch.tensor(df_sub[self.target_label], dtype=torch.float)
                data.y = targets

                data_list.append(data)

        self.data, self.slices = self.collate(data_list)
        torch.save((self.data, self.slices), self.processed_paths[0])


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
        if df3.shape[0] > 1:
            lags = 1
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

    flat_y_centroids = [item for sublist in all_nodes_y_position for item in sublist]
    flat_x_centroids = [item for sublist in all_nodes_x_position for item in sublist]
    flat_z_centroids = [item for sublist in all_nodes_z_position for item in sublist]
    return all_nodes, flat_x_centroids, flat_y_centroids, flat_z_centroids


def get_nodes_and_edges(dataframe, groupby_val, node_cols, groupby_split):

    dataframe = dataframe.set_index("CellId")
    multiple_graph_nodes, multiple_graph_edges = [], []
    multiple_graph_edge_attr = []
    all_splits = []
    for split in groupby_split:
        all_nodes, all_nodes_list = [], []
        all_edges, all_edge_attr = [], []
        all_nodes_x_position, all_nodes_y_position, all_nodes_z_position = [], [], []
        all_x, all_y, all_z = [], [], []
        df3 = dataframe.loc[dataframe[groupby_val] == split]
        nodes = []
        nodes_x_position, nodes_y_position, nodes_z_position = [], [], []
        for index, row in df3.iterrows():
            try:
                neighbor_ids = ast.literal_eval(row["neighbors"])
            except:
                neighbor_ids = None
            if neighbor_ids is not None:
                node = list(row[node_cols])
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
        flat_list = [item for sublist in all_nodes_list for item in sublist]

        all_x.append(nodes_x_position)
        all_y.append(nodes_y_position)
        all_z.append(nodes_z_position)

        flat_x_centroids = [item for sublist in all_x for item in sublist]
        flat_y_centroids = [item for sublist in all_y for item in sublist]
        flat_z_centroids = [item for sublist in all_z for item in sublist]

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

        if len(edges) == 0:
            edges.append([0, 0])
            edge_attr.append(0)

        edges = np.asarray(edges)
        edge_attr = np.asarray(edge_attr)

        all_edges.append(edges)
        all_edge_attr.append(edge_attr)
        all_nodes = np.concatenate(all_nodes, axis=0)
        all_edges = np.concatenate(all_edges, axis=0)
        all_edge_attr = np.concatenate(all_edge_attr, axis=0)
        all_splits.append(split)
        multiple_graph_nodes.append(all_nodes)
        multiple_graph_edges.append(all_edges)
        multiple_graph_edge_attr.append(all_edge_attr)

    return (
        multiple_graph_nodes,
        multiple_graph_edges,
        multiple_graph_edge_attr,
        all_splits,
    )
