import logging
from typing import Union, Dict
from pathlib import Path
import ast
from serotiny.io.dataframe.utils import filter_columns

import pytorch_lightning as pl
import numpy as np
import torch
from torch_geometric.utils import degree
from sklearn.preprocessing import MinMaxScaler
from torch_geometric.loader import GraphSAINTRandomWalkSampler, RandomNodeSampler, DataLoader

# from torch_geometric.data import GraphSAINTRandomWalkSampler

from serotiny.datamodules.datasets import WholeGraph

log = logging.getLogger(__name__)


class MultipleMovieDatamodule(pl.LightningDataModule):
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
        save_dir1: str,
        validated_manifest1: Union[Path, str],
        url1: str,
        save_dir2: str,
        validated_manifest2: Union[Path, str],
        url2: str,
        save_dir3: str,
        validated_manifest3: Union[Path, str],
        url3: str,
        node_loader: Dict,
        task_dict: dict,
        subset: bool,
        normalize_input: bool,
        normalize_target: bool,
        min_data: int,
        max_data: int,
        update_values: bool,
        single_graph: bool,
        dataloader_type: str,
    ):

        super().__init__()

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.save_dir1 = save_dir1
        self.save_dir2 = save_dir2
        self.save_dir3 = save_dir3
        self.node_loader = node_loader
        self.subset = subset
        self.min_data = min_data
        self.max_data = max_data
        self.update_values = update_values
        self.single_graph = single_graph
        self.dataloader_type = dataloader_type
        # To delete later, this is only to get length

        log.info('Computing all datasets')
        self.train_dataset = WholeGraph(
            root=save_dir1,
            node_loader=node_loader,
            num_cores=num_cores,
            single_graph=self.single_graph,
            task_dict=task_dict,
            url=url1,
            val_manifest=validated_manifest1,
            train_test_split=False,
            subset=subset,
            normalize=normalize_input,
        )
        log.info('Finished computing dataset 1')

        self.val_dataset = WholeGraph(
            root=save_dir2,
            node_loader=node_loader,
            num_cores=num_cores,
            single_graph=self.single_graph,
            task_dict=task_dict,
            url=url2,
            val_manifest=validated_manifest2,
            train_test_split=False,
            subset=subset,
            normalize=normalize_input,
        )
        log.info('Finished computing dataset 2')

        self.test_dataset = WholeGraph(
            root=save_dir3,
            node_loader=node_loader,
            num_cores=num_cores,
            single_graph=self.single_graph,
            task_dict=task_dict,
            url=url3,
            val_manifest=validated_manifest3,
            train_test_split=False,
            subset=subset,
            normalize=normalize_input,
        )
        log.info('Finished computing dataset 3')

        if self.update_values:
            if not self.single_graph:
                log.info('Updating multiple graph dataset values')
                self.train_dataset = get_multiple_graph_data(
                    self.train_dataset, node_loader, self.train_dataset.mask_df, self.train_dataset.val_manifest, 
                    task_dict, self.min_data, self.max_data
                )
                log.info('Finished updating dataset1 values') 

                self.val_dataset = get_multiple_graph_data(
                    self.val_dataset, node_loader, self.val_dataset.mask_df, self.val_dataset.val_manifest, 
                    task_dict, self.min_data, self.max_data
                )
                log.info('Finished updating dataset2 values') 
                self.test_dataset = get_multiple_graph_data(
                    self.test_dataset, node_loader, self.test_dataset.mask_df, self.test_dataset.val_manifest, 
                    task_dict, self.min_data, self.max_data
                )
                log.info('Finished updating dataset3 values') 
            else:
                log.info('Updating single graph dataset values')
                self.train_dataset, scaler_input, scaler_target = get_single_graph_data(self.train_dataset, node_loader, self.train_dataset.mask_df, 
                task_dict, save_dir1, self.train_dataset.val_manifest, normalize_input, normalize_target, None, None)
                log.info('Finished updating dataset1 values') 

                self.val_dataset, _, _ = get_single_graph_data(self.val_dataset, node_loader, self.val_dataset.mask_df, 
                task_dict, save_dir2, self.val_dataset.val_manifest, normalize_input, normalize_target, scaler_input, scaler_target)
                log.info('Finished updating dataset2 values') 

                self.test_dataset, _, _ = get_single_graph_data(self.test_dataset, node_loader, self.test_dataset.mask_df, 
                task_dict, save_dir3, self.test_dataset.val_manifest, normalize_input, normalize_target, scaler_input, scaler_target)
                log.info('Finished updating dataset3 values') 
        else:
            if self.single_graph:
                self.train_dataset = self.train_dataset[0]
                self.val_dataset = self.val_dataset[0]
                self.test_dataset = self.test_dataset[0]
  

        print(f"Number of training graphs: {len(self.train_dataset)}")
        print(f"Number of val graphs: {len(self.val_dataset)}")
        print(f"Number of test graphs: {len(self.test_dataset)}")

    def make_dataloader(self, dataset, shuffle, save_dir):
        if self.single_graph:
            if self.dataloader_type == "graph_saint":
                save_path = Path(save_dir + f"graph_saint")
                save_path.mkdir(parents=True, exist_ok=True)
                return GraphSAINTRandomWalkSampler(
                    dataset,
                    batch_size=self.batch_size,
                    walk_length=2,
                    num_steps=20,
                    sample_coverage=100,
                    save_dir=save_path,
                    num_workers=self.num_workers,
                )
            else:
                return RandomNodeSampler(dataset, num_parts=1)
        else:
            return DataLoader(dataset, self.batch_size, shuffle)

    def train_dataloader(self):
        return self.make_dataloader(self.train_dataset, False, self.save_dir1)

    def val_dataloader(self):
        return self.make_dataloader(self.val_dataset, False, self.save_dir2)

    def test_dataloader(self):
        return self.make_dataloader(self.test_dataset, False, self.save_dir3)

def get_single_graph_data(dataset, node_loader, df, task_dict, save_dir, df_val, normalize_input, normalize_target, 
    scaler_input=None, scaler_target=None):
    data = dataset[0]
    node_cols = filter_columns(df.columns.to_list(), **node_loader)
    target_label = task_dict["target_label"]
    if normalize_input:
        if scaler_input is None:
            scaler_input = MinMaxScaler()
            scaler_input.fit(df[node_cols])
        df[node_cols] = scaler_input.transform(df[node_cols])
        
    if normalize_target:
        if scaler_target is None:
            scaler_target = MinMaxScaler()
            scaler_target.fit(df[[target_label]])
        df[[target_label]] = scaler_target.transform(df[[target_label]])

    if df_val is not None:

        df_val["mask"] = True

        df_val_mask = df_val[
            ["CellId", "mask"]
        ]
        df_val_mask = df_val_mask.replace({np.NaN: False})

        df = df.drop(
            columns=["mask"]
        )
        df = df.merge(
            df_val_mask, how="left", left_on="CellId", right_on="CellId"
        )
        df = df.replace({np.NaN: False})

        data.mask = torch.tensor(
            df["mask"].values, dtype=torch.bool
        )
    else:
        pass
    # adding data.x again in case we are using the same dataset
    # but want to change the node cols
    # import ipdb
    # ipdb.set_trace()
    data.x = torch.tensor(
        df[node_cols].values.astype(np.float64), dtype=torch.float,
    )

    save_path = Path(save_dir + "/data")
    csv_path_mask = save_path / "input_to_graph_with_masks.csv"

    if not csv_path_mask.is_file():
        df.to_csv(csv_path_mask)

    # Do this again in case we want to load new columns
    if task_dict["task"] == "classification":
        num_bins = int(task_dict["num_bins"])

        df[target_label + "_int"] = pd.qcut(
            df[target_label],
            q=[i / (num_bins) for i in range(num_bins)] + [1],
            labels=False,
        )
        df[target_label + "_int"] = df[
            target_label + "_int"
        ].fillna(num_bins)
        targets = torch.tensor(
            df[target_label + "_int"], dtype=torch.long
        )
    elif task_dict["task"] == "regression":
        df[target_label] = df[target_label].fillna(10)
        targets = torch.tensor(df[target_label], dtype=torch.float)

    data.y = targets
    data.cellid = df['CellId'].values
    weights = torch.tensor(
        [
            len(data.y) / i
            for i in np.unique(np.array(data.y), return_counts=True)[1]
        ]
    )
    data.weights = weights
    row, col = data.edge_index
    data.edge_weight = 1.0 / degree(col, data.num_nodes)[col]
    return data, scaler_input, scaler_target


def get_multiple_graph_data(dataset, node_loader, df, df_val, task_dict, min_data=0, max_data=1000):
    all_data = []
    node_cols = filter_columns(df.columns.to_list(), **node_loader)
    target_cols = task_dict["target_label"]
    df[node_cols] = df[node_cols].apply(
        lambda x: (x - x.min()) / (x.max() - x.min())
    )

    num_true = 0
    for index, data in enumerate(dataset):
        if min_data < data.x.shape[0] < max_data:
            df3 = df.loc[df['T_index'] == index]
            labels, nodes, mask = [], [], []
            for index2, row in df3.iterrows():
                try:
                    neighbor_ids =  ast.literal_eval(row['neighbors']) 
                except:
                    neighbor_ids = None
                if neighbor_ids is not None:
                    node = list(row[node_cols])
                    y = [row[target_cols]]
                    labels.append(y)
                    nodes.append(node)
                    if row['CellId'] in df_val['CellId'].values:
                        mask.append(True)
                        num_true += 1
                    else:
                        mask.append(False)
            nodes = np.asarray(nodes)
            labels = np.asarray(labels)
            mask = np.asarray(mask)
            if len(nodes.shape) == 2 :
                assert data.x.shape[0] == nodes.shape[0]
                assert data.y.shape[0] == labels.shape[0]
                data.x = torch.tensor(nodes, dtype=torch.float)
                data.y = torch.tensor(labels, dtype=torch.float)
                data.mask = torch.tensor(mask, dtype=torch.bool)
                all_data.append(data)
    return all_data