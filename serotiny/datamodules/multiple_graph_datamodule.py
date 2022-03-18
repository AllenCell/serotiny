import logging
from typing import Union, Dict
from pathlib import Path

import pytorch_lightning as pl

from torch_geometric.loader import DataLoader

# from torch_geometric.data import GraphSAINTRandomWalkSampler

from serotiny.datamodules.datasets import WholeGraph

log = logging.getLogger(__name__)


class MultipleGraphDatamodule(pl.LightningDataModule):
    """A pytorch lightning datamodule that handles the logic for iterating over a folder
    of files.

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
    ):

        super().__init__()

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.save_dir = save_dir
        self.model_type = model_type
        self.node_loader = node_loader
        self.subset = subset
        # To delete later, this is only to get length

        dataset = WholeGraph(
            root=save_dir,
            node_loader=node_loader,
            num_cores=num_cores,
            single_graph=False,
            task_dict=task_dict,
            val_manifest=validated_manifest,
            train_test_split=False,
            subset=subset,
        )

        self.dataset = dataset
        self.length = len(dataset)

        print()
        print(f"Dataset: {dataset}:")
        print("======================")
        print(f"Number of graphs: {len(dataset)}")
        print(f"Number of features: {dataset[0].x.shape[1]}")
        print(f"Number of classes: {dataset.num_classes}")

        val_size = int((len(self.dataset) - int(0.8 * len(self.dataset))) / 2)

        self.train_dataset = self.dataset[: int(0.8 * len(self.dataset))]
        self.val_dataset = self.dataset[
            int(0.8 * len(self.dataset)) : int(0.8 * len(self.dataset)) + val_size
        ]
        self.test_dataset = self.dataset[int(0.8 * len(self.dataset)) + val_size :]

        print(f"Number of training graphs: {len(self.train_dataset)}")
        print(f"Number of val graphs: {len(self.val_dataset)}")
        print(f"Number of test graphs: {len(self.test_dataset)}")

        # self.dataloader = self.make_dataloader()

    def make_dataloader(self, dataset, shuffle):

        return DataLoader(dataset, self.batch_size, shuffle)

    def train_dataloader(self):
        return self.make_dataloader(self.train_dataset, True)

    def val_dataloader(self):
        return self.make_dataloader(self.val_dataset, False)

    def test_dataloader(self):
        return self.make_dataloader(self.test_dataset, False)
