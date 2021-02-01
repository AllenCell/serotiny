import pytorch_lightning as pl

# Note - you must have torchvision installed for this example
from torchvision import transforms
from os import listdir
from ..csv import load_csv
from ..image import png_loader
from pathlib import Path
from ...constants import DatasetFields
from ...library.data import load_data_loader, LoadImage, LoadClass, LoadId


class Mitotic2DDataModule(pl.LightningDataModule):

    def __init__(
        self, 
        config: dict,
        batch_size: int, 
        num_workers: int, 
        data_dir: str = './',
    ):
        super().__init__()

        self.data_dir = data_dir
        self.channels = config["channels"]
        self.channel_indexes = config["channel_indexes"]
        self.classes = config["classes"]
        self.id_fields = config["id_fields"]
        self.batch_size = batch_size
        self.num_workers = num_workers

        self._num_channels = len(self.channels)
        chosen_channels = self.channel_indexes
        self.channel_indexes = None

        if chosen_channels is not None:
            try:
                self.channel_indexes = [
                    self.channels.index(channel_name) for channel_name
                    in chosen_channels
                ]
                self._num_channels = len(self.channel_indexes)
            except ValueError:
                raise Exception(
                    (
                        f"channel indexes {self.channel_indexes} "
                        f"do not match channel names {self.channels}"
                    )
                )

        self.transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize(256),
                transforms.CenterCrop(256),
                transforms.ToTensor(),
            ]
        )
        self.train_transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.CenterCrop(256),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ]
        )

        self.x_label = "projection_image"
        self.y_label = "mitotic_class"

        self.loaders = {
            # Use callable class objects here because lambdas aren't picklable
            "id": LoadId(self.id_fields),
            self.y_label: LoadClass(len(self.classes)),
            self.x_label: LoadImage(
                DatasetFields.Chosen2DProjectionPath,
                self._num_channels,
                self.channel_indexes,
                self.transform,
            ),
        }
        
        # self.dims is returned when you call dm.size()
        # Setting default dims here because we know them.
        # Could optionally be assigned dynamically in dm.setup()
        self.dims = (1, 28, 28)

    def prepare_data(self):
        # download
        # Currently empty, we assume that data is saved as csv's
        pass

    def setup(self, stage=None):

        filenames = listdir(self.data_dir)
        dataset_splits = [split for split in filenames if split.endswith(".csv")]
        dataset_paths = [Path(self.data_dir + split) for split in dataset_splits]

        for path in dataset_paths:
            if not path.exists():
                raise Exception(f"not all datasets are present, missing {path}")

        self.datasets = {}
        for split_csv, path in zip(dataset_splits, dataset_paths):
            split = split_csv.split(".")[0]

            dataset = load_csv(path, [])

            self.datasets[split] = dataset

        # Load a test image to get image dimensions after transform
        test_image = png_loader(
            dataset[DatasetFields.Chosen2DProjectionPath].iloc[0],
            channel_order="CYX",
            indexes={"C": self.channel_indexes or range(self._num_channels)},
            transform=self.transform,
        )

        # Get test image dimensions
        dimensions = (test_image.shape[1], test_image.shape[2])

        self.dims = dimensions

    def train_dataloader(self):

        train_dataset = self.datasets['train']
        train_loaders = self.loaders.copy()
        train_loaders["projection_image"] = LoadImage(
            DatasetFields.Chosen2DProjectionPath,
            self._num_channels,
            self.channel_indexes,
            self.train_transform,
        )
        train_dataloader = load_data_loader(
            train_dataset,
            train_loaders,
            transform=self.train_transform,
            shuffle=False,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

        return train_dataloader

    def val_dataloader(self):

        val_dataset = self.datasets['valid']
        val_dataloader = load_data_loader(
            val_dataset,
            self.loaders,
            transform=self.transform,
            shuffle=False,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )
        return val_dataloader

    def test_dataloader(self):
        test_dataset = self.datasets['test']
        test_dataloader = load_data_loader(
            test_dataset,
            self.loaders,
            transform=self.transform,
            shuffle=False,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )
        return test_dataloader
