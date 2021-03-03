import glob
import tarfile
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from torchvision import transforms
from ..image import png_loader
from ...library.data import download_quilt_data
from ...library.data import load_data_loader, Load2DImage, LoadClass, LoadId
from .base_datamodule import BaseDataModule
from .utils import subset_channels


def make_manifest(dataset_path):
    cells = []
    for split in ["train", "test"]:
        _split_path = str((Path(dataset_path) / split) / "*")
        for structure_path in glob.glob(_split_path):
            _struct_path = str(Path(structure_path) / "*")
            structure = structure_path.split("/")[-1]
            for cell_img in glob.glob(_struct_path):
                cells.append(
                    dict(cellpath2dict(cell_img), structure=structure, split=split,
                         path=str(Path(cell_img).resolve()))
                )

    return pd.DataFrame(cells)

def cellpath2dict(path):
    cell = path.split("/")[-1]
    cell = cell.split(".")[0]
    cell = cell.split("_")
    return {
        cell[i*2]: cell[i*2 + 1]
        for i in range(len(cell)//2)
    }

class AICS_MNIST_DataModule(BaseDataModule):

    def __init__(
        self,
        config: dict,
        batch_size: int,
        num_workers: int,
        data_dir: str,
        x_label: str,
        y_label: str,
    ):

        self.channels = config["channels"]
        self.select_channels = config["select_channels"]
        self.num_channels = len(self.channels)

        self.channel_indexes, self.num_channels = subset_channels(
            channel_subset=self.select_channels,
            channels=self.channels,
        )

        super().__init__(
            config=config,
            batch_size=batch_size,
            num_workers=num_workers,
            transform_list=[
            ],
            train_transform_list=[
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
            ],
            x_label=x_label,
            y_label=y_label,
            data_dir=data_dir,
        )

        self.y_encoded_label = y_label + "_encoded"
        self.id_fields = config["id_fields"]
        self.num_classes = 10

        self.loaders = {
            # Use callable class objects here because lambdas aren't picklable
            "id": LoadId(self.id_fields),
            self.y_label: LoadClass(self.num_classes, self.y_encoded_label),
            self.x_label: Load2DImage(
                "path",
                self.num_channels,
                self.channel_indexes,
                self.transform,
            ),
        }

        # self.dims is returned when you call dm.size()
        # Setting default dims here because we know them.
        # Could optionally be assigned dynamically in dm.setup()
        self.dims = (28, 28)

    @classmethod
    def get_dataset(cls, data_dir):
        data_dir = Path(data_dir)

        if not (data_dir / "aics_mnist_rgb.tar.gz").exists():
            download_quilt_data(
                package="aics/aics_mnist",
                bucket="s3://allencell",
                data_save_loc=data_dir
            )

        if not (data_dir / "aics_mnist_rgb").exists():
            with tarfile.open(data_dir / "aics_mnist_rgb.tar.gz") as f:
                f.extractall(data_dir)

        if not (data_dir / "aics_mnist_rgb.csv").exists():
            manifest = make_manifest(data_dir / "aics_mnist_rgb")
            manifest["structure_encoded"] = LabelEncoder().fit_transform(manifest["structure"])
            manifest.to_csv(data_dir / "aics_mnist_rgb.csv", index=False)

    def setup(self, stage=None):
        self.data_dir = Path(self.data_dir)

        all_data = pd.read_csv(self.data_dir / "aics_mnist_rgb.csv")

        all_data[self.y_label] = all_data["structure"]
        all_data[self.y_encoded_label] = all_data["structure_encoded"]

        self.datasets["test"] = all_data.loc[all_data.split == "test"]

        train_val = all_data.loc[all_data.split == "train"]

        train_inds, val_inds = train_test_split(
            train_val.index,
            test_size=len(self.datasets["test"]),
            stratify=train_val.structure,
            random_state=42,
        )

        self.datasets["train"] = train_val.loc[train_inds]
        self.datasets["valid"] = train_val.loc[val_inds]
        self.datasets["valid"]["split"] = "valid"


        self.dims = (28, 28)

    def prepare_data(self):
        self.get_dataset(self.data_dir)

    def load_image(self, dataset):
        return png_loader(
            dataset["path"].iloc[0],
            channel_order="CYX",
            indexes={"C": self.channel_indexes or range(self.num_channels)},
            transform=self.transform,
        )

    def get_dims(self, img):
        return (img.shape[1], img.shape[2])

    def train_dataloader(self):
        train_dataset = self.datasets['train']
        train_loaders = self.loaders.copy()
        train_loaders[self.x_label] = Load2DImage(
            "path",
            self.num_channels,
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
            weights_col=None,
        )

        return train_dataloader
