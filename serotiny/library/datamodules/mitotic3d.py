import pytorch_lightning as pl

# Note - you must have torchvision installed for this example
from torchvision import transforms
from os import listdir
from ..csv import load_csv
from ..image import png_loader, tiff_loader_CZYX
from pathlib import Path
from ...constants import DatasetFields
from ...library.data import load_data_loader, Load3DImage, LoadClass, LoadId
from .base_datamodule import BaseDataModule

from aicsimageprocessing.resize import resize_to

class Mitotic3DDataModule(BaseDataModule):

    def __init__(
        self,
        config: dict,
        x_label: str,
        y_label: str,
        batch_size: int,
        num_workers: int,
        data_dir: str = './',
    ):

        super().__init__(
            transform_list=[
                transforms.Lambda(lambda x: resize_to(x, (6, 64, 128, 96))),
                transforms.ToTensor()
            ],
            train_transform_list=[
                transforms.Lambda(lambda x: resize_to(x, (6, 64, 128, 96))),
                transforms.ToTensor()
            ],
            x_label="cell_image",
            y_label="mitotic_class"
        )

        self.loaders = {
            # Use callable class objects here because lambdas aren't picklable
            "id": LoadId(self.id_fields),
            self.y_label: LoadClass(len(self.classes)),
            self.x_label: Load3DImage(
                DatasetFields.CellImage3DPath,
                self._num_channels,
                self.channel_indexes,
                self.transform,
            ),
        }

    def load_image(self, dataset):
        return tiff_loader_CZYX(
            path_str=dataset[DatasetFields.CellImage3DPath].iloc[0],
            channel_indexes=self.channel_indexes,
            select_channels=None,
            output_dtype=np.float32,
            channel_masks=None,
            mask_thresh=0
        )

    def get_dims(self, img):
        return (img.shape[1], img.shape[2], img.shape[3])
