from typing import Sequence, Optional
import numpy as np
from torchvision.transforms import Compose

from serotiny.io.image import image_loader
from serotiny.utils import load_multiple
from .abstract_loader import Loader


class Load2DImage(Loader):
    """
    Loader class, used to retrieve 2d images from paths given in a dataframe column
    """

    def __init__(
        self,
        column: str,
        file_type: str = "tiff",
        select_channels: Optional[Sequence] = None,
        transforms: Optional[Sequence] = None,
        reader: str = 'aicsimageio.readers.ome_tiff_reader.OmeTiffReader'
    ):
        """
        Parameters
        ----------
        column: str
            Dataframe column which contains the image path

        file_type: str
            File format of the image. For now, "tiff" is the
            only supported format. But in the future other formats (like zarr)
            might be supported too

        select_channels: Optional[Sequence] = None
            List of channels to include in the loaded image.

        transforms: Optional[Sequence] = None
            List of transforms to apply upon loading the image. The
            transforms are provided as a list of configuration dicts,
            loaded dynamically via serotiny's dynamic import utils

        """
        super().__init__()
        self.column = column
        self.select_channels = select_channels
        self.reader = reader

        if file_type not in ("tiff"):
            raise NotImplementedError(f"File type {file_type} not supported.")

        self.file_type = file_type

        if transforms is not None:
            transforms = load_multiple(transforms)
            transforms = Compose(transforms)
        self.transforms = transforms

    def __call__(self, row):
        if self.file_type == "tiff":
            return image_loader(
                row[self.column],
                select_channels=self.select_channels,
                output_dtype=np.float32,
                transform=self.transforms,
                reader=self.reader
            )
