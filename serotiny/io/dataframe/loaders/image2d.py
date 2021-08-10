from collections import defaultdict

# from aicsfiles import FileManagementSystem

from serotiny.io.image import png_loader
from .abstract_loader import Loader
from .utils import load_transforms


class Load2DImage(Loader):
    """
    Loader class, used to retrieve images from paths given in a dataframe column
    """

    def __init__(
        self, column="image", num_channels=1, channel_indexes=None, transforms=None
    ):
        # fms=False):

        super().__init__()
        self.column = column
        self.num_channels = num_channels
        self.channel_indexes = channel_indexes
        self.transforms = load_transforms(transforms)

        # self.fms = (FileManagementSystem() if fms else None)

    def _get_path(self, row):
        # if self.fms is not None:
        #     return self.fms.get_file_by_id(row[self.column]).path
        return row[self.column]

    def __call__(self, row):
        return png_loader(
            self._get_path(row),
            channel_order="CYX",
            indexes={"C": self.channel_indexes or range(self.num_channels)},
            transform=self.transforms,
        )
