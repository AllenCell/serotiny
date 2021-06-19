from collections import defaultdict

from aicsfiles import FileManagementSystem

from serotiny.io.image import png_loader
from .abstract_loader import Loader
from .utils import load_transforms


class Load2DImage(Loader):
    """
    Loader class, used to retrieve images from paths given in a dataframe column
    """

    def __init__(
            self,
            column='image',
            num_channels=1,
            channel_indexes=None,
            transforms=None):

        super().__init__()
        self.column = column
        self.num_channels = num_channels
        self.channel_indexes = channel_indexes

        self.transforms = defaultdict(None)
        for key, transforms_config in transforms.items():
            self.transforms[key] = load_transforms(transforms_config)


    def __call__(self, row):
        return png_loader(
            row[self.column],
            channel_order="CYX",
            indexes={"C": self.channel_indexes or range(self.num_channels)},
            transform=self.transforms.get(self.mode)
        )
