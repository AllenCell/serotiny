from .apply_transforms import transform_batch
from ..omegaconf_decorator import omegaconf_decorator


class ImageCLI:
    def __init__(self):
        self.transform = omegaconf_decorator(transform_batch, "transforms_to_apply")
