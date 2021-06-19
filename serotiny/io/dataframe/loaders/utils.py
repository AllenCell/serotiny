from torchvision import transforms
from serotiny.utils import get_classes_from_config

def load_transforms(transforms_dict):
    if transforms_dict is not None:
        return transforms.Compose(
            get_classes_from_config(transforms_dict)
        )
    return None
