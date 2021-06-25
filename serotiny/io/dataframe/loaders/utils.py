from torchvision import transforms
from serotiny.utils import path_invocations

def load_transforms(transforms_config):
    if isinstance(transforms_config, dict):
        invocations = list(path_invocations(transforms_config).values())
    elif isinstance(transforms_config, list):
        invocations = path_invocations(transforms_config)
    elif transforms_config is None:
        invocations = None
    else:
        raise Exception(f"Can only load transforms from dicts or iterables, not {transforms_config}")

    if invocations is not None:
        return transforms.Compose(invocations)
