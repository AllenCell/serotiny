from torchvision import transforms
from serotiny.utils import load_multiple

def load_transforms(transforms_config):
    if isinstance(transforms_config, dict):
        invocations = list(load_multiple(transforms_config).values())
    elif isinstance(transforms_config, list):
        invocations = load_multiple(transforms_config)
    elif transforms_config is None:
        invocations = None
    else:
        raise Exception(f"Can only load transforms from dicts or iterables, not {transforms_config}")

    if invocations is not None:
        return transforms.Compose(invocations)
