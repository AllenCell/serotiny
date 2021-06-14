from typing import Dict

import importlib

def module_get(module, key):
    if key not in module.__dict__:
        raise KeyError(
            f"Chosen {module} module {key} not available.\n"
            f"Available {module}(s):\n"
            f"{module.__all__}"
        )

    return module.__dict__[key]


def get_class_from_path(class_path: str):
    """
    Given a class path as a string (e.g. some.module.ClassName),
    retrieve the class
    """
    class_path = class_path.split(".")
    class_module = ".".join(class_path[:-1])
    class_name = class_path[-1]
    return getattr(importlib.import_module(class_module), class_name)


def get_classes_from_config(configs: Dict):
    """
    Return a list of instantiated classes given by `configs`. Each key in
    `configs` is a class path, to be imported dynamically via importlib,
    with arguments given by the correponding value in the dict.
    """
    instantiated_classes = []
    for class_path, class_config in configs.items():
        the_class = get_class_from_path(class_path)
        instantiated_class = the_class(**class_config)
        instantiated_classes.append(instantiated_class)

    return instantiated_classes
