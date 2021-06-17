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


def module_path(module, path):
    if not path:
        return module
    else:
        down = module_get(module, path[0])
        return module_path(down, path[1:])


def search_modules(modules, path):
    found = None
    for module in modules:
        try:
            found = module_path(module, path)
            break
        except KeyError:
            pass
    return found


def get_class_from_path(class_path: str):
    """
    Given a class path as a string (e.g. some.module.ClassName),
    retrieve the class
    """
    class_path = class_path.split(".")
    class_module = ".".join(class_path[:-1])
    class_name = class_path[-1]
    return getattr(importlib.import_module(class_module), class_name)


def invoke_path(name, config):
    invoke_class = get_class_from_path(name)
    return invoke_class(**config)


def module_or_path(module, key):
    try:
        return module_get(module, key)
    except KeyError:
        return get_class_from_path(key)


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

def keep(d: Dict, f: callable):
    return {
        key: value
        for key, value in d.items()
        if f(key, value)}

PATH_KEY = '^invoke'

def invoke_class(config):
    invoke = get_class_from_path(
        config[PATH_KEY])
    arguments = keep(
        config,
        lambda k, v: k != PATH_KEY)

    return invoke(**arguments)

def path_invocations(configs):
    if isinstance(configs, dict):
        return {
            key: invoke_class(config)
            for key, config in configs.items()}
    elif iter(configs):
        return [
            invoke_class(config)
            for config in configs]
    else:
        raise Exception(f"can only invoke paths from a dict or an iterable, not {configs}")
