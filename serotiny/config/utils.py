import fire
import importlib
import inspect

from omegaconf import OmegaConf
from omegaconf._utils import is_primitive_type

def get_obj_from_path(path: str):
    """
    Given a class/function/variable path as a string (e.g. some.module.ClassName),
    retrieve the class/function/variable
    """
    path = path.split(".")
    module = ".".join(path[:-1])
    name = path[-1]
    return getattr(importlib.import_module(module), name)

def _create_config(obj, path=None, add_partial=False):

    if path is None:
        if hasattr(obj, "__name__"):
            path = obj.__module__ + "." + obj.__name__
        else:
            path = obj.__module__ + "." + obj.__class__.__name__

    if hasattr(obj, "__init__"):
        sig = inspect.getfullargspec(obj.__init__)
    else:
        sig = inspect.getfullargspec(obj)

    args_dict = dict()
    if add_partial:
        args_dict["_partial_"] = True
    args_dict["_target_"] = path

    args = sig.args
    if sig.defaults is not None:
        while len(args) > len(sig.defaults):
            args_dict[args.pop(0)] = None

        assert len(args) == len(sig.defaults)
        for arg, default_value in zip(args, sig.defaults):
            args_dict[arg] = default_value

    kwargs = sig.kwonlyargs
    if sig.kwonlydefaults is not None:
        while len(kwargs) > len(sig.kwonlydefaults):
            args_dict[kwargs.pop(0)] = None

        assert len(args) == len(sig.defaults)
        for arg, default_value in zip(kwargs, sig.kwonlydefaults):
            args_dict[arg] = default_value

    for arg in (sig.args + sig.kwonlyargs):
        if arg not in args_dict:
            args_dict[arg] = None

    if "self" in args_dict:
        del args_dict["self"]

    for arg, default in args_dict.items():
        if isinstance(default, tuple):
            args_dict[arg] = list(default)
        elif not is_primitive_type(default):
            try:
                args_dict[arg] = _create_config(
                    default,
                    add_partial=inspect.isclass(default)
                )
            except Exception:
                args_dict[arg] = f"Unable to give default value of {type(default)}"

    return args_dict

def create_config(path):
    obj = get_obj_from_path(path)
    config = _create_config(obj, path)
    print(OmegaConf.to_yaml(OmegaConf.create(config)))

def main():
    fire.Fire(create_config)

if __name__ == "__main__":
    main()
