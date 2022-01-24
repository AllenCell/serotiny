import fire
import importlib
import inspect

from omegaconf import OmegaConf
from omegaconf._utils import is_primitive_type, split_key

def get_obj_from_path(obj_path: str):
    """
    Given a class/function/variable path as a string (e.g. some.module.ClassName),
    retrieve the class/function/variable
    """
    obj_path = obj_path.split(".")
    module = ".".join(obj_path[:-1])
    name = obj_path[-1]
    return getattr(importlib.import_module(module), name)

def _create_config(obj, obj_path=None, add_partial=False):

    if obj_path is None:
        if hasattr(obj, "__name__"):
            obj_path = obj.__module__ + "." + obj.__name__
        else:
            obj_path = obj.__module__ + "." + obj.__class__.__name__

    if hasattr(obj, "__init__"):
        sig = inspect.getfullargspec(obj.__init__)
    else:
        sig = inspect.getfullargspec(obj)

    args_dict = dict()
    if add_partial:
        args_dict["_partial_"] = True
    args_dict["_target_"] = obj_path

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

def create_config(obj_path):
    obj = get_obj_from_path(obj_path)
    config = _create_config(obj, obj_path)
    return OmegaConf.to_yaml(OmegaConf.create(config))


def _maybe_int(value):
    return int(value) if value.isdigit() else value


def add_config(obj_path, dest):
    dest_key = None
    if ":" in dest:
        assert len([c for c in dest if c == ":"]) == 1
        dest, dest_key = dest.split(":")

    obj = get_obj_from_path(obj_path)
    config = _create_config(obj, obj_path)

    if dest_key is not None:
        dest_config = OmegaConf.load(dest)
        dest_path = split_key(dest_key)
        for step in dest_path[::-1]:
            config = {step:config}

        dest_config = OmegaConf.merge(dest_config, config)

    else:
        dest_config = config


    prompt_str_suffix = "" if dest_key is None else f":{dest_key}"
    answer = input(f"This is going to overwrite {dest}{prompt_str_suffix}"
                    "\nContinue? [Y/n] ")
    if answer is None or answer.lower().strip() in ["", "yes", "y"]:
        with open(dest, "w") as f:
            f.write(OmegaConf.to_yaml(dest_config))
    else:
        print("Not writing the result. The resulting config would have been:\n")
        print(OmegaConf.to_yaml(dest_config))
