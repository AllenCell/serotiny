import fire
import importlib
import inspect

from omegaconf import OmegaConf

def get_obj_from_path(path: str):
    """
    Given a class/function/variable path as a string (e.g. some.module.ClassName),
    retrieve the class/function/variable
    """
    path = path.split(".")
    module = ".".join(path[:-1])
    name = path[-1]
    return getattr(importlib.import_module(module), name)

def create_config(path):
    obj = get_obj_from_path(path)

    if hasattr(obj, "__init__"):
        sig = inspect.getfullargspec(obj.__init__)
    else:
        sig = inspect.getfullargspec(obj)

    args_dict = dict()
    args_dict["_target_"] = path

    if sig.defaults is not None:
        for ix, default_value in enumerate(sig.defaults):
            args_dict[sig.args[ix]] = default_value

    if sig.kwonlydefaults is not None:
        for ix, default_value in enumerate(sig.kwonlydefaults):
            args_dict[sig.kwonlyargs[ix]] = default_value

    for arg in (sig.args + sig.kwonlyargs):
        if arg not in args_dict:
            args_dict[arg] = None

    if "self" in args_dict:
        del args_dict["self"]

    print(OmegaConf.to_yaml(OmegaConf.create(args_dict)))

def main():
    fire.Fire(create_config)

if __name__ == "__main__":
    main()
