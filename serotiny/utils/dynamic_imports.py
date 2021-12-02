import importlib
from functools import partial
from omegaconf import DictConfig

INVOKE_KEY = "^invoke"
BIND_KEY = "^bind"
INIT_KEY = "^init"


class _bind(partial):
    """
    An improved version of partial which accepts Ellipsis (...) as a placeholder
    """

    def __call__(self, *args, **keywords):
        keywords = {**self.keywords, **keywords}
        iargs = iter(args)
        args = (next(iargs) if arg is ... else arg for arg in self.args)
        return self.func(*args, *iargs, **keywords)


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


def get_name_from_path(path: str):
    """
    Given a class/function/variable path as a string (e.g. some.module.ClassName),
    retrieve the class/function/variable
    """
    path = path.split(".")
    module = ".".join(path[:-1])
    name = path[-1]
    return getattr(importlib.import_module(module), name)


def module_or_path(module, key):
    try:
        return module_get(module, key)
    except KeyError:
        return get_name_from_path(key)


def get_name_and_arguments(key, config):
    path = config[key]
    name = get_name_from_path(path)
    arguments = config.copy()
    del arguments[key]

    return name, arguments


def _try_load_zipped(zipped_tups, output):
    for key, value in zipped_tups:
        if isinstance(value, dict):
            try:
                output[key] = load_config(value)
            except:
                pass
    return output


def get_load_method_and_args(key, config, recurrent=True):
    to_load, arguments = get_name_and_arguments(key, config)

    positional_args = (arguments.pop("^positional_args")
                       if "^positional_args" in arguments else [])
    if recurrent:
        arguments = _try_load_zipped(arguments.items(), arguments)
        positional_args = _try_load_zipped(enumerate(positional_args), positional_args)

    return to_load, positional_args, arguments


def invoke(config, recurrent=True):
    to_invoke, positional_args, arguments = (
        get_load_method_and_args(INVOKE_KEY, config, recurrent)
    )
    return to_invoke(*positional_args, **arguments)


def init(config, recurrent=True):
    to_init, positional_args, arguments = (
        get_load_method_and_args(INIT_KEY, config, recurrent)
    )

    if not isinstance(to_init, type):
        raise TypeError(
            f"Expected {to_init} to be a class, but it is {type(to_init)}"
        )

    return to_init(*positional_args, **arguments)


def bind(config, recurrent=True):
    to_bind, positional_args, arguments = (
        get_load_method_and_args(BIND_KEY, config, recurrent)
    )
    return _bind(to_bind, *positional_args, **arguments)


def load_config(config, recurrent=True, loaded_ok=False):
    if hasattr(config, "items"):
        if BIND_KEY in config:
            return bind(config, recurrent)
        elif INIT_KEY in config:
            return init(config, recurrent)
        elif INVOKE_KEY in config:
            return invoke(config, recurrent)

    if not loaded_ok:
        raise KeyError(
            f"None of [{BIND_KEY}, {INVOKE_KEY}, {INIT_KEY}] found " f"in config."
        )
    return config



def load_multiple(configs, recurrent=True, loaded_ok=False):
    if hasattr(configs, "items"):
        return {key: load_config(config, recurrent, loaded_ok)
                for key, config in configs.items()}
    elif iter(configs):
        return [load_config(config, recurrent, loaded_ok)
                for config in configs]
    else:
        raise TypeError(
            f"can only bind/invoke/init paths from a dict or an "
            f"iterable, not {type(configs)}"
        )
