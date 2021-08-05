from typing import Dict

import importlib
from inspect import isfunction
from functools import partial

INVOKE_KEY = '^invoke'
BIND_KEY = '^bind'
INIT_KEY = '^init'


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
    arguments = {k:v for k,v in config.items() if k != key}
    return name, arguments



def invoke(config):
    to_invoke, arguments = get_name_and_arguments(INVOKE_KEY, config)

    for key, value in arguments.items():
        if isinstance(value, dict):
            try:
                arguments[key] = load_config(value)
            except:
                pass

    return to_invoke(**arguments)


def init(config):
    to_init, arguments = get_name_and_arguments(INIT_KEY, config)
    if not isinstance(to_init, type):
        raise TypeError(f"Expected {to_init} to be a class, but it is "
                        f"{type(to_init)}")

    for key, value in arguments.items():
        if isinstance(value, dict):
            try:
                arguments[key] = load_config(value)
            except:
                pass

    return to_init(**arguments)


def bind(config):
    to_bind, arguments = get_name_and_arguments(BIND_KEY, config)

    for key, value in arguments.items():
        if isinstance(value, dict):
            try:
                arguments[key] = load_config(value)
            except:
                pass

    return _bind(to_bind, **arguments)


def load_config(config):
    if BIND_KEY in config:
        return bind(config)
    elif INIT_KEY in config:
        return init(config)
    elif INVOKE_KEY in config:
        return invoke(config)
    else:
        raise KeyError(f"None of [{BIND_KEY}, {INVOKE_KEY}, {INIT_KEY}] found "
                       f"in config.")


def load_multiple(configs):
    if isinstance(configs, dict):
        return {
            key: load_config(config)
            for key, config in configs.items()
        }
    elif iter(configs):
        return [
            load_config(config)
            for config in configs
        ]
    else:
        raise TypeError(f"can only bind/invoke/init paths from a dict or an "
                        f"iterable, not {type(configs)}")
