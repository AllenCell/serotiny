import importlib
from functools import partial

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

    positional_args = (
        arguments.pop("^positional_args") if "^positional_args" in arguments else []
    )
    if recurrent:
        arguments = _try_load_zipped(arguments.items(), arguments)
        positional_args = _try_load_zipped(enumerate(positional_args), positional_args)

    return to_load, positional_args, arguments


def invoke(config, recurrent=True):
    to_invoke, positional_args, arguments = get_load_method_and_args(
        INVOKE_KEY, config, recurrent
    )
    return to_invoke(*positional_args, **arguments)


def init(config, recurrent=True):
    to_init, positional_args, arguments = get_load_method_and_args(
        INIT_KEY, config, recurrent
    )

    if not isinstance(to_init, type):
        raise TypeError(f"Expected {to_init} to be a class, but it is {type(to_init)}")

    return to_init(*positional_args, **arguments)


def bind(config, recurrent=True):
    to_bind, positional_args, arguments = get_load_method_and_args(
        BIND_KEY, config, recurrent
    )
    return _bind(to_bind, *positional_args, **arguments)


def load_config(config, recurrent=True):
    if BIND_KEY in config:
        return bind(config, recurrent)
    elif INIT_KEY in config:
        return init(config, recurrent)
    elif INVOKE_KEY in config:
        return invoke(config, recurrent)
    else:
        raise KeyError(
            f"None of [{BIND_KEY}, {INVOKE_KEY}, {INIT_KEY}] found " f"in config."
        )


def load_multiple(configs, recurrent=True):
    if isinstance(configs, dict):
        return {key: load_config(config, recurrent) for key, config in configs.items()}
    elif iter(configs):
        return [load_config(config, recurrent) for config in configs]
    else:
        raise TypeError(
            f"can only bind/invoke/init paths from a dict or an "
            f"iterable, not {type(configs)}"
        )


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
    return {key: value for key, value in d.items() if f(key, value)}


PATH_KEY = "^invoke"


def invoke_class(config):
    invoke = get_class_from_path(config[PATH_KEY])
    arguments = keep(config, lambda k, v: k != PATH_KEY)

    return invoke(**arguments)


def path_invocations(configs):
    if isinstance(configs, dict):
        return {key: invoke_class(config) for key, config in configs.items()}
    elif iter(configs):
        return [invoke_class(config) for config in configs]
    else:
        raise Exception(
            f"can only invoke paths from a dict or an iterable, not {configs}"
        )


import importlib
from functools import partial

INVOKE_KEY = "^invoke"
BIND_KEY = "^bind"
INIT_KEY = "^init"


def get_dynamic(config, default=""):
    if INIT_KEY in config:
        return config[INIT_KEY]
    elif INVOKE_KEY in config:
        return config[INVOKE_KEY]
    else:
        return default


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
    arguments = {k: v for k, v in config.items() if k != key}
    return name, arguments


def invoke(config):
    to_invoke, arguments = get_name_and_arguments(INVOKE_KEY, config)

    return to_invoke(**arguments)


def init(config):
    to_init, arguments = get_name_and_arguments(INIT_KEY, config)

    if not isinstance(to_init, type):
        raise TypeError(
            f"Expected {to_init} to be a class, but it is " f"{type(to_init)}"
        )

    for key, value in arguments.items():
        if isinstance(value, dict):
            try:
                arguments[key] = load_config(value)
            except:
                pass

    return to_init(**arguments)


def init_or_invoke(config):
    if INIT_KEY in config:
        return init(config)
    elif INVOKE_KEY in config:
        return invoke(config)
    else:
        raise TypeError(f"neither {INIT_KEY} or {INVOKE_KEY} in config {config}")


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
        raise KeyError(
            f"None of [{BIND_KEY}, {INVOKE_KEY}, {INIT_KEY}] found "
            f"in config. {config}"
        )


def load_multiple(configs):
    if isinstance(configs, dict):
        return {key: load_config(config) for key, config in configs.items()}
    elif iter(configs):
        return [load_config(config) for config in configs]
    else:
        raise TypeError(
            f"can only bind/invoke/init paths from a dict or an "
            f"iterable, not {type(configs)}"
        )