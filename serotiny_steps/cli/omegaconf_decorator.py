from inspect import Parameter, signature
from makefun import wraps
from omegaconf import OmegaConf, ListConfig
from omegaconf._utils import split_key


def omegaconf_decorator(func, *config_args):
    """
    Decorator to wrap around serotiny steps, to give them OmegaConf capabilities.
    This allows easy integration in workflows, overriding config parameters, and
    loading configuration elements directly from yaml files

    Parameters
    ----------
    func: callable
        The function to be decorated

    config_args: Sequence[str]
        List of arguments to be registered as "configuration arguments" for the
        function `func`. This tells the decorator to check whether these arguments
        are paths to .yaml files, and if so, it loads them

    """

    @wraps(func, append_args=Parameter(name="dotlist", kind=Parameter.VAR_POSITIONAL))
    def wrapper(*args, **kwargs):
        func_sig = signature(func)
        func_arg_names = list(func_sig.parameters.keys())

        base_args = {}
        for arg_ix, arg_value in enumerate(args):
            base_args[func_arg_names[arg_ix]] = arg_value

        base_args.update(kwargs)
        base_conf = OmegaConf.create(base_args)

        for config in config_args:
            if isinstance(base_conf[config], str):
                config_path = base_conf[config]
                dotstring = None
                if ":" in config_path:
                    config_path, dotstring = config_path.split(":")

                _this_config = OmegaConf.load(config_path)
                if dotstring is not None:
                    for field in split_key(dotstring):
                        if isinstance(_this_config, ListConfig):
                            field = int(field)
                        _this_config = _this_config[field]
                base_conf[config] = _this_config

        override_conf = OmegaConf.from_dotlist(args[len(func_arg_names) :])

        conf = OmegaConf.merge(base_conf, override_conf)

        return func(**conf)

    return wrapper
