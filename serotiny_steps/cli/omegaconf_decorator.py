from inspect import Parameter, signature
from makefun import wraps
from omegaconf import OmegaConf


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
    def wrapper(*args):
        func_sig = signature(func)

        func_arg_names = list(func_sig.parameters.keys())
        base_conf = OmegaConf.create(
            {arg_name: args[arg_ix] for arg_ix, arg_name in enumerate(func_arg_names)}
        )

        for config in config_args:
            if isinstance(base_conf[config], str):
                base_conf[config] = OmegaConf.load(base_conf[config])

        override_conf = OmegaConf.from_dotlist(args[len(func_arg_names) :])

        conf = OmegaConf.merge(base_conf, override_conf)

        func(**conf)

    return wrapper
