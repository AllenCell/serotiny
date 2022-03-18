from inspect import Parameter, signature
from makefun import wraps
from omegaconf import OmegaConf, ListConfig
from omegaconf._utils import split_key
from omegaconf.errors import ConfigTypeError


def _maybe_int(value):
    return int(value) if value.isdigit() else value


def _merge_override(cfg, override):
    if sum(c == "=" for c in override) > 1:
        raise NotImplementedError("Can't handle expressions with multiple '=' chars")

    target, value = override.split("=")
    value = _maybe_int(value)
    path = split_key(target)

    to_update = cfg
    for step in path[:-1]:
        to_update = to_update[_maybe_int(step)]
    to_update[_maybe_int(path[-1])] = value


def _try_read_yaml(param_value):
    if isinstance(param_value, str) and ".yaml:" in param_value:
        config_path = param_value
        dotstring = None
        _spl = config_path.split(":")
        config_path = _spl[0]
        if len(_spl) > 1:
            dotstring = "".join(_spl[1:])

        _this_config = OmegaConf.load(config_path)
        if dotstring is not None:
            for field in split_key(dotstring):
                if isinstance(_this_config, ListConfig):
                    field = int(field)
                _this_config = _this_config[field]
        return _this_config
    return param_value


def omegaconf_decorator(func):
    """
    Decorator to wrap around serotiny steps, to give them OmegaConf capabilities.
    This allows easy integration in workflows, overriding config parameters, and
    loading configuration elements directly from yaml files

    Parameters
    ----------
    func: callable
        The function to be decorated
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        func_sig = signature(func)
        args = list(args)
        _kwargs = {}
        _positional_arg_names = []

        # iterate over function parameters, consuming them from `args`
        # and `kwargs` appropriately. the elements that remain in `args`
        # after this loop will be part of a variadic argument (if they exist)
        # for each of the arguments registered in `config_args`, check
        # if they are a path to a .yaml file or a field therein. if so,
        # read the config from there
        for param_name, param in func_sig.parameters.items():
            if param.kind == Parameter.POSITIONAL_ONLY:
                raise NotImplementedError(
                    "This decorator doesn't support functions with "
                    "positional-only arguments."
                )

            elif param.kind == Parameter.POSITIONAL_OR_KEYWORD:
                param_value = (kwargs.pop(param_name) if param_name in kwargs
                               else args.pop(0))
                param_value = _try_read_yaml(param_value)
                _kwargs[param_name] = param_value
                _positional_arg_names.append(param_name)

        for param_name, param_value in kwargs.items():
            param_value = _try_read_yaml(param_value)
            _kwargs[param_name] = param_value

        variadic_args = args
        conf = OmegaConf.create(_kwargs)

        # this gets returned by Fire and either is provided
        # with additional arguments, which will be dotlist
        # overrides, or no overrides are passed and it gets
        # called as is
        def _merge_overrides_or_call(*args):
            _conf = conf
            if len(args) > 0:
                try:
                    override_conf = OmegaConf.from_dotlist(args)
                    _conf = OmegaConf.merge(conf, override_conf)
                except ConfigTypeError:
                    for override in args:
                        _merge_override(conf, override)
                    _conf = conf

            pos_args = [_conf.pop(arg) for arg in _positional_arg_names]
            return func(*pos_args, *variadic_args, **_conf)

        return _merge_overrides_or_call

    return wrapper
