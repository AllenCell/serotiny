from serotiny_steps.train_model import train_model
from serotiny_steps.apply_model import apply_model
from .._utils import omegaconf_decorator

class ModelCLI:
    train = omegaconf_decorator(train_model)
    predict = omegaconf_decorator(apply_model)
