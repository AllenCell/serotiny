from copy import copy
from pytorch_lightning.loggers import LightningLoggerBase as Logger, MLFlowLogger

def add_mlflow_conf(trainer_conf, mlflow_conf=None):
    new_trainer_conf = trainer_conf.copy()
    if mlflow_conf is None:
        return new_trainer_conf

    new_logger = copy(new_trainer_conf.get("logger", []))

    if isinstance(new_logger, MLFlowLogger):
        return new_trainer_conf

    # mlf_logger = MLFlowLogger(**mlflow_conf)

    if isinstance(new_logger, Logger):
        new_logger = [new_logger, mlf_logger]
    elif isinstance(new_logger, list):
        if len(new_logger) == 0:
            new_logger = mlf_logger
        else:
            new_logger.append(mlf_logger)
    elif new_logger is None:
        new_logger = mlf_logger
    else:
        raise TypeError(f"Unexpected type for `logger`: {type(new_logger)}")
    # import ipdb
    # ipdb.set_trace()
    # new_trainer_conf["logger"] = new_logger
    new_trainer_conf.logger = new_logger

    return new_trainer_conf
