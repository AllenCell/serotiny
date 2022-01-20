import os

from typing import Dict, List
import yaml
import fire
import pytorch_lightning as pl

from serotiny.models.zoo import get_model
from serotiny.utils import init_or_invoke, load_multiple


def apply_model(
    model_class: str = None,
    model_id: str = None,
    datamodule: Dict = None,
    trainer: Dict = None,
    model_zoo: Dict = None,
    callbacks: Dict = None,
    gpu_ids: List[int] = None,
    loggers: List[Dict] = [],
    config: Dict = None,
    configfile: str = None,
):
    if configfile:
        with open(configfile, "r") as yaml_cf:
            config = yaml.load(yaml_cf)

    if config:
        model_class = config.get("model_class")
        model_id = config.get("model_id", "10")
        datamodule_config = config.get("datamodule", {})
        trainer_config = config.get("trainer", {})
        model_zoo_config = config.get("model_zoo", {})
        loggers_config = config.get("loggers", [])
        callbacks_config = config.get("callbacks", [])
        gpu_ids = config.get("gpu_ids", [0])

    else:
        datamodule_config = datamodule or {}
        trainer_config = trainer or {}
        model_zoo_config = model_zoo or {}
        loggers_config = loggers or []
        callbacks_config = callbacks or []
        gpu_ids = gpu_ids or [0]

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join([str(id) for id in gpu_ids])
    num_gpus = len(gpu_ids)
    num_gpus = num_gpus if num_gpus != 0 else None

    model_zoo_path = model_zoo_config.get("path")
    model = get_model(model_class, model_id, model_zoo_path)

    datamodule = init_or_invoke(datamodule_config)
    datamodule.setup()

    loggers = load_multiple(loggers_config)
    callbacks = load_multiple(callbacks_config)

    trainer = pl.Trainer(
        **trainer_config,
        logger=loggers,
        callbacks=callbacks,
        gpus=num_gpus,
    )

    trainer.test(model=model, datamodule=datamodule)


if __name__ == "__main__":
    fire.Fire(apply_model)
