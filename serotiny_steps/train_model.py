import os
import importlib
import logging

from typing import Optional, List, Dict
from pathlib import Path
from datetime import datetime

import fire
import pytorch_lightning as pl

import serotiny.datamodules as datamodules
import serotiny.models as models
from serotiny.models.zoo import get_checkpoint_callback, store_called_args

log = logging.getLogger(__name__)

def module_get(module, key):
    if key not in module.__dict__:
        raise KeyError(
            f"Chosen {module} module {key} not available.\n"
            f"Available {module}(s):\n"
            f"{module.__all__}"
        )

    return module.__dict__[key]


def get_classes_from_config(configs: Dict):
    """
    Return a list of instantiated classes given by `configs`. Each key in
    `configs` is a class path, to be imported dynamically via importlib,
    with arguments given by the correponding value in the dict.
    """
    instantiated_classes = []
    for class_path, class_config in configs.items():
        class_path = class_path.split(".")
        class_module = ".".join(class_path[:-1])
        class_name = class_path[-1]
        the_class = getattr(importlib.import_module(class_module), class_name)
        instantiated_class = the_class(**class_config)
        instantiated_classes.append(instantiated_class)

    return instantiated_classes


def train_model(
    model_name: str,
    model_config: Dict,
    datamodule_name: str,
    datamodule_config: Dict,
    trainer_config: Dict,
    gpu_ids: List[int],
    callbacks: Dict = {},
    loggers: Dict = {},
    model_zoo_path: Optional[str] = None,
    seed: int = 42,
    store_config: bool = True,
    checkpoint_monitor: Optional[str] = None,
    checkpoint_mode: str = "min",
):
    called_args = locals()

    pl.seed_everything(seed)

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(gpu_ids)
    num_gpus = len(gpu_ids)
    num_gpus = (num_gpus if num_gpus != 0 else None)

    model_class = module_get(models, model_name)
    model = model_class(**model_config)
    version_string = "version_" + datetime.now().strftime("%d-%m-%Y--%H-%M-%S")

    if store_config:
        store_called_args(called_args, model_name, version_string, model_zoo_path)

    create_datamodule = module_get(datamodules, datamodule_name)
    datamodule = create_datamodule(**datamodule_config)
    datamodule.setup()

    if checkpoint_monitor is not None:
        checkpoint_callback = get_checkpoint_callback(
            model_name,
            version_string,
            checkpoint_monitor,
            checkpoint_mode,
            model_zoo_path
        )
    else:
        checkpoint_callback = None

    callbacks = get_classes_from_config(callbacks)
    loggers = get_classes_from_config(loggers)

    trainer = pl.Trainer(
        **trainer_config,
        logger=loggers,
        gpus=num_gpus,
        checkpoint_callback=checkpoint_callback,
        callbacks=callbacks,
    )

    log.info("Calling trainer.fit")
    trainer.fit(model, datamodule)

    trainer.test(datamodule=datamodule)


if __name__ == "__main__":
    fire.Fire(train_model)
