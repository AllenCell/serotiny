import os
import logging
from collections import OrderedDict

from typing import List, Dict, Optional
from datetime import datetime

import fire
import pytorch_lightning as pl

import serotiny.datamodules as datamodules
import serotiny.models as models
from serotiny.models.zoo import get_checkpoint_callback, store_metadata
from serotiny.utils import module_get, get_classes_from_config

log = logging.getLogger(__name__)


def train_model(
    model_name: str,
    model_config: Dict,
    datamodule_name: str,
    datamodule_config: Dict,
    trainer_config: Dict,
    gpu_ids: List[int],
    model_zoo_config: Dict,
    callbacks: Dict = {},
    loggers: Dict = {},
    version_string: Optional[str] = None,
    seed: int = 42,
    metadata: Dict = {},
):

    pl.seed_everything(seed)

    model_zoo_path = model_zoo_config.get("path")
    store_metadata_flag = model_zoo_config.get("store_metadata", True)
    checkpoint_monitor = model_zoo_config.get("checkpoint_monitor", None)
    checkpoint_mode = model_zoo_config.get("checkpoint_mode", None)

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join([str(_) for _ in gpu_ids])
    num_gpus = len(gpu_ids)
    num_gpus = (num_gpus if num_gpus != 0 else None)

    model_class = module_get(models, model_name)
    model = model_class(**model_config)

    if version_string is None:
        version_string = "version_" + datetime.now().strftime("%d-%m-%Y--%H-%M-%S")

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
    callbacks += [checkpoint_callback]

    loggers = get_classes_from_config(loggers)

    trainer = pl.Trainer(
        **trainer_config,
        logger=loggers,
        gpus=num_gpus,
        callbacks=callbacks,
    )

    log.info("Calling trainer.fit")
    trainer.fit(model, datamodule)

    trainer.test(datamodule=datamodule)

    if store_metadata_flag:
        store_metadata(metadata, model_name, version_string, model_zoo_path)


if __name__ == "__main__":
    fire.Fire(train_model)
