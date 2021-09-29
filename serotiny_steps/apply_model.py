import os

from typing import Dict, List

import fire
import pytorch_lightning as pl

from serotiny.models.zoo import get_model
from serotiny.utils import init_or_invoke, load_multiple


def apply_model(
    model_path: str,
    datamodule: Dict,
    trainer: Dict,
    model_zoo: Dict,
    callbacks: List[Dict] = [],
    gpu_ids: List[int] = [0],
    loggers: List[Dict] = [],
):

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join([str(id) for id in gpu_ids])
    num_gpus = len(gpu_ids)
    num_gpus = num_gpus if num_gpus != 0 else None

    datamodule_config = datamodule
    trainer_config = trainer
    model_zoo_config = model_zoo
    callbacks_config = callbacks
    loggers_config = loggers

    model_zoo_path = model_zoo_config.get("path")
    model = get_model(model_path, model_zoo_path)

    datamodule = init_or_invoke(datamodule_config)
    datamodule.setup()

    # loggers = path_invocations(loggers_config)
    loggers = load_multiple(loggers_config)
    callbacks = load_multiple(callbacks_config)

    trainer = pl.Trainer(
        **trainer_config, logger=loggers, callbacks=callbacks, gpus=num_gpus,
    )

    trainer.test(model=model, datamodule=datamodule)


if __name__ == "__main__":
    fire.Fire(apply_model)
