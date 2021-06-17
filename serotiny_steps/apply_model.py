import os

from typing import Dict, List

import fire
import pytorch_lightning as pl

import serotiny.datamodules as datamodules
from serotiny.models.zoo import get_model
from serotiny.utils import module_get, get_classes_from_config, module_or_path


def apply_model(
    model_path: str,
    datamodule_name: str,
    datamodule_config: Dict,
    model_zoo_config: Dict,
    trainer_config: Dict,
    gpu_ids: List[int],
    callbacks: Dict,
):

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join([str(id) for id in gpu_ids])
    num_gpus = len(gpu_ids)
    num_gpus = (num_gpus if num_gpus != 0 else None)

    model_zoo_path = model_zoo_config.get("path")
    model = get_model(model_path, model_zoo_path)

    create_datamodule = module_or_path(datamodules, datamodule_name)
    datamodule = create_datamodule(**datamodule_config)
    datamodule.setup()

    callbacks = get_classes_from_config(callbacks)
    trainer = pl.Trainer(
        **trainer_config,
        callbacks=callbacks,
        gpus=num_gpus,
    )

    trainer.test(
        model=model,
        datamodule=datamodule)


if __name__ == "__main__":
    fire.Fire(apply_model)
