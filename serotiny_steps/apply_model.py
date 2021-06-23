import os

from typing import Dict, List

import fire
import pytorch_lightning as pl

import serotiny.datamodules as datamodules
from serotiny.models.zoo import get_model
from serotiny.utils import PATH_KEY, invoke_class, path_invocations


def apply_model(
    model_path: str,
    datamodule_config: Dict,
    trainer_config: Dict,
    model_zoo_config: Dict,
    callbacks_config: Dict,
    gpu_ids: List[int],
):

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join([str(id) for id in gpu_ids])
    num_gpus = len(gpu_ids)
    num_gpus = (num_gpus if num_gpus != 0 else None)

    model_zoo_path = model_zoo_config.get("path")
    model = get_model(model_path, model_zoo_path)

    datamodule = invoke_class(datamodule_config)
    datamodule.setup()

    callbacks = path_invocations(callbacks_config)
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
