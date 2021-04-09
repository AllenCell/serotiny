#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import logging
from pathlib import Path
from typing import Optional, Sequence, Any, Dict
from datetime import datetime

import fire
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger
from pytorch_lightning.callbacks import ModelCheckpoint, GPUStatsMonitor, EarlyStopping
from pytorch_lightning.profiler import PyTorchProfiler

#from serotiny.networks.vae import CBVAEDecoder, CBVAEEncoder
#from serotiny.models import CBVAEModel
# from serotiny.networks._3d.unet import Unet
# from serotiny.models.unet import UnetModel

import serotiny.datamodules as datamodules
import serotiny.losses as losses
import serotiny.networks as networks
import serotiny.models as models
from serotiny.progress_bar import GlobalProgressBar

log = logging.getLogger(__name__)
pl.seed_everything(42)

def module_get(module, key):
    if key not in module.__dict__:
        raise KeyError(
            f"Chosen {module} module {key} not available.\n"
            f"Available {module}(s):\n"
            f"{module.__all__}"
        )

    return module.__dict__[key]

def train_model(
    datamodule: Dict[str, Any],
    network: Dict[str, Any],
    loss: Dict[str, Any],
    model: Dict[str, Any],
    training: Dict[str, Any],
):
    """
    Instantiate and train a bVAE.

    Parameters
    ----------
    data_dir: str
        Path to dataset, read by the datamodule
    output_path: str
        Path to store model and logs
    datamodule: str,
        String specifying which datamodule to use
    batch_size: int
        Batch size used for trainig and evaluation
    num_gpus: int
        Number of GPU cards to use
    num_workers: int
        Number of workers allocated to the dataloader
    num_epochs: int
        Maximum number of epochs to train on
    lr: float
        Learning rate used for training
    optimizer: str
        String to specify the optimizer to use
    loss: str
        String to specify the loss function to use
    test: bool
        Flag to tell whether to run the test step after training
    # x_label: str
    #     String to specify the inputs (x)
    # y_label: str
    #     String to specify the outputs (y)
    # input_channels: Sequence[int]
    #     Which channel (indices) to use as input
    # output_channels: Sequence[int]
    #     Which channel (indices) to use as output
    # depth: int
    #     How many layers the Unet will have
    # auto_padding: bool = False
    #     Whether to apply padding to ensure images from down double conv match up double conv

    """

    datamodule_key = training['datamodule']
    create_datamodule = module_get(datamodules, datamodule_key)
    datamodule = create_datamodule(**datamodule)
    datamodule.setup()
    
    print(f'datamodule.dims = {datamodule.dims}')

    loss_key = training['loss']
    create_loss = module_get(losses, loss_key)
    loss = create_loss(**loss)

    network_key = training['network']
    create_network = module_get(networks, network_key)
    network = create_network(**network)
    network.print_network()
    
    version_string = "version_" + datetime.now().strftime("%d-%m-%Y--%H-%M-%S")
    output_path = training['output_path']
    lightning_logs_path = Path(output_path) / "lightning_logs"

    model_key = training['model']
    create_model = module_get(models, model_key)
    model_params = dict(
        model,
        loss=loss,
        network=network,
        input_dims=datamodule.dims,
        output_path=output_path,
        version_string=version_string,
    )    
    model = create_model(**model_params)
    
    tb_logger = TensorBoardLogger(
        save_dir=lightning_logs_path,
        version=version_string,
        name="",
    )

    ckpt_path = os.path.join(
        lightning_logs_path,
        tb_logger.version,
        "checkpoints",
    )

    # Initialize model checkpoint
    checkpoint_callback = ModelCheckpoint(
        dirpath=ckpt_path,
        # if save_top_k = 1, all files in this local staging dir
        # will be deleted when a checkpoint is saved
        # save_top_k=1,
        monitor="validation_loss",
        verbose=True,
    )

    # TODO: make early_stopping optional
    early_stopping = EarlyStopping("validation_loss")

    callbacks = [
        GPUStatsMonitor(),
        GlobalProgressBar(),
        early_stopping,
    ]
    
    # TODO: option to create profiler
    profiler = PyTorchProfiler(profile_memory=True)
    
    trainer = pl.Trainer(
        logger=[tb_logger],
        # accelerator="ddp",
        # replace_sampler_ddp=False,
        gpus=training['num_gpus'],
        max_epochs=training['num_epochs'],
        progress_bar_refresh_rate=5,
        checkpoint_callback=checkpoint_callback,
        callbacks=callbacks,
        benchmark=False,
        # TODO: make profiling optional
        profiler=profiler,
        deterministic=True,
        # TODO: provide ability to specify optimizer
        #automatic_optimization=False,  # Set this to True (default) for automatic optimization
        # TODO: provide an option for precision?
        precision=16,
    )
    
    #import torch
    #print(f'GPU {num_gpus}: allocated memory = {torch.cuda.memory_allocated(num_gpus)}, cached memory = {torch.cuda.memory_reserved(num_gpus)}')
    
    trainer.fit(model, datamodule)

    # test the model
    if training.get('test'):
        trainer.test(datamodule=datamodule)

    return checkpoint_callback.best_model_path


if __name__ == "__main__":
    fire.Fire(train_model)
