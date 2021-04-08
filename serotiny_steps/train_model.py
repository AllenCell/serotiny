#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import logging
from pathlib import Path
from typing import Optional, Sequence
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
    data_dir: str,
    output_path: str,
    datamodule: str,
    network: str,
    model: str,
    id_fields: Sequence[str],
    batch_size: int,
    num_gpus: int,
    num_workers: int,
    num_epochs: int,
    lr: float,
    optimizer: str,
    loss: str,  # Unet
    test: bool,
    # x_label: str,
    # y_label: str,  # Unet
    # input_column: str,
    # output_column: str,
    # input_channels: Sequence[str],  # Unet
    # output_channels: Sequence[str],  # Unet
    # depth: int,  # Unet
    # channel_fan_top: int,  # Unet
    # auto_padding: bool = False,  # Unet
    **kwargs,
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

    print(f"all kwargs: {kwargs}")

    datamodule_key = datamodule
    create_datamodule = module_get(datamodules, datamodule_key)

    # if datamodule not in datamodules.__dict__:
    #     raise KeyError(
    #         f"Chosen datamodule {datamodule} not available.\n"
    #         f"Available datamodules:\n{datamodules.__all__}"
    #     )

    # Load data module
    datamodule = create_datamodule(
        batch_size=batch_size,
        num_workers=num_workers,
        id_fields=id_fields,
        data_dir=data_dir,
        # x_label=x_label,
        # y_label=y_label,
        # input_column=input_column,
        # output_column=output_column,
        # input_channels=input_channels,
        # output_channels=output_channels,
        **kwargs,
    )
    datamodule.setup()
    
    print(f'datamodule.dims = {datamodule.dims}')

    # if loss not in losses.__dict__:
    #     raise KeyError(
    #         f"Chosen reconstruction criterion {crit_recon} not"
    #         f"available.\n Available losses:\n"
    #         f"{losses.__all__}"
    #     )

    # loss = losses.__dict__[loss]()

    loss_key = loss
    create_loss = module_get(losses, loss_key)
    loss = create_loss()

    # if network not in networks.__dict__:
    #     raise KeyError(
    #         f"Chosen network {network} not available.\n"
    #         f"Available networks:\n"
    #         f"{networks.__all__}"
    #     )

    # create_network = networks.__dict__[network]
    
    network_key = network
    create_network = module_get(networks, network_key)
    network = create_network(
        # depth=depth, 
        # channel_fan_top=channel_fan_top, 
        # num_input_channels=len(input_channels), 
        # num_output_channels=len(output_channels), 
        **kwargs,
    )
    
    network.print_network()
    
    version_string = "version_" + datetime.now().strftime("%d-%m-%Y--%H-%M-%S")
    test_image_output = Path(output_path) / "test_images" / version_string
    lightning_logs_path = Path(output_path) / "lightning_logs"

    model_key = model
    create_model = module_get(models, model_key)
    model = create_model(
        network=network,
        optimizer=optimizer,
        loss=loss,
        lr=lr,
        **kwargs
    )
    
    print(f"created unet {unet_model}")

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

    early_stopping = EarlyStopping("validation_loss")

    callbacks = [
        GPUStatsMonitor(),
        GlobalProgressBar(),
        early_stopping,
    ]
    
    profiler = PyTorchProfiler(profile_memory=True)
    
    trainer = pl.Trainer(
        logger=[tb_logger],
        # accelerator="ddp",
        # replace_sampler_ddp=False,
        gpus=num_gpus,
        max_epochs=num_epochs,
        progress_bar_refresh_rate=5,
        checkpoint_callback=checkpoint_callback,
        callbacks=callbacks,
        benchmark=False,
        profiler=profiler,
        deterministic=True,
        #automatic_optimization=False,  # Set this to True (default) for automatic optimization
        precision=16,
    )
    
    #import torch
    #print(f'GPU {num_gpus}: allocated memory = {torch.cuda.memory_allocated(num_gpus)}, cached memory = {torch.cuda.memory_reserved(num_gpus)}')
    
    trainer.fit(model, datamodule)

    # test the model
    if test is True:
        trainer.test(datamodule=datamodule)

    return checkpoint_callback.best_model_path


if __name__ == "__main__":
    fire.Fire(train_model)
