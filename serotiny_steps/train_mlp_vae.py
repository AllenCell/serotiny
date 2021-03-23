#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import logging
from typing import Optional, List
from datetime import datetime

import fire
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger
from pytorch_lightning.callbacks import ModelCheckpoint, GPUStatsMonitor, EarlyStopping

from serotiny.progress_bar import GlobalProgressBar
from serotiny.networks.vae import CBVAEEncoderMLP, CBVAEDecoderMLP
from serotiny.models import CBVAEMLPModel

import serotiny.datamodules as datamodules
from serotiny.models.callbacks import MLPVAELogging, SpharmLatentWalk

log = logging.getLogger(__name__)
pl.seed_everything(42)


def train_mlp_vae(
    data_dir: str,
    output_path: str,
    datamodule: str,
    batch_size: int,
    num_gpus: int,
    num_workers: int,
    num_epochs: int,
    lr: float,
    optimizer: str,
    scheduler: str,
    x_label: str,
    c_label: str,
    c_label_ind: str,
    x_dim: int,
    c_dim: int,
    enc_layers: list,
    dec_layers: list,
    beta: float,
    num_classes: Optional[int] = None,
    **kwargs,
):
    """
    Instantiate and train a bVAE.

    Parameters
    ----------

    """
    if datamodule not in datamodules.__dict__:
        raise KeyError(
            f"Chosen datamodule {datamodule} not available.\n"
            f"Available datamodules:\n{datamodules.__all__}"
        )

    # Load data module
    if datamodule == "GaussianDataModule":
        dm = datamodules.__dict__[datamodule](
            batch_size=batch_size,
            num_workers=num_workers,
            data_dir=data_dir,
            x_label=x_label,
            c_label=c_label,
            c_label_ind=c_label_ind,
            x_dim=x_dim,
            shuffle=True,
            **kwargs,  # Like corr, length
        )

        dm_no_shuffle = datamodules.__dict__[datamodule](
            batch_size=batch_size,
            num_workers=num_workers,
            data_dir=data_dir,
            x_label=x_label,
            c_label=c_label,
            c_label_ind=c_label_ind,
            x_dim=x_dim,
            shuffle=False,
            **kwargs,
        )
    else:
        dm = datamodules.__dict__[datamodule](
            batch_size=batch_size,
            num_workers=num_workers,
            data_dir=data_dir,
            x_label=x_label,
            c_label=c_label,
            c_label_ind=c_label_ind,
            x_dim=x_dim,
            **kwargs,
        )
        dm.prepare_data()
        dm.setup()

    encoder = CBVAEEncoderMLP(
        x_dim=x_dim,
        c_dim=c_dim,
        enc_layers=enc_layers,
    )

    decoder = CBVAEDecoderMLP(
        x_dim=x_dim,
        c_dim=c_dim,
        dec_layers=dec_layers,
    )

    vae = CBVAEMLPModel(
        encoder=encoder,
        decoder=decoder,
        optimizer=optimizer,
        scheduler=scheduler,
        x_label=x_label,
        c_label=c_label,
        c_label_ind=c_label_ind,
    )

    tb_logger = TensorBoardLogger(
        save_dir=str(output_path) + "/lightning_logs",
        version="version_" + datetime.now().strftime("%d-%m-%Y--%H-%M-%S"),
        name="",
    )

    csv_logger = CSVLogger(
        save_dir=str(output_path) + "/lightning_logs" + "/csv_logs",
        version="version_" + datetime.now().strftime("%d-%m-%Y--%H-%M-%S"),
        name="",
    )

    ckpt_path = os.path.join(
        str(output_path) + "/lightning_logs",
        tb_logger.version,
        "checkpoints",
    )

    # Initialize model checkpoint
    checkpoint_callback = ModelCheckpoint(
        dirpath=ckpt_path,
        # if save_top_k = 1, all files in this local staging dir
        # will be deleted when a checkpoint is saved
        # save_top_k=1,
        monitor="val_loss",
        verbose=True,
    )

    if datamodule == "GaussianDataModule":
        callbacks = [
            GPUStatsMonitor(),
            GlobalProgressBar(),
            MLPVAELogging(datamodule=dm_no_shuffle),
        ]
    elif datamodule == "VarianceSpharmCoeffs":
        callbacks = [
            GPUStatsMonitor(),
            GlobalProgressBar(),
            MLPVAELogging(datamodule=dm),
            SpharmLatentWalk(),
        ]

    trainer = pl.Trainer(
        logger=[tb_logger, csv_logger],
        accelerator="ddp",
        replace_sampler_ddp=False,
        gpus=num_gpus,
        max_epochs=num_epochs,
        progress_bar_refresh_rate=5,
        checkpoint_callback=checkpoint_callback,
        callbacks=callbacks,
        benchmark=False,
        profiler=False,
        deterministic=True,
        automatic_optimization=False,
    )

    trainer.fit(vae, dm)

    # test the model
    if datamodule == "GaussianDataModule":
        trainer.test(datamodule=dm_no_shuffle)
    else:
        trainer.test(datamodule=dm)

    return checkpoint_callback.best_model_path


if __name__ == "__main__":
    # example command:
    # python -m serotiny.steps.train_model \
    #     --datasets_path "./results/splits/" \
    #     --output_path "./results/models/" \

    fire.Fire(train_mlp_vae)