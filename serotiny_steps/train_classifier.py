#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import logging
from datetime import datetime

import fire

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger
from pytorch_lightning.callbacks import ModelCheckpoint, GPUStatsMonitor, EarlyStopping
from pl_bolts.callbacks import PrintTableMetricsCallback
from ray.tune.integration.pytorch_lightning import TuneReportCallback
from ray import tune

from serotiny.models.callbacks.progress_bar import GlobalProgressBar
from serotiny.models.classification import ClassificationModel

import serotiny.datamodules as datamodules

###############################################################################

log = logging.getLogger(__name__)
pl.seed_everything(42)

###############################################################################


def train_classifier_config(
    config,
):
    """
    Config version of train_classifier
    Only used in tune_model.py
    """
    train_classifier(**config)


def train_classifier(
    datasets_path: str,
    output_path: str,
    datamodule: str,
    model: str,
    batch_size: int,
    num_gpus: int,
    num_workers: int,
    num_epochs: int,
    lr: float,
    optimizer: str,
    lr_scheduler: str,
    test: bool,
    tune_bool: bool,
    x_label: str,
    y_label: str,
    classes: list,
    dimensionality: int,
    precision: int,
    **datamodule_kwargs
):
    """
    Initialize dataloaders and model
    Call trainer.fit()

    Parameters
    ------------
    datasets_path: str,
        Path to data directory containing csv's for train
        tes and val splits

    output_path: str,
        Path to output directory for saving trained model
        and tensorboard logs

    datamodule: str,
        String key to retrieve a datamodule

    model: str,
        String key to instantiate a model

    batch_size: int,
        Batch size for dataloader

    num_gpus: int,
        Number of gpus to use

    num_workers: int,
        Number of worker processes to use in dataloader

    num_epochs: int,
        Number of epochs to train model

    lr: float,
        Learning rate for optimizer

    optimizer: str,
        String key to retrive an optimizer

    lr_scheduler: str,
        String key to retrive a scheduler

    test: bool,
        Whether to test the model or not

    tune_bool: bool,
        Whether to tune hyper params or not

    x_label: str,
        x label key for loading image in datamodule

    y_label: str,
        y label key for loading image label
        in datamodule

    classes: List
        list of classes in y_label

    dimensionality: int
        Dimensionality of input data

    precision: int
        Select 32-bit or 16-bit precision for training

    **datamodule_kwargs:
        Any additional keyword arguments required by the datamodule
    """

    if precision not in [16, 32]:
        raise ValueError("Precision must be 16 or 32")

    if dimensionality == 2:
        import serotiny.networks.classification._2d as available_nets
    elif dimensionality == 3:
        import serotiny.networks.classification._3d as available_nets
    elif dimensionality == 1:
        raise NotImplementedError("No networks for 1-dimensional inputs available (yet)")
    else:
        raise ValueError("Parameter `dimensionality` should be 1, 2 or 3")

    if model not in available_nets.__dict__:
        raise KeyError(f"Chosen network {model} not available.\n"
                       f"Available networks, for the selected dimensionality "
                       f"({dimensionality}):\n{available_nets.__all__}")

    network_class = available_nets.__dict__[model]

    # Load data module
    datamodule = datamodules.__dict__[datamodule](
        batch_size=batch_size,
        num_workers=num_workers,
        data_dir=datasets_path,
        x_label=x_label,
        y_label=y_label,
        **datamodule_kwargs
    )
    datamodule.setup()

    in_channels = datamodule.num_channels
    input_dims = datamodule.dims

    # init model
    network_config = {
        "in_channels": in_channels,
        "num_classes": len(classes),
        "input_dims": input_dims,
    }

    network = network_class(**network_config)

    classification_model = ClassificationModel(
        network,
        x_label=datamodule.x_label,
        y_label=datamodule.y_label,
        in_channels=in_channels,
        classes=classes,
        dimensions=input_dims,
        lr=lr,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
    )

    # Initialize a logger
    if tune_bool is False:

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

        early_stopping = EarlyStopping("val_loss")

        callbacks = [
            PrintTableMetricsCallback(),
            GlobalProgressBar(),
            early_stopping,
        ]

        if num_gpus > 0:
            callbacks.append(GPUStatsMonitor())

        # Initialize a trainer
        trainer = pl.Trainer(
            logger=[tb_logger, csv_logger],
            accelerator="ddp",
            replace_sampler_ddp=False,
            gpus=num_gpus,
            max_epochs=num_epochs,
            progress_bar_refresh_rate=20,
            checkpoint_callback=checkpoint_callback,
            callbacks=callbacks,
            precision=precision,
            benchmark=False,
            profiler=False,
            weights_summary="full",
            deterministic=True,
        )

        # Train the model ⚡
        trainer.fit(classification_model, datamodule)

        # test the model
        if test is True:
            trainer.test(datamodule=datamodule)

        # Use this to get best model path from callback
        print("Best mode path is", checkpoint_callback.best_model_path)
        print("Use checkpoint = torch.load[CKPT_PATH] to get checkpoint")
        print("use model = ClassificationModel(),")
        print("trainer = Trainer(resume_from_checkpoint=CKPT_PATH)")
        print("to load trainer")

        return checkpoint_callback.best_model_path
    else:
        # Add tensorboard logs to tune directory so there
        # are no repeat logs
        logger = TensorBoardLogger(save_dir=tune.get_trial_dir(), name="", version=".")

        # Tune callback tracks val loss and accuracy
        tune_callback = TuneReportCallback(
            {"loss": "val_loss_epoch", "mean_accuracy": "val_accuracy_epoch"},
            on="validation_end",
        )
        # Use ddp to split training across gpus
        trainer = pl.Trainer(
            max_epochs=num_epochs,
            accelerator="ddp",
            replace_sampler_ddp=False,
            gpus=num_gpus,
            logger=logger,
            progress_bar_refresh_rate=0,
            callbacks=[tune_callback],
        )
        # Train the model ⚡
        trainer.fit(classification_model, datamodule)
    # return


if __name__ == "__main__":
    # example command:
    # python -m serotiny.steps.train_classifier \
    #     --datasets_path "./results/splits/" \
    #     --output_path "./results/models/" \

    fire.Fire(train_classifier)
