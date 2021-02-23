#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
import fire

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger
from pytorch_lightning.callbacks import ModelCheckpoint, GPUStatsMonitor, EarlyStopping
from pl_bolts.callbacks import PrintTableMetricsCallback
from ray.tune.integration.pytorch_lightning import TuneReportCallback
from ray import tune
from datetime import datetime
import os

from ..library.progress_bar import GlobalProgressBar
from ..library.models.classification import (
    ClassificationModel,
    AVAILABLE_NETWORKS,
)

# from ..library.models.callbacks import (
#     MyPrintingCallback,
# )
import serotiny.library.datamodules as datamodules

###############################################################################

log = logging.getLogger(__name__)
pl.seed_everything(42)

###############################################################################


def train_model_config(
    config,
):
    train_model(
        datasets_path=config["datasets_path"],
        output_path=config["output_path"],
        classes=config["classes"],
        model=config["model"],
        batch_size=config["batch_size"],
        num_gpus=config["num_gpus"],
        num_workers=config["num_workers"],
        channel_indexes=config["channel_indexes"],
        num_epochs=config["num_epochs"],
        lr=config["lr"],
        optimizer=config["optimizer"],
        scheduler=config["scheduler"],
        id_fields=config["id_fields"],
        channels=config["channels"],
        test=config["test"],
        tune_bool=config["tune_bool"],
    )


def train_model(
    datasets_path: str,
    output_path: str,
    data_config: dict,
    datamodule: str,
    model: str,
    batch_size: int,
    num_gpus: int,
    num_workers: int,
    num_epochs: int,
    lr: float,
    optimizer: str,
    scheduler: str,
    test: bool,
    tune_bool: bool,
    x_label: str,
    y_label: str,
):
    """
    Initialize dataloaders and model
    Call trainer.fit()
    """
    # Load data module
    dm_class = datamodules.__dict__[datamodule]

    dm = dm_class(
        batch_size=batch_size,
        num_workers=num_workers,
        config=data_config,
        data_dir=datasets_path,
        x_label=x_label,
        y_label=y_label,
    )
    dm.setup()

    in_channels = dm.num_channels
    dimensions = dm.dims

    # init model
    network_config = {
        "in_channels": in_channels,
        "num_classes": len(data_config["classes"]),
        "dimensions": dimensions,
    }
    model = {"type": model}
    network_config.update(model)
    model_type = network_config.pop("type")

    if model_type in AVAILABLE_NETWORKS:
        network = AVAILABLE_NETWORKS[model_type](**network_config)
    else:
        raise Exception(
            (
                f"network type {model_type} not available, "
                f"options are {list(AVAILABLE_NETWORKS.keys())}"
            )
        )

    classification_model = ClassificationModel(
        network,
        x_label=dm.x_label,
        y_label=dm.y_label,
        in_channels=in_channels,
        classes=data_config["classes"],
        dimensions=dimensions,
        lr=lr,
        optimizer=optimizer,
        scheduler=scheduler,
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
            filepath=ckpt_path,
            # if save_top_k = 1, all files in this local staging dir
            # will be deleted when a checkpoint is saved
            # save_top_k=1,
            monitor="val_loss",
            verbose=True,
        )

        early_stopping = EarlyStopping("val_loss")

        callbacks = [
            PrintTableMetricsCallback(),
            GPUStatsMonitor(),
            GlobalProgressBar(),
            early_stopping,
        ]

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
            precision=16,
            benchmark=False,
            profiler=False,
            weights_summary="full",
            deterministic=True,
        )

        # Train the model ⚡
        trainer.fit(classification_model, dm)

        # test the model
        if test is True:
            trainer.test(datamodule=dm)

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
        trainer.fit(classification_model, dm)
    # return


if __name__ == "__main__":
    # example command:
    # python -m serotiny.steps.train_model \
    #     --datasets_path "./results/splits/" \
    #     --output_path "./results/models/" \

    fire.Fire(train_model)
