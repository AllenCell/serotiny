#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
from pathlib import Path
# from typing import Optional
from os import listdir
import fire

from torchvision import transforms
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    GPUStatsMonitor,
    EarlyStopping
)
from pl_bolts.callbacks import PrintTableMetricsCallback
from ray.tune.integration.pytorch_lightning import TuneReportCallback
from ray import tune

# from torchvision.utils import save_image

from ..constants import DatasetFields
from ..library.data import load_data_loader, LoadId, LoadImage, LoadClass
from ..library.progress_bar import GlobalProgressBar
from ..library.lightning_model import (
    ClassificationModel,
    MyPrintingCallback,
    available_networks,
)
from ..library.csv import load_csv
from ..library.image import png_loader

###############################################################################

log = logging.getLogger(__name__)

###############################################################################


def train_model_config(
    config,
):
    train_model(
        datasets_path=config["datasets_path"],
        output_path=config["output_path"],
        classes=config['classes'],
        model=config['model'],
        label=config['label'],
        batch_size=config['batch_size'],
        num_gpus=config['num_gpus'],
        num_workers=config['num_workers'],
        channel_indexes=config['channel_indexes'],
        num_epochs=config["num_epochs"],
        lr=config['lr'],
        model_optimizer=config['model_optimizer'],
        model_scheduler=config['model_scheduler'],
        id_fields=config['id_fields'],
        channels=config['channels'],
        test=config['test'],
        tune_bool=config['tune_bool'],
    )


def train_model(
    datasets_path: str,
    output_path: str,
    classes: list = ["M0", "M1/M2", "M3", "M4/M5", "M6/M7"],
    model: str = "resnet18",
    label: str = "Draft mitotic state resolved",
    batch_size: int = 64,
    num_gpus: int = 1,
    num_workers: int = 50,
    channel_indexes: list = ["dna", "membrane"],
    num_epochs: int = 1,
    lr: int = 0.001,
    model_optimizer: str = "sgd",
    model_scheduler: str = "reduce_lr_plateau",
    id_fields: list = ["CellId", "CellIndex", "FOVId"],
    channels: list = ["membrane", "structure", "dna"],
    test: bool = True,
    tune_bool: bool = False
):
    """
    Initialize dataloaders and model
    Call trainer.fit()
    """
    filenames = listdir(datasets_path)
    dataset_splits = [split for split in filenames if split.endswith(".csv")]
    dataset_paths = [Path(datasets_path + split) for split in dataset_splits]
    num_channels = len(channels)
    chosen_channels = channel_indexes
    channel_indexes = None

    if chosen_channels is not None:
        try:
            channel_indexes = [
                channels.index(channel_name) for channel_name
                in chosen_channels
            ]
            num_channels = len(channel_indexes)
        except ValueError:
            raise Exception(
                (
                    f"channel indexes {channel_indexes} "
                    f"do not match channel names {channels}"
                )
            )

    for path in dataset_paths:
        if not path.exists():
            raise Exception(f"not all datasets are present, missing {path}")

    transform = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize(256),
            transforms.CenterCrop(256),
            transforms.ToTensor(),
        ]
    )

    # TODO choose either one hot encoding or mitotic class integer
    # This choice is useful when deciding what loss function to use later
    loaders = {
        # Use callable class objects here because lambdas aren't picklable
        "id": LoadId(id_fields),
        "mitotic_class": LoadClass(len(classes)),
        "projection_image": LoadImage(
            DatasetFields.Chosen2DProjectionPath,
            num_channels,
            channel_indexes,
            transform,
        ),
    }

    # TODO configure and make augmented dataloaders
    dataloaders = {}
    for split_csv, path in zip(dataset_splits, dataset_paths):
        split = split_csv.split(".")[0]
        # TODO: add required fields
        dataset = load_csv(path, [])
        shuffle = False

        if split == "train":
            transform = transforms.Compose(
                [
                    transforms.ToPILImage(),
                    transforms.CenterCrop(256),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                ]
            )
            loaders["projection_image"] = LoadImage(
                DatasetFields.Chosen2DProjectionPath,
                num_channels,
                channel_indexes,
                transform,
            )
        # Load a test image to get image dimensions after transform
        test_image = png_loader(
            dataset[DatasetFields.Chosen2DProjectionPath].iloc[0],
            channel_order="CYX",
            indexes={"C": channel_indexes or range(num_channels)},
            transform=transform,
        )

        # Get test image dimensions
        dimensions = (test_image.shape[1], test_image.shape[2])

        dataloaders[split] = load_data_loader(
            dataset,
            loaders,
            transform=transform,
            shuffle=shuffle,
            batch_size=batch_size,
            num_workers=num_workers,
        )

    # init model
    network_config = {
        "num_channels": num_channels,
        "num_classes": len(classes),
        "dimensions": dimensions,
    }
    model = {"type": model}
    network_config.update(model)
    model_type = network_config.pop("type")

    if model_type in available_networks:
        network = available_networks[model_type](**network_config)
    else:
        raise Exception(
            (
                f"network type {model_type} not available, "
                f"options are {list(available_networks.keys())}"
            )
        )

    ae = ClassificationModel(
        network,
        x_label="projection_image",
        y_label="mitotic_class",
        num_channels=num_channels,
        classes=classes,
        label=label,
        image_x=dimensions[0],
        image_y=dimensions[1],
        lr=lr,
        optimizer=model_optimizer,
        scheduler=model_scheduler,
    )

    # Initialize a logger
    if tune_bool is False:

        logger = TensorBoardLogger(
            str(output_path) + "/lightning_logs",
        )
        # Initialize model checkpoint
        checkpoint_callback = ModelCheckpoint(
            filepath=str(output_path) + "/checkpoints/",
            # if save_top_k = 1, all files in this local staging dir
            # will be deleted when a checkpoint is saved
            # save_top_k=1,
            monitor="val_loss",
            verbose=True,
        )

        early_stopping = EarlyStopping("val_loss")

        callbacks = [
            MyPrintingCallback(),
            PrintTableMetricsCallback(),
            GPUStatsMonitor(),
            GlobalProgressBar(),
            early_stopping,
        ]

        # Initialize a trainer
        trainer = pl.Trainer(
            logger=logger,
            accelerator="ddp",
            replace_sampler_ddp=False,
            gpus=num_gpus,
            max_epochs=num_epochs,
            progress_bar_refresh_rate=20,
            checkpoint_callback=checkpoint_callback,
            callbacks=callbacks,
            precision=16,
            benchmark=True,
            profiler=True,
            weights_summary="full",
        )

        # Train the model ⚡
        trainer.fit(ae, dataloaders["train"], dataloaders["valid"])

        # test the model
        if test is True:
            trainer.test(test_dataloaders=dataloaders["test"])

        # Use this to get best model path from callback
        print("Best mode path is", checkpoint_callback.best_model_path)
        print("Use checkpoint = torch.load[CKPT_PATH] to get checkpoint")
        print("use model = ClassificationModel(),")
        print("trainer = Trainer(resume_from_checkpoint=CKPT_PATH)")
        print("to load trainer")
    else:
        # Add tensorboard logs to tune directory so there
        # are no repeat logs
        logger = TensorBoardLogger(
            save_dir=tune.get_trial_dir(), name="", version="."
        )

        # Tune callback tracks val loss and accuracy
        tune_callback = TuneReportCallback(
            {
                "loss": "val_loss_epoch",
                "mean_accuracy": "val_accuracy_epoch"
            },
            on="validation_end"
        )
        # Use ddp to split training across gpus
        trainer = pl.Trainer(
            max_epochs=num_epochs,
            accelerator="ddp",
            replace_sampler_ddp=False,
            gpus=num_gpus,
            logger=logger,
            progress_bar_refresh_rate=0,
            callbacks=[tune_callback]
        )
        # Train the model ⚡
        trainer.fit(ae, dataloaders["train"], dataloaders["valid"])
    # return


if __name__ == '__main__':
    # example command:
    # python -m serotiny.steps.train_model \
    #     --datasets_path "./results/splits/" \
    #     --output_path "./results/models/" \

    fire.Fire(train_model)
