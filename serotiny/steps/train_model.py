#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple

import torch
from torchvision import transforms
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, GPUStatsMonitor, EarlyStopping
from pl_bolts.callbacks import PrintTableMetricsCallback

# from torchvision.utils import save_image

from datastep import Step, log_run_params
from ..train_test_split import TrainTestSplit
from ...constants import DatasetFields
from ...library.data import load_data_loader
from ...library.progress_bar import GlobalProgressBar
from ...library.lightning_model import (
    ClassificationModel,
    MyPrintingCallback,
    available_networks,
)
from ...library.csv import load_csv
from ...library.image import png_loader

###############################################################################

log = logging.getLogger(__name__)

###############################################################################


class LoadId(object):
    def __init__(self, id_fields):
        self.id_fields = id_fields

    def __call__(self, row):
        return {key: row[key] for key in self.id_fields}


class LoadClass(object):
    def __init__(self, num_classes):
        self.num_classes = num_classes

    def __call__(self, row):
        # return torch.tensor([row[str(i)] for i in range(self.num_classes)])
        return torch.tensor(
            row["ChosenMitoticClassInteger"],
        )


class LoadImage(object):
    def __init__(self, chosen_label, num_channels, channel_indexes, transform):
        self.chosen_label = chosen_label
        self.num_channels = num_channels
        self.channel_indexes = channel_indexes
        self.transform = transform

    def __call__(self, row):
        return png_loader(
            # DatasetFields.Chosen2DProjectionPath
            row[self.chosen_label],
            channel_order="CYX",
            indexes={"C": self.channel_indexes or range(self.num_channels)},
            transform=self.transform,
        )


class TrainModel(Step):
    def __init__(
        self,
        direct_upstream_tasks: List["Step"] = [TrainTestSplit],
        config: Optional[Union[str, Path, Dict[str, str]]] = None,
    ):
        super().__init__(direct_upstream_tasks=direct_upstream_tasks, config=config)

    @log_run_params
    def run(
        self,
        datasets: List[str],
        dimensions: Tuple[int, int],
        one_hot_len: int,
        model,
        training,
        id_fields,
        channels,
        metrics,
        **kwargs,
    ):
        dataset_paths = {split: Path(path) for split, path in datasets.items()}
        num_channels = len(channels)
        chosen_channels = training["channel_indexes"]
        channel_indexes = None

        if chosen_channels is not None:
            try:
                channel_indexes = [
                    channels.index(channel_name) for channel_name in chosen_channels
                ]
                num_channels = len(channel_indexes)
            except ValueError:
                raise Exception(
                    (
                        f"channel indexes {channel_indexes} "
                        f"do not match channel names {channels}"
                    )
                )

        for path in dataset_paths.values():
            if not path.exists():
                raise Exception(f"not all datasets are present, missing {path}")

        # transform = transforms.Compose(
        #     [transforms.CenterCrop(256), transforms.Resize(256)]
        # )
        transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.CenterCrop(256),
                transforms.Resize(256),
                transforms.ToTensor(),
            ]
        )
        # transform = None

        # TODO choose either one hot encoding or mitotic class integer
        # This choice is useful when deciding what loss function to use later
        loaders = {
            # Use callable class objects here because lambdas aren't picklable
            "id": LoadId(id_fields),
            "mitotic_class": LoadClass(one_hot_len),
            # "mitotic_class": lambda row: torch.tensor(
            #     row["ChosenMitoticClassInteger"],
            # ),
            "projection_image": LoadImage(
                DatasetFields.Chosen2DProjectionPath,
                num_channels,
                channel_indexes,
                transform,
            ),
        }

        # TODO configure and make augmented dataloaders
        dataloaders = {}
        for split, path in dataset_paths.items():
            # TODO: add required fields
            dataset = load_csv(path, [])
            # shuffle = split == "train"
            shuffle = False

            if split == "train":
                transform = transforms.Compose(
                    [
                        transforms.ToPILImage(),
                        transforms.CenterCrop(256),
                        transforms.Resize(256),
                        transforms.RandomRotation(90),
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

            image_x = test_image.shape[1]
            image_y = test_image.shape[2]
            dimensions = (image_x, image_y)

            dataloaders[split] = load_data_loader(
                dataset,
                loaders,
                transform=transform,
                shuffle=shuffle,
                batch_size=training["batch_size"],
                num_workers=training["num_workers"],
            )

        # init model
        network_config = {
            "num_channels": num_channels,
            "num_classes": one_hot_len,
            "dimensions": dimensions,
        }

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

        # TODO add all config args to model
        print("dimensions", dimensions)
        ae = ClassificationModel(
            network,
            x_label="projection_image",
            y_label="mitotic_class",
            num_channels=num_channels,
            num_classes=one_hot_len,
            image_x=image_x,
            image_y=image_y,
            lr=training["lr"],
            optimizer=training["optimizer"],
            scheduler=training["scheduler"],
            metrics=metrics,
        )

        # Initialize a logger
        # TODO configure output model location
        logger = TensorBoardLogger(
            str(self.step_local_staging_dir) + "/lightning_logs",
        )

        # Initialize model checkpoint
        checkpoint_callback = ModelCheckpoint(
            filepath=str(self.step_local_staging_dir) + "/checkpoints/",
            # if save_top_k = 1, all files in this local staging dir
            # will be deleted when a checkpoint is saved
            # save_top_k=1,
            monitor="val_loss",
            verbose=True,
        )

        early_stopping = EarlyStopping('val_loss')

        # Initialize a trainer
        trainer = pl.Trainer(
            logger=logger,
            accelerator="ddp",
            replace_sampler_ddp=False,
            gpus=training["num_gpus"],
            max_epochs=training["num_epochs"],
            progress_bar_refresh_rate=20,
            checkpoint_callback=checkpoint_callback,
            callbacks=[
                MyPrintingCallback(),
                PrintTableMetricsCallback(),
                GPUStatsMonitor(),
                GlobalProgressBar(),
                early_stopping,
            ],
            precision=16,
            benchmark=True,
            profiler=True,
            weights_summary="full",
        )

        # Train the model ⚡
        trainer.fit(ae, dataloaders["train"], dataloaders["valid"])

        # test the model
        # trainer.test(test_dataloaders=dataloaders["test"])

        # Use this to get best model path from callback
        print("Best mode path is", checkpoint_callback.best_model_path)
        print("Use checkpoint = torch.load[CKPT_PATH] to get checkpoint")
        print("use model = ClassificationModel(),")
        print("trainer = Trainer(resume_from_checkpoint=CKPT_PATH) to load trainer")
        return
