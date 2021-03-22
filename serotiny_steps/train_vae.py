#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import logging
from typing import Optional, Sequence
from datetime import datetime

import fire
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger
from pytorch_lightning.callbacks import ModelCheckpoint, GPUStatsMonitor, EarlyStopping

from serotiny.networks.vae import CBVAEDecoder, CBVAEEncoder
from serotiny.models import CBVAEModel

import serotiny.datamodules as datamodules
import serotiny.losses as losses
from serotiny.progress_bar import GlobalProgressBar

log = logging.getLogger(__name__)
pl.seed_everything(42)


def train_vae(
    data_dir: str,
    output_path: str,
    datamodule: str,
    batch_size: int,
    num_gpus: int,
    num_workers: int,
    num_epochs: int,
    lr: float,
    optimizer_encoder: str,
    optimizer_decoder: str,
    crit_recon: str,
    test: bool,
    x_label: str,
    class_label: str,
    n_latent_dim: int,
    n_classes: int,
    activation: str,
    activation_last: str,
    conv_channels_list: Sequence[int],
    input_dims: Sequence[int],
    target_channels: Sequence[int],
    reference_channels: Optional[Sequence[int]],
    beta: float,
    dimensionality: int,
    auto_padding: bool = False,
    kld_reduction: str = "sum",
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
    optimizer_encoder: str
        String to specify the optimizer to use for the encoder net
    optimizer_decoder: str
        String to specify the optimizer to use for the decoder net
    crit_recon: str
        String to specify the loss function to use for reconstruction
    test: bool
        Flag to tell whether to run the test step after training
    x_label: str
        String to specify the inputs (x)
    class_label: str
        String to specify the classes
    n_latent_dim: int
        Dimension of the latent space
    n_classes: int
        Number of classes (0 if not conditioning)
    target_channels: Sequence[int]
        Which channel (indices) to use as target
    reference_channels: Optional[Sequence[int]],
        Which channel (indices) to use as reference
    activation: str
        String to specify the activation function to be used in inner layers
    activation_last: str
        String to specify the activation function to be used in the decoder's last
        layer. For binary outputs, sigmoid should be used
    conv_channels_list: Sequence[int]
        Sequence of channels of the intermediate steps in the encoder. For the decoder,
        the reverse of this list will be used.
    input_dims: Sequence[int]
        Dimensions of the input images
    beta: float
        Value of Beta, the weight of the KLD term in the loss function
    dimensionality: int
        Whether this is a 2d or 3d model.
    auto_padding: bool = False
        Whether to apply padding to ensure generated images match the input size
    kld_reduction: str = "sum"
        Reduction operation to use for the KLD term

    """

    if datamodule not in datamodules.__dict__:
        raise KeyError(
            f"Chosen datamodule {datamodule} not available.\n"
            f"Available datamodules:\n{datamodules.__all__}"
        )

    # Load data module
    datamodule = datamodules.__dict__[datamodule](
        batch_size=batch_size,
        num_workers=num_workers,
        data_dir=data_dir,
        x_label=x_label,
        y_label=class_label,
        **kwargs,
    )
    datamodule.setup()

    if crit_recon not in losses.__dict__:
        raise KeyError(
            f"Chosen reconstruction criterion {crit_recon} not"
            f"available.\n Available datamodules:\n"
            f"{datamodules.__all__}"
        )

    crit_recon = losses.__dict__[crit_recon]()

    encoder = CBVAEEncoder(
        dimensionality=dimensionality,
        n_latent_dim=n_latent_dim,
        n_classes=n_classes,
        n_ch_target=len(target_channels),
        n_ch_ref=len(reference_channels),
        conv_channels_list=conv_channels_list,
        input_dims=input_dims,
        activation=activation,
    )

    decoder = CBVAEDecoder(
        dimensionality=dimensionality,
        n_latent_dim=n_latent_dim,
        n_classes=n_classes,
        n_ch_target=len(target_channels),
        n_ch_ref=len(reference_channels),
        # assuming that conv_channels_list for the decoder is
        # the reverse of that for the encoder
        conv_channels_list=conv_channels_list[::-1],
        imsize_compressed=encoder.imsize_compressed,
        activation=activation,
        activation_last=activation_last,
    )

    vae = CBVAEModel(
        encoder,
        decoder,
        optimizer_encoder=optimizer_encoder,
        optimizer_decoder=optimizer_decoder,
        crit_recon=crit_recon,
        x_label=x_label,
        class_label=class_label,
        num_classes=n_classes,
        target_channels=target_channels,
        reference_channels=reference_channels,
        lr=lr,
        beta=beta,
        kld_reduction=kld_reduction,
        input_dims=input_dims,
        auto_padding=auto_padding,
    )

    tb_logger = TensorBoardLogger(
        save_dir=str(output_path) + "/lightning_logs",
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
        monitor="validation_loss",
        verbose=True,
    )

    early_stopping = EarlyStopping("validation_loss")

    callbacks = [
        GPUStatsMonitor(),
        GlobalProgressBar(),
        early_stopping,
    ]
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
        profiler=False,
        deterministic=True,
        automatic_optimization=False,
    )

    trainer.fit(vae, datamodule)

    # test the model
    if test is True:
        trainer.test(datamodule=datamodule)

    return checkpoint_callback.best_model_path


if __name__ == "__main__":
    fire.Fire(train_vae)
