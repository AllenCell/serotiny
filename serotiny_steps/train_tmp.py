#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pytorch_lightning as pl
from serotiny.datamodules import VarianceSpharmCoeffs, GaussianDataModule, IntensityRepresentation
import pandas as pd
from pytorch_lightning.loggers import TensorBoardLogger
from datetime import datetime
import os
from pytorch_lightning.callbacks import ModelCheckpoint
import os
import fire
from serotiny.networks.vae import CBVAEEncoderMLP, CBVAEDecoderMLP
from serotiny.models import CBVAEMLPModel
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger
from pytorch_lightning.callbacks import ModelCheckpoint, GPUStatsMonitor, EarlyStopping

###############################################################################
import logging
log = logging.getLogger(__name__)

###############################################################################


def train_tmp():

    """
    """
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
    os.environ["CUDA_VISIBLE_DEVICES"] = "6"

    pl.seed_everything(42)

    x_label="PathToRepresentationFile"
    c_label="CellId"
    c_label_ind="CellId_ind"

    dm = IntensityRepresentation(
        batch_size=2048,
        num_workers=0,
        x_label=x_label,
        c_label=c_label,
        c_label_ind=c_label_ind,
        x_dim=289,
        id_fields = ['CellId'],
        set_zero=True,
        overwrite=False,
        # subset=2048*1,
        source_path="/allen/aics/assay-dev/MicroscopyOtherData/Viana/projects/cvapipe_analysis/",
        modified_source_save_dir="/allen/aics/modeling/ritvik/projects/flowgarden/workflows/bvae_mlp_workflow",
        align="MEM",
        skew="no",
    )
    dm.prepare_data()
    dm.setup()


    print("Initialized datamodule")
    # x_dim = 18040
    x_dim = 53300
    # x_dim = 1065220
    c_dim = 1
    hidden_layers = [256]
    latent_dims = 64
    optimizer = "adam"
    scheduler = "reduce_lr_plateau"
    prior_mode = "isotropic"
    learn_prior_logvar = False
    init_logvar_pca = False
    prior_logvar = None

    encoder = CBVAEEncoderMLP(
        x_dim=x_dim,
        c_dim=c_dim,
        hidden_layers=hidden_layers,
        latent_dims=latent_dims,
    )


    decoder = CBVAEDecoderMLP(
        x_dim=x_dim,
        c_dim=c_dim,
        hidden_layers=hidden_layers,
        latent_dims=latent_dims,
    )


    vae = CBVAEMLPModel(
        encoder=encoder,
        decoder=decoder,
        optimizer=optimizer,
        scheduler=scheduler,
        x_label=x_label,
        c_label=c_label,
        c_label_ind=c_label_ind,
        prior_mode=prior_mode,
        learn_prior_logvar=learn_prior_logvar,
        prior_logvar=prior_logvar,
    )

    output_path = "./test_PIR_serotiny_only_seg_more_res/"
    checkpoint_path = output_path + "checkpoints/"
    tb_logger = TensorBoardLogger(save_dir=str(output_path))

    csv_logger = CSVLogger(
        save_dir=str(output_path) + "/csv_logs",
    )

    # Initialize model checkpoint
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_path,
        filename="{epoch}-{val_loss:.2f}",
        # if save_top_k = 1, all files in this local staging dir
        # will be deleted when a checkpoint is saved
        # save_top_k=1,
        monitor="val_loss",
        verbose=True,
    )

    early_stopping = EarlyStopping("val_loss")

    callbacks = [
        early_stopping,
    ]
    trainer = pl.Trainer(
        logger=[tb_logger, csv_logger],
        accelerator="ddp",
        replace_sampler_ddp=False,
        gpus=1,
        max_epochs=70,
        progress_bar_refresh_rate=5,
        precision=16,
        checkpoint_callback=checkpoint_callback,
        callbacks=callbacks,
        profiler=True,
    )

    trainer.fit(vae, dm)


if __name__ == "__main__":
    # example command:
    # python -m serotiny.steps.split_data \
    #     --dataset_path "data/filtered.csv" \
    #     --output_path "data/splits/" \
    #     --class_column "ChosenMitoticClass" \
    #     --id_column "CellId" \
    #     --ratios "{'train': 0.7, 'test': 0.2, 'valid': 0.1}"

    fire.Fire(train_tmp)
