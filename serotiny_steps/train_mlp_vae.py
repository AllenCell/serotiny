#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
from typing import Optional

import fire
import pytorch_lightning as pl
import numpy as np
import yaml
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger
from pytorch_lightning.callbacks import ModelCheckpoint, GPUStatsMonitor, EarlyStopping

from serotiny.networks.vae import CBVAEEncoderMLP, CBVAEDecoderMLP
from serotiny.models import CBVAEMLPModel
import os

import serotiny.datamodules as datamodules
from serotiny.models.callbacks import (
    MLPVAELogging,
    SpharmLatentWalk,
    GetEmbeddings,
    GetClosestCellsToDims,
    GlobalProgressBar,
)

log = logging.getLogger(__name__)
pl.seed_everything(42)


def train_mlp_vae(
    source_path: str,
    modified_source_save_dir: str,
    output_path: str,
    checkpoint_path: str,
    datamodule: str,
    batch_size: int,
    gpu_id: str,
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
    latent_dims: int,
    hidden_layers: list,
    beta: float,
    cvapipe_analysis_config_path: str,
    latent_walk_range: list,
    n_cells: int,  # No of closets cells to find per location
    align: str,  # DNA/MEM
    skew: str,  # yes/no
    length: Optional[int] = None,  # For Gaussian
    corr: Optional[bool] = False,  # For Gaussian
    id_fields: Optional[list] = None,  # For Spharm
    set_zero: Optional[bool] = False,  # For Spharm
    overwrite: Optional[bool] = False,  # For Spharm
    values: Optional[list] = None,  # For Spharm
):
    """
    Instantiate and train a bVAE.

    Parameters
    ----------

    """

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id

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
            data_dir=modified_source_save_dir,
            x_label=x_label,
            c_label=c_label,
            c_label_ind=c_label_ind,
            x_dim=x_dim,
            shuffle=True,
            length=length,
            corr=corr,
        )

        dm_no_shuffle = datamodules.__dict__[datamodule](
            batch_size=batch_size,
            num_workers=num_workers,
            data_dir=modified_source_save_dir,
            x_label=x_label,
            c_label=c_label,
            c_label_ind=c_label_ind,
            x_dim=x_dim,
            shuffle=False,
            length=length,
            corr=corr,
        )
    else:
        dm = datamodules.__dict__[datamodule](
            batch_size=batch_size,
            num_workers=num_workers,
            source_path=source_path,
            modified_source_save_dir=modified_source_save_dir,
            x_label=x_label,
            c_label=c_label,
            c_label_ind=c_label_ind,
            x_dim=x_dim,
            set_zero=set_zero,
            overwrite=overwrite,
            id_fields=id_fields,
            align=align,
            skew=skew,
        )
        dm.prepare_data()
        dm.setup()

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
    )

    print(output_path)
    tb_logger = TensorBoardLogger(save_dir=str(output_path))

    csv_logger = CSVLogger(
        save_dir=str(output_path) + "/csv_logs",
    )
    print(checkpoint_path)
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

    if datamodule == "GaussianDataModule":
        callbacks = [
            GPUStatsMonitor(),
            GlobalProgressBar(),
            MLPVAELogging(datamodule=dm_no_shuffle),
        ]
    elif datamodule == "VarianceSpharmCoeffs":

        mlp_vae_logging = MLPVAELogging(datamodule=dm, values=values)

        get_embeddings = GetEmbeddings(
            resample_n=2, x_label=dm.x_label, c_label=dm.c_label, id_fields=dm.id_fields
        )

        get_closest_cells_to_dims = GetClosestCellsToDims(
            np.array(latent_walk_range),
            spharm_coeffs_cols=dm.spharm_cols,
            metric="euclidean",
            id_col="CellId",
            N_cells=n_cells,
            c_shape=c_dim,
        )
        config = yaml.load(
            open(cvapipe_analysis_config_path, "r"),
            Loader=yaml.FullLoader,
        )

        spharm_latent_walk = SpharmLatentWalk(
            config=config,
            spharm_coeffs_cols=dm.spharm_cols,
            latent_walk_range=latent_walk_range,
            ignore_mesh_and_contour_plots=True,
        )
        callbacks = [
            GPUStatsMonitor(),
            GlobalProgressBar(),
            EarlyStopping("val_loss", patience=15),
            mlp_vae_logging,
            get_embeddings,
            get_closest_cells_to_dims,
            spharm_latent_walk,
        ]

    trainer = pl.Trainer(
        logger=[tb_logger, csv_logger],
        accelerator="ddp",
        replace_sampler_ddp=False,
        gpus=1,
        max_epochs=num_epochs,
        progress_bar_refresh_rate=5,
        checkpoint_callback=checkpoint_callback,
        callbacks=callbacks,
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
