#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import logging
from typing import Optional

import fire
import yaml

import torch
import torch.nn.functional as F
import pytorch_lightning as pl
import numpy as np
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    GPUStatsMonitor,
    EarlyStopping,
    ProgressBar
)
from sklearn.decomposition import PCA

from serotiny.networks.vae import CBVAEEncoderMLP, CBVAEDecoderMLP
from serotiny.models import CBVAEMLPModel
from serotiny.utils.metric_utils import get_singular_values


import serotiny.datamodules as datamodules
from serotiny.models.callbacks import (
    MLPVAELogging,
    SpharmLatentWalk,
    GetEmbeddings,
    GetClosestCellsToDims,
    #GlobalProgressBar,
    EmbeddingScatterPlots,
    MarginalKL,
    EmpiricalKL
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
    prior_mode: str,  # "isotropic', 'anisotropic'
    learn_prior_logvar: bool,  # Default None
    init_logvar_pca: bool,
    length: Optional[int] = None,  # For Gaussian
    corr: Optional[bool] = False,  # For Gaussian
    id_fields: Optional[list] = None,  # For Spharm
    set_zero: Optional[bool] = False,  # For Spharm
    overwrite: Optional[bool] = False,  # For Spharm
    values: Optional[list] = None,  # For Spharm
    hues: Optional[list] = None,  # For spharm
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
        log.info("Instantiating datamodule")
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

    log.info("Instantiating encoder")
    encoder = CBVAEEncoderMLP(
        x_dim=x_dim,
        c_dim=c_dim,
        hidden_layers=hidden_layers,
        latent_dims=latent_dims,
    )

    log.info("Instantiating decoder")
    decoder = CBVAEDecoderMLP(
        x_dim=x_dim,
        c_dim=c_dim,
        hidden_layers=hidden_layers,
        latent_dims=latent_dims,
    )

    log.info("Fitting PCA")
    fitted_pca = PCA(n_components=latent_dims).fit(dm.datasets["train"][dm.spharm_cols])

    if init_logvar_pca:
        prior_logvar = fitted_pca.singular_values_[:latent_dims] ** 2
        prior_logvar = prior_logvar / prior_logvar.sum()
        prior_logvar = F.log_softmax(torch.tensor(prior_logvar))
        log.info(f"Initializing prior_logvar to {prior_logvar.tolist()}")
    else:
        prior_logvar = None

    if learn_prior_logvar:
        log.info("Learning prior_logvar")

    log.info("Instantiating VAE")
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
            #GlobalProgressBar(),
            MLPVAELogging(datamodule=dm_no_shuffle),
        ]
    elif datamodule == "VarianceSpharmCoeffs":

        marginal_kl = MarginalKL(
            n_samples=20,
            x_label=dm.x_label,
            c_label=dm.c_label,
            embedding_dim=latent_dims,
        )

        empirical_kl = EmpiricalKL(
            x_label=dm.x_label,
            c_label=dm.c_label,
            embedding_dim=latent_dims,
        )

        mlp_vae_logging = MLPVAELogging(datamodule=dm, values=values)

        get_embeddings = GetEmbeddings(
            x_label=dm.x_label, c_label=dm.c_label, id_fields=dm.id_fields
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

        embedding_scatterplots = EmbeddingScatterPlots(
            fitted_pca=fitted_pca,
            n_components=len(hues),
            c_dim=c_dim
        )
        callbacks = [
            #GPUStatsMonitor(),
            ProgressBar(),
            EarlyStopping("val_loss", patience=15),
            marginal_kl,
            empirical_kl,
            mlp_vae_logging,
            get_embeddings,
            embedding_scatterplots,
            get_closest_cells_to_dims,
            spharm_latent_walk,
        ]

    trainer = pl.Trainer(
        logger=[tb_logger, csv_logger],
        #accelerator="ddp",
        #replace_sampler_ddp=False,
        #gpus=1,
        gpus=None,
        max_epochs=num_epochs,
        progress_bar_refresh_rate=5,
        checkpoint_callback=checkpoint_callback,
        callbacks=callbacks,
    )

    log.info("Calling trainer.fit")
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
