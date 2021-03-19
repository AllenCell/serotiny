import logging
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
import math
from .utils import compute_generative_metric_tabular
from .utils import visualize_encoder_tabular
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
from brokenaxes import brokenaxes

LOGGER = logging.getLogger(__name__)


def make_plot_fid(
    path_save_dir,
    feature_path,
    model,
    gpu_id,
    enc_layers,
    X_test,
    C_test,
    save=True,
):

    X_test = X_test.view(-1, X_test.size()[-1])
    C_test = C_test.view(-1, C_test.size()[-1])

    print(X_test.size(), C_test.size())

    sns.set_context("talk")

    csv_greedy_features = pd.read_csv(feature_path)

    # conds = [i for i in range(this_kwargs)]
    conds = [
        i for i in csv_greedy_features["selected_feature_number"] if not math.isnan(i)
    ]

    fid_data = {"num_conds": [], "fid": []}

    for i in range(len(conds) + 1):

        tmp1, tmp2 = torch.split(
            C_test.clone(), int(C_test.clone().size()[-1] / 2), dim=1
        )
        for kk in conds:
            tmp1[:, int(kk)], tmp2[:, int(kk)] = 0, 0
        cond_d = torch.cat((tmp1, tmp2), 1)

        print(len(torch.nonzero(cond_d)))

        this_fid = compute_generative_metric_tabular(
            X_test.clone(), C_test.clone(), gpu_id, enc_layers, model, conds
        )
        print("fid", this_fid)

        fid_data["num_conds"].append(X_test.size()[-1] - len(conds))
        fid_data["fid"].append(this_fid)

        try:
            conds.pop()
        except:
            pass

    fid_data = pd.DataFrame(fid_data)

    fig, ax = plt.subplots(1, 1, figsize=(7 * 4, 5))
    sns.lineplot(ax=ax, data=fid_data, x="num_conds", y="fid")
    sns.scatterplot(ax=ax, data=fid_data, x="num_conds", y="fid", s=100, color=".2")

    if save is True:
        path_csv = path_save_dir / Path("fid_data.csv")
        fid_data.to_csv(path_csv)
        LOGGER.info(f"Saved: {path_csv}")

        path_save_fig = path_save_dir / Path("fid_score.png")
        fig.savefig(path_save_fig, bbox_inches="tight")
        LOGGER.info(f"Saved: {path_save_fig}")


def make_plot_encoding(
    path_save_dir,
    model,
    dec_layers,
    enc_layers,
    stats_all,
    X_test,
    C_test,
    datamodule_name="Gaussian",
    beta=1,
    resample_n=10,
    this_dataloader_color=None,
    save=True,
    mask=True,
) -> None:
    """
    Make some diagnostic plots to visualize encoding of model

    Parameters
    -----------
    path_save_dir: Where to save the plots

    model: Trained model set in eval() mode and loaded on a gpu

    gpu_id: What gpu to put the model on

    dec_layers: List of layers used in CBVAEMLP for decoder

    enc_layers: List of layers used in CBVAEMLP for encoder

    stats_all: Dataframe containing columns
    epoch, total_train_ELBO, total_val_ELBO, total_train_losses,
    total_val_losses

    X_test: The input X to pass through the model. Pass in a batch

    C_test: The condition C to pass through the model. Pass in a batch

    beta: Beta to use to compute loss (default=1)

    resample_n: How many times to sample from the latent space (default=10)

    this_dataloader_color: Color for a colormap plot (default=None)

    save: whether to save or not (default=True)

    mask: Whether to not compute loss when there is missing data or not.
    Default True
    """
    sns.set_context("talk")

    # Number of conditions is the same as the input/output size
    conds = [i for i in range(dec_layers[-1])]

    # Latent dims is the size of last encoder layer
    latent_dims = enc_layers[-1]

    # Plot ELBO (no beta) vs epoch for train and val
    fig, ax = plt.subplots(1, 1, figsize=(7, 5))
    sns.lineplot(ax=ax, data=stats_all, x="epoch", y="total_train_ELBO")
    sns.lineplot(ax=ax, data=stats_all, x="epoch", y="total_val_ELBO")
    ax.set_ylim([0, stats_all.total_val_ELBO.quantile(0.95)])
    ax.legend(["Train loss", "Val loss"])
    ax.set_ylabel("Loss")
    ax.set_title("Actual ELBO (no beta) vs epoch")

    if save is True:
        path_save_fig = path_save_dir / Path("ELBO.png")
        fig.savefig(path_save_fig, bbox_inches="tight")
        LOGGER.info(f"Saved: {path_save_fig}")

    # Plot 4 subplots for encoding
    # Subplot 1 - ELBO (beta*KLD + RCL) vs epoch for train and val
    # Subplot 2 - Scatter plot of mu_1 vs mu_2 (top 2 dims) with condition
    # Subplot 3 - KLD per latent dim with condition
    # Subplot 4 - Mean MSE vs Mean KLD with condition
    fig, (ax1, ax, ax2, ax3) = plt.subplots(1, 4, figsize=(7 * 4, 5))

    # Also make a brokenaxes plot of Subplot 3
    fig2 = plt.figure(figsize=(12, 10))
    bax = brokenaxes(
        xlims=((0, latent_dims - 50), (latent_dims - 4, latent_dims)), hspace=0.15
    )

    if "total_train_losses" in stats_all.columns:
        sns.lineplot(ax=ax1, data=stats_all, x="epoch", y="total_train_losses")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
    if "total_test_losses" in stats_all.columns:
        sns.lineplot(ax=ax1, data=stats_all, x="epoch", y="total_val_losses")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.set_ylim([0, stats_all.total_val_losses.quantile(0.95)])
        ax1.legend(["Train loss", "Val loss"])
    ax1.set_title("ELBO (beta*KLD + RCL) vs epoch")

    this_kwargs = dec_layers[-1]

    conds = [i for i in range(this_kwargs)]

    color = this_dataloader_color

    for i in range(len(conds) + 1):
        if i == 0:
            (
                z_means_x,
                z_means_y,
                kl_per_lt,
                _,
                _,
                kl_vs_rcl,
            ) = visualize_encoder_tabular(
                model,
                conds,
                X_test.clone(),
                C_test.clone(),
                datamodule_name,
                beta,
                resample_n,
                mask,
                kl_per_lt=None,
                kl_vs_rcl=None,
            )
            ax.scatter(z_means_x, z_means_y, marker=".", s=30, label=str(i))
            if color is not None:
                colormap_plot(
                    path_save_dir, X_test.clone(), z_means_x, z_means_y, color, conds
                )
        else:
            (
                z_means_x,
                z_means_y,
                kl_per_lt,
                _,
                _,
                kl_vs_rcl,
            ) = visualize_encoder_tabular(
                model,
                conds,
                X_test.clone(),
                C_test.clone(),
                datamodule_name,
                beta,
                resample_n,
                mask,
                kl_per_lt,
                kl_vs_rcl,
            )
            ax.scatter(z_means_x, z_means_y, marker=".", s=30, label=str(i))
            if color is not None:
                colormap_plot(
                    path_save_dir, X_test.clone(), z_means_x, z_means_y, color, conds
                )
        try:
            conds.pop()
        except:
            pass

    kl_per_lt = pd.DataFrame(kl_per_lt)
    kl_vs_rcl = pd.DataFrame(kl_vs_rcl)

    if save is True:
        path_csv = path_save_dir / Path("encoding_kl_per_lt.csv")
        kl_per_lt.to_csv(path_csv)
        LOGGER.info(f"Saved: {path_csv}")
        path_csv = path_save_dir / Path("encoding_kl_vs_rcl.csv")
        kl_vs_rcl.to_csv(path_csv)
        LOGGER.info(f"Saved: {path_csv}")

    ax.set_title("Latent space")
    ax.legend()

    conds = [i for i in range(this_kwargs)]

    for i in range(len(conds) + 1):
        # if conds = [0,1,2] for a 2D Gaussian (X_test.size()[-1]),
        # then num_conds = 0, so
        # num_conds = X_test.size()[-1] - len(conds)
        tmp = kl_per_lt.loc[kl_per_lt["condition"] == X_test.size()[-1] - len(conds)]
        tmp_2 = kl_vs_rcl.loc[kl_vs_rcl["condition"] == X_test.size()[-1] - len(conds)]
        tmp = tmp.sort_values(by="kl_divergence", ascending=False)
        tmp = tmp.reset_index(drop=True)
        x = tmp.index.values
        y = tmp.iloc[:, 1].values
        sns.lineplot(
            ax=ax2,
            data=tmp,
            x=tmp.index,
            y="kl_divergence",
            label=str(i),
            legend="brief",
        )
        bax.plot(x, y)
        ax3.scatter(tmp_2["RCL"].mean(), tmp_2["KLD"].mean(), label=str(i))
        # sns.scatterplot(
        #     ax=ax3,
        #     data=tmp,
        #     x="rcl",
        #     y="kl_divergence",
        #     label=str(i),
        #     legend='brief'
        #     )
        try:
            conds.pop()
        except:
            pass

    ax2.set_xlabel("Latent dimension")
    ax2.set_ylabel("KLD")
    ax2.set_title("KLD per latent dim (test set)")
    ax3.set_xlabel("MSE")
    ax3.set_ylabel("KLD")
    ax3.set_title("MSE vs KLD (test set)")
    # bax.legend(loc="best")
    bax.set_xlabel("Latent dimension")
    bax.set_ylabel("KLD")
    bax.set_title("KLD per latent dim (test set)")

    conds = [i for i in range(this_kwargs)]
    if len(conds) > 30:
        ax.get_legend().remove()
        ax2.get_legend().remove()

    if save is True:
        path_save_fig = path_save_dir / Path("encoding_test_plots.png")
        fig.savefig(path_save_fig, bbox_inches="tight")
        LOGGER.info(f"Saved: {path_save_fig}")

    if save is True:
        path_save_fig = path_save_dir / Path("brokenaxes_KLD_per_dim.png")
        fig2.savefig(path_save_fig, bbox_inches="tight")
        LOGGER.info(f"Saved: {path_save_fig}")


def colormap_plot(path_save_dir, x_test, z_means_x, z_means_y, color, conds) -> None:

    x_test = x_test.cpu().numpy().astype(np.int32)

    color = color.cpu().numpy().astype(np.int32)

    fig = plt.figure(figsize=(7 * 2, 5))

    ax1 = fig.add_subplot(121, projection="3d")

    try:
        ax1.scatter(x_test[:, 0], x_test[:, 1], x_test[:, 2], c=color)
    except:
        ax1.scatter(x_test[:, 0], x_test[:, 1], c=color)

    ax = fig.add_subplot(122)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05, title="arc length")
    a = ax.scatter(z_means_x, z_means_y, c=color)
    fig.colorbar(a, cax=cax)

    path_save_fig = path_save_dir / Path(
        "latent_space_colormap_conds_" + str(3 - len(conds)) + ".png"
    )
    fig.savefig(path_save_fig, bbox_inches="tight")
    LOGGER.info(f"Saved: {path_save_fig}")
