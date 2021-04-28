import logging
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
import math
from .metric_utils import compute_generative_metric_tabular
from .metric_utils import visualize_encoder_tabular
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
from brokenaxes import brokenaxes
from aicscytoparam import cytoparam
from pytorch_lightning import LightningModule, Trainer
from .model_utils import to_device
from .mesh_utils import get_mesh_from_series

from sklearn.decomposition import PCA

LOGGER = logging.getLogger(__name__)


def plot_bin_count_table(
    bin_counts: pd.DataFrame,
    save_dir,
):

    # sns.set_context('talk')
    fig, ax = plt.subplots()

    # hide axes
    fig.patch.set_visible(False)
    ax.axis("off")
    ax.axis("tight")

    table = ax.table(
        cellText=bin_counts.values,
        colLabels=bin_counts.columns,
        rowLabels=bin_counts.index,
        colWidths=[0.2] * bin_counts.shape[1],
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(13)
    table.scale(2, 2)

    fig.savefig(save_dir / "bin_counts.png", bbox_inches="tight")


def make_embedding_pairplots(
    all_embeddings: pd.DataFrame,
    fitted_pca: PCA,
    n_components: int,
    ranked_z_dim_list: list,
    model,
    save_dir,
    cond_size,
):

    pca_std = np.sqrt(fitted_pca.explained_variance_)

    ranked_ixs = ranked_z_dim_list[:n_components]
    ranked_cols = [f"mu_{j}" for j in ranked_z_dim_list[:n_components]]

    mus = all_embeddings[[col for col in all_embeddings.columns
                          if col not in ("CellId", "split")]]
    mus_mean = mus.values.mean(axis=0)
    mus_std = mus.values.std(axis=0)
    mus = (mus - mus_mean)/mus_std
    mus = mus[ranked_cols]

    walk_points = [-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2]
    for pc in range(n_components):
        walk_std = np.zeros_like(pca_std)
        walk_std[pc] = pca_std[pc]

        spharm_walk = fitted_pca.inverse_transform([n * walk_std for n in walk_points])

        with torch.no_grad():
            _, latent_walk, _, _, _, _, _, _ = model(
                torch.tensor(spharm_walk).float(),
                torch.zeros((len(spharm_walk), cond_size)),
            )

        sns.set_context("talk")
        #g = sns.pairplot(mus, corner=True,
        #                 kind="hist",
        #                 plot_kws=dict(bins=100, color="grey"))
        #                 #plot_kws=dict(s=5, alpha=0.2, color="grey"))

        latent_walk = (latent_walk - mus_mean)/mus_std

        g = sns.PairGrid(mus, corner=True, diag_sharey=True)
        g.map_lower(sns.histplot, cmap="Reds", binrange=((-3, 3),(-3, 3)), bins=100)
        g.map_diag(sns.histplot, binrange=(-3, 3))

        for row_ix, ax_row in enumerate(g.axes[1:]):
            row_ix += 1
            for col_ix in range(row_ix):
                ax = g.axes[row_ix][col_ix]
                ax.set_xlim(-3, 3)
                ax.set_ylim(-3, 3)
                x_ix = ranked_ixs[col_ix]
                y_ix = ranked_ixs[row_ix]

                ax.plot(latent_walk[:, x_ix], latent_walk[:, y_ix])
                sc = ax.scatter(
                    latent_walk[:, x_ix], latent_walk[:, y_ix], c=walk_points
                )

        cax = g.fig.add_axes([0.85, 0.85, 0.01, 0.1])
        g.fig.colorbar(sc, cax=cax)
        g.savefig(save_dir / f"pairplot_embeddings_PC_{pc + 1}.png")


def make_pca_pairplots(
    all_embeddings: pd.DataFrame,
    fitted_pca: PCA,
    all_pcs: pd.DataFrame,
    n_components: int,
    ranked_z_dim_list: list,
    model,
    save_dir,
    cond_size,
):

    ranked_ixs = ranked_z_dim_list[:n_components]
    ranked_cols = [f"mu_{j}" for j in ranked_z_dim_list[:n_components]]
    walk_points = [-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2]
    mus = all_embeddings[[col for col in all_embeddings.columns
                          if "mu_" in col]].values

    walk_mu = mus.mean(axis=0)
    walk_std = mus.std(axis=0)

    all_pcs = all_pcs[[col for col in all_pcs.columns if col != "CellId"]]
    pca_mean = all_pcs.values.mean(axis=0)
    pca_std = all_pcs.values.std(axis=0)
    all_pcs = (all_pcs - pca_mean)/pca_std

    for dim in ranked_ixs:
        latent_walk = torch.stack(
            [torch.tensor(walk_mu + walk_std[dim] * n) for n in walk_points]
        ).float()

        with torch.no_grad():
            spharm_walk = model.decoder(
                latent_walk, torch.zeros((len(latent_walk), cond_size))
            )

        pca_walk = fitted_pca.transform(spharm_walk)
        pca_walk = (pca_walk - pca_mean)/pca_std

        sns.set_context('talk')
        g = sns.pairplot(all_pcs[all_pcs.columns[:n_components]],
                         corner=True,
                         kind="hist",
                         plot_kws=dict(bins=100, cmap="Reds",
                                       binrange=((-3, 3), (-3, 3))))

        for row_ix, ax_row in enumerate(g.axes[1:]):
            row_ix += 1
            for col_ix in range(row_ix):
                ax = g.axes[row_ix][col_ix]
                ax.set_xlim(-3, 3)
                ax.set_ylim(-3, 3)
                x_ix = ranked_ixs[col_ix]
                y_ix = ranked_ixs[row_ix]
                ax.plot(pca_walk[:, x_ix], pca_walk[:, y_ix])
                sc = ax.scatter(pca_walk[:, x_ix], pca_walk[:, y_ix], c=walk_points)

        cax = g.fig.add_axes([0.85, 0.85, 0.01, 0.1])
        g.fig.colorbar(sc, cax=cax)
        g.savefig(save_dir / f"pairplot_embeddings_latent_{dim}.png")


def decode_latent_walk_closest_cells(
    trainer: Trainer,
    pl_module: LightningModule,
    closest_cells_df: pd.DataFrame,
    all_embeddings: pd.DataFrame,
    ranked_z_dim_list: list,
    mu_std_list: list,
    dir_path,
    path_in_stdv,
    c_shape,
    dna_spharm_cols,
):
    batch_size = trainer.test_dataloaders[0].batch_size
    latent_dims = pl_module.encoder.enc_layers[-1]

    for index, z_dim in enumerate(ranked_z_dim_list):
        # Set subplots
        fig, ax_array = plt.subplots(
            3,
            len(path_in_stdv),
            squeeze=False,
            figsize=(15, 7),
        )
        subset_df = closest_cells_df.loc[closest_cells_df["ranked_dim"] == z_dim]
        subset_df = subset_df.drop_duplicates()
        print(subset_df)
        mu_std = mu_std_list[index]

        for loc_index, location in enumerate(subset_df["location"].unique()):
            subset_sub_df = subset_df.loc[subset_df["location"] == location]
            subset_sub_df = subset_sub_df.reset_index(drop=True)
            this_cell_embedding = all_embeddings.loc[
                all_embeddings["CellId"] == subset_sub_df["CellId"].item()
            ]

            z_inf = torch.zeros(batch_size, latent_dims)
            walk_cols = [i for i in this_cell_embedding.columns if "mu" in i]

            this_cell_id = subset_sub_df.iloc[0]["CellId"]

            for walk in walk_cols:
                # walk_cols is mu_{dim}, so walk[3:]
                z_inf[:, int(walk[3:])] = torch.from_numpy(
                    np.array(this_cell_embedding.iloc[0][walk])
                )
            z_inf = z_inf.cuda(device=0)
            z_inf = z_inf.float()
            decoder = pl_module.decoder

            y_test = torch.zeros(batch_size, c_shape)

            z_inf, y_test = to_device(z_inf, y_test, pl_module.device)
            x_hat = decoder(z_inf, y_test)

            test_spharm = x_hat[0, :]
            test_spharm = test_spharm.detach().cpu().numpy()
            test_spharm = pd.DataFrame(test_spharm).T
            test_spharm.columns = dna_spharm_cols
            test_spharm_series = test_spharm.iloc[0]

            mesh = get_mesh_from_series(test_spharm_series, "dna", 32)
            img, origin = cytoparam.voxelize_meshes([mesh])

            for proj in [0, 1, 2]:
                plt.style.use("dark_background")
                ax_array[proj, loc_index].set_title(
                    f"{path_in_stdv[loc_index]} $\sigma$  \n ID {this_cell_id}",
                    fontsize=14,
                )
                ax_array[proj, loc_index].imshow(img.max(proj), cmap="gray")

                ax_array[proj, loc_index].set_xlim([0, 140])
                ax_array[proj, loc_index].set_ylim([0, 120])
                for tick in ax_array[proj, loc_index].xaxis.get_major_ticks():
                    tick.label.set_fontsize(5)
                for tick in ax_array[proj, loc_index].yaxis.get_major_ticks():
                    tick.label.set_fontsize(5)
                plt.style.use("default")

        # [ax.axis("off") for ax in ax_array.flatten()]
        # Save figure
        ax_array.flatten()[0].get_figure().savefig(
            dir_path / f"dim_{z_dim}_rank_{index + 1}_closest_real_cell.png"
        )
        # Close figure, otherwise clogs memory
        plt.close(fig)


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
    conds_list,
    datamodule,
    value,
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

    conds_list: List of all conditions
    In the case of Gaussian datamodule, this specifies
    which columns in condition to set to 0
    In the case of Spharm datamodule, this specifies
    which integer to provide as condition

    value: int - what value to set condition to.

    beta: Beta to use to compute loss (default=1)

    resample_n: How many times to sample from the latent space (default=10)

    this_dataloader_color: Color for a colormap plot (default=None)

    save: whether to save or not (default=True)

    mask: Whether to not compute loss when there is missing data or not.
    Default True
    """
    sns.set_context("talk")

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
    if latent_dims > 54:
        fig2 = plt.figure(figsize=(12, 10))
        bax = brokenaxes(
            xlims=((0, latent_dims - 50), (latent_dims - 4, latent_dims)), hspace=0.15
        )
    else:
        fig2, bax = plt.subplots(1, 1, figsize=(12, 10))

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

    color = this_dataloader_color

    for i in range(len(conds_list)):
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
                conds_list[i],
                X_test.clone(),
                C_test.clone(),
                datamodule,
                value,
                beta,
                resample_n,
                mask,
                kl_per_lt=None,
                kl_vs_rcl=None,
            )
            ax.scatter(z_means_x, z_means_y, marker=".", s=30, label=str(conds_list[i]))
            if color is not None:
                colormap_plot(
                    path_save_dir,
                    X_test.clone(),
                    z_means_x,
                    z_means_y,
                    color,
                    conds_list[i],
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
                conds_list[i],
                X_test.clone(),
                C_test.clone(),
                datamodule,
                value,
                beta,
                resample_n,
                mask,
                kl_per_lt,
                kl_vs_rcl,
            )
            ax.scatter(z_means_x, z_means_y, marker=".", s=30, label=str(conds_list[i]))
            if color is not None:
                colormap_plot(
                    path_save_dir,
                    X_test.clone(),
                    z_means_x,
                    z_means_y,
                    color,
                    conds_list[i],
                )

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

    for i in range(len(conds_list)):
        # if conds = [0,1,2] for a 2D Gaussian (X_test.size()[-1]),
        # then num_conds = 0, so
        # num_conds = X_test.size()[-1] - len(conds)

        tmp = kl_per_lt.loc[kl_per_lt["condition"] == str(conds_list[i])]
        tmp_2 = kl_vs_rcl.loc[kl_vs_rcl["condition"] == str(conds_list[i])]
        tmp = tmp.sort_values(by="kl_divergence", ascending=False)
        tmp = tmp.reset_index(drop=True)
        tmp["explained_variance"] = (
            tmp["kl_divergence"] / tmp["kl_divergence"].sum()
        ) * 100
        x = tmp.index.values
        y = tmp.iloc[:, 1].values
        sns.lineplot(
            ax=ax2,
            data=tmp,
            x=tmp.index,
            y="kl_divergence",
            label=str(conds_list[i]),
            legend="brief",
        )
        bax.plot(x, y)
        ax3.scatter(tmp_2["RCL"].mean(), tmp_2["KLD"].mean(), label=str(i))

        # Make plot for explained variance
        fig_explained_var, ax_explained_var = plt.subplots(1, 1, figsize=(7, 5))
        sns.lineplot(
            ax=ax_explained_var,
            data=tmp,
            x=tmp.index,
            y="explained_variance",
            label=str(conds_list[i]),
            legend="brief",
        )
        ax_explained_var.set_xlabel("Latent dimension")
        ax_explained_var.set_ylabel("(KLD per dim/Total KLD) * 100")
        ax_explained_var.set_title("Explained Variance")

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

    if len(conds_list) > 30:
        ax.get_legend().remove()
        ax2.get_legend().remove()

    if save is True:
        path_save_fig = path_save_dir / Path(f"encoding_test_plots_value_{value}.png")
        fig.savefig(path_save_fig, bbox_inches="tight")
        LOGGER.info(f"Saved: {path_save_fig}")

        path_save_fig = path_save_dir / Path(f"brokenaxes_KLD_per_dim_{value}.png")
        fig2.savefig(path_save_fig, bbox_inches="tight")
        LOGGER.info(f"Saved: {path_save_fig}")

        path_save_fig = path_save_dir / Path(f"explained_variance_{value}.png")
        fig_explained_var.savefig(path_save_fig, bbox_inches="tight")
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
