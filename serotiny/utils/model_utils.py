import inspect
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as opt
from typing import Sequence, Tuple, Union
from torch import device, Tensor
import pytorch_lightning as pl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pytorch_lightning import LightningModule
from cvapipe_analysis.steps.pca_path_cells.utils import scan_pc_for_cells
from tqdm import trange
from tqdm import tqdm
import math
import random
from scipy.stats import multivariate_normal


def to_device(
    batch_x: Sequence, batch_y: Sequence, device: Union[str, device]
) -> Tuple[Tensor, Tensor]:

    # last input is for online eval
    batch_x = batch_x.to(device)
    batch_y = batch_y.to(device)

    return batch_x, batch_y


def get_ranked_dims(
    dir_path,
    cutoff_kld_per_dim,
    max_num_shapemodes,
):
    stats = pd.read_csv(dir_path / "stats_per_dim_test.csv")

    stats = (
        stats.loc[stats["test_kld_per_dim"] > 0]
        .sort_values(by=["test_kld_per_dim"])
        .reset_index(drop=True)
    )

    mu_mean_list = [i for i in stats["mu_mean_per_dim"][::-1]]

    stats = (
        stats.loc[stats["test_kld_per_dim"] > cutoff_kld_per_dim]
        .sort_values(by=["test_kld_per_dim"])
        .reset_index(drop=True)
    )

    ranked_z_dim_list = [i for i in stats["dimension"][::-1]]
    mu_std_list = [i for i in stats["mu_std_per_dim"][::-1]]

    if len(ranked_z_dim_list) > max_num_shapemodes:
        ranked_z_dim_list = ranked_z_dim_list[:max_num_shapemodes]
        mu_std_list = mu_std_list[:max_num_shapemodes]

    return ranked_z_dim_list, mu_std_list, mu_mean_list


def get_all_embeddings(
    train_dataloader,
    val_dataloader,
    test_dataloader,
    pl_module: LightningModule,
    x_label: str,
    c_label: str,
    id_fields: list,
):

    all_embeddings = []
    cell_ids = []
    split = []

    zip_iter = zip(
        ["train", "val", "test"], [train_dataloader, val_dataloader, test_dataloader]
    )

    with torch.no_grad():
        for split_name, dataloader in zip_iter:
            for batch in dataloader:
                input_x = batch[x_label]
                cond_c = batch[c_label]
                cell_id = batch["id"][id_fields[0]]

                _, mus, _, _, _, _, _, _ = pl_module(input_x.float(), cond_c.float())
                all_embeddings.append(mus)

                cell_ids.append(cell_id)
                split.append([split_name] * mus.shape[0])

    all_embeddings = torch.cat(all_embeddings, dim=0)
    cell_ids = torch.cat(cell_ids, dim=0)
    split = [item for sublist in split for item in sublist]
    all_embeddings = all_embeddings.cpu().numpy()

    df1 = pd.DataFrame(
        all_embeddings, columns=[f"mu_{i}" for i in range(all_embeddings.shape[1])]
    )
    df2 = pd.DataFrame(cell_ids, columns=["CellId"])
    df3 = pd.DataFrame(split, columns=["split"])
    frames = [df1, df2, df3]
    result = pd.concat(frames, axis=1)

    return result


def find_outliers(
    ranked_z_dim_list: list,
    test_embeddings: pd.DataFrame,
    bins: list,
):
    for dim in ranked_z_dim_list:
        mu_array = np.array(test_embeddings[[f"mu_{dim}"]]).astype(np.float32)
        mu_array -= mu_array.mean()
        mu_array /= mu_array.std()

        binw = 0.5 * np.diff(bins).mean()
        bin_edges = np.unique([(b - binw, b + binw) for b in bins])

        inds = np.digitize(mu_array, bin_edges)
        # Find outliers per dim and add to embeddings
        left_outliers = np.where(inds.flatten() == 0)
        right_outliers = np.where(inds.flatten() == len(bin_edges))
        outlier_col = np.zeros(test_embeddings.shape[0], dtype="object")

        if left_outliers[0].size > 0:
            outlier_col[left_outliers[0]] = "Left Outlier"
            inds[left_outliers[0]] = 0  # Map left outlier to bin 1
        if right_outliers[0].size > 0:
            outlier_col[right_outliers[0]] = "Right Outlier"
            inds[right_outliers[0]] = (
                len(bin_edges) - 1
            )  # Map right outlier to last bin

        outlier_col[np.where(outlier_col == 0)] = False
        test_embeddings[f"outliers_mu_{dim}"] = outlier_col

    return test_embeddings


def get_bins_for_each_cell(
    ranked_z_dim_list: list,
    test_embeddings: pd.DataFrame,
    bins: list,
):

    for dim in ranked_z_dim_list:
        mu_array = np.array(test_embeddings[[f"mu_{dim}"]]).astype(np.float32)
        mu_array -= mu_array.mean()
        mu_array /= mu_array.std()

        binw = 0.5 * np.diff(bins).mean()
        bin_edges = np.unique([(b - binw, b + binw) for b in bins])
        bin_edges[0] = -np.inf
        bin_edges[-1] = np.inf

        inds = np.digitize(mu_array, bin_edges)

        # Add bins per dim to embeddings
        bin_values = []
        for i in inds:
            bin_values.append((bin_edges[i - 1].item(), bin_edges[i].item()))
        test_embeddings[f"bins_mu_{dim}"] = bin_values

    bin_embeddings = test_embeddings[[i for i in test_embeddings.columns if "bin" in i]]

    ranked_z_dim_bin_counts = []
    for dim in ranked_z_dim_list:
        bin_counts = bin_embeddings.pivot_table(
            index=[f"bins_mu_{dim}"], aggfunc="size"
        )
        bin_counts = bin_counts.to_frame(f"bin_count_mu_{dim}")

        ranked_z_dim_bin_counts.append(bin_counts)

    all_dim_bin_counts = pd.concat(ranked_z_dim_bin_counts, axis=1)

    return test_embeddings, all_dim_bin_counts


def get_closest_cells(
    ranked_z_dim_list,
    mu_std_list,
    all_embeddings,
    dir_path,
    path_in_stdv,
    metric,
    id_col,
    N_cells,
):
    embeddings_most_important_dims = all_embeddings[
        [f"mu_{i}" for i in ranked_z_dim_list]
    ]

    dist_cols = embeddings_most_important_dims.columns

    df_list = []
    dims = []
    for index, dim in enumerate(ranked_z_dim_list):
        mu_std = mu_std_list[index]
        df_cells = scan_pc_for_cells(
            all_embeddings,
            pc=index + 1,  # This function assumes first index is 1
            path=np.array(path_in_stdv) * mu_std,
            dist_cols=dist_cols,
            metric=metric,
            id_col=id_col,
            N_cells=N_cells,
        )
        dims.append([dim] * df_cells.shape[0])
        df_list.append(df_cells)

    tmp = pd.concat(df_list)
    tmp = tmp.reset_index(drop=True)
    dims = [item for sublist in dims for item in sublist]
    df2 = pd.DataFrame(dims, columns=["ranked_dim"])
    result = pd.concat([tmp, df2], axis=1)

    # import ipdb
    # ipdb.set_trace()

    path = dir_path / "closest_real_cells_to_top_dims.csv"

    if path.exists():
        result.to_csv(path, header="column_names", index=False)

    return result


def acc_prec_recall(n_classes):
    """
    util function to instantiate a ModuleDict for metrics
    """
    return nn.ModuleDict(
        {
            "accuracy": pl.metrics.Accuracy(),
            "precision": pl.metrics.Precision(num_classes=n_classes, average="macro"),
            "recall": pl.metrics.Recall(num_classes=n_classes, average="macro"),
        }
    )


def matplotlib_imshow(img, one_channel=False):
    """
    Plot image via matplotlib's imshow
    """
    if one_channel:
        img = img.mean(dim=0)
    img = img / 2 + 0.5  # unnormalize
    npimg = img.cpu().numpy()
    if one_channel:
        plt.imshow(npimg, cmap="Greys")
    else:
        plt.imshow(np.transpose(npimg, (1, 2, 0)))


def images_to_probs(net, images):
    """
    Generates predictions and corresponding probabilities from a trained
    network and a list of images
    """
    output = net(images)
    # convert output probabilities to predicted class
    _, preds_tensor = torch.max(output, 1)
    preds = np.squeeze(preds_tensor.cpu().numpy())
    return preds, [F.softmax(el, dim=0)[i].item() for i, el in zip(preds, output)]


def plot_classes_preds(net, images, labels, classes):
    """
    Generates matplotlib Figure using a trained network, along with images
    and labels from a batch, that shows the network's top prediction along
    with its probability, alongside the actual label, coloring this
    information based on whether the prediction was correct or not.
    Uses the "images_to_probs" function.
    """
    preds, probs = images_to_probs(net, images)

    labels = list(labels.cpu().numpy())
    labels = [int(i) for i in labels]
    # plot the images in the batch, along with predicted and true labels
    fig = plt.figure(figsize=(20, 15))
    for idx in np.arange(4):
        ax = fig.add_subplot(1, 4, idx + 1, xticks=[], yticks=[])
        matplotlib_imshow(images[idx], one_channel=True)
        ax.set_title(
            "{0}, {1:.1f}%\n(label: {2})".format(
                classes[preds[idx]],
                probs[idx] * 100.0,
                classes[labels[idx]],
            ),
        )
    return fig


# helper function
def add_pr_curve_tensorboard(
    logger, classes, class_index, test_probs, test_preds, global_step=0, name="_val"
):
    """
    Takes in a "class_index" and plots the corresponding
    precision-recall curve
    """
    tensorboard_preds = test_preds == class_index
    tensorboard_probs = test_probs[:, class_index]

    logger.experiment.add_pr_curve(
        str(classes[class_index]) + name,
        tensorboard_preds,
        tensorboard_probs,
        global_step=global_step,
    )
    logger.experiment.close()


def show_activations(x, logger, network, current_epoch):
    """
    Currently parametrized only for the Resnet18 Network
    """
    # logging reference image
    for i in range(4):
        logger.experiment.add_image(
            "input",
            torch.Tensor.cpu(x[i][0]),
            global_step=current_epoch,
            dataformats="HW",
        )

    first_conv_output = network.feature_extractor_first_layer(x)

    for i in range(4):  # Log 4 images per epoch
        logger.experiment.add_image(
            "first_conv",
            torch.Tensor.cpu(first_conv_output[i][0]),
            global_step=current_epoch,
            dataformats="HW",
        )

    # logging first convolution activations
    second_convolution = nn.Sequential(
        OrderedDict(
            [
                ("conv1", network.feature_extractor.conv1),
                ("bn1", network.feature_extractor.bn1),
                ("relu", network.feature_extractor.relu),
            ]
        )
    )
    second_convolution_output = second_convolution(first_conv_output)
    # img_grid = torchvision.utils.make_grid(out)
    for i in range(4):  # Log 4 images per epoch
        logger.experiment.add_image(
            "second_conv",
            torch.Tensor.cpu(second_convolution_output[i][0]),
            global_step=current_epoch,
            dataformats="HW",
        )

    # logging classifier activations
    # logging first convolution activations
    basic_block = nn.Sequential(
        OrderedDict(
            [
                ("maxpool", network.feature_extractor.maxpool),
                ("layer1", network.feature_extractor.layer1),
            ]
        )
    )
    basic_block_output = basic_block(second_convolution_output)
    # img_grid_classifier = torchvision.utils.make_grid(out_classifier)
    for i in range(4):
        logger.experiment.add_image(
            "first_basic_block",
            torch.Tensor.cpu(basic_block_output[i][0]),
            global_step=current_epoch,
            dataformats="HW",
        )


def index_to_onehot(index, n_classes):
    index = index.long()

    onehot = torch.zeros(len(index), n_classes).type_as(index).float()
    onehot.scatter_(1, index, 1)

    return onehot


def log_metrics(outputs, prefix, current_epoch, dir_path):
    """
    Produce metrics and save them
    """

    dataframe = {
        "epoch": [],
        f"total_{prefix}_losses": [],
        f"total_{prefix}_ELBO": [],
        f"total_{prefix}_rcl": [],
        f"total_{prefix}_klds": [],
        "condition": [],
        f"{prefix}_rcl": [],
        f"{prefix}_kld": [],
    }

    batch_rcl, batch_kld, batch_mu = (
        torch.empty([0]),
        torch.empty([0]),
        torch.empty([0]),
    )

    batch_rcl = batch_rcl.type_as(outputs[0]["rcl_per_elem"])
    batch_kld = batch_kld.type_as(outputs[0]["rcl_per_elem"])
    batch_mu = batch_mu.type_as(outputs[0]["rcl_per_elem"])

    all_rcl, all_kld, all_mu = (
        torch.empty([0]),
        torch.empty([0]),
        torch.empty([0]),
    )

    all_rcl = all_rcl.type_as(outputs[0]["rcl_per_elem"])
    all_kld = all_kld.type_as(outputs[0]["rcl_per_elem"])
    all_mu = all_mu.type_as(outputs[0]["rcl_per_elem"])

    batch_length = 0
    num_data_points = 0
    loss, rcl_loss, kld_loss = 0, 0, 0

    total_x = torch.cat([x["input"] for x in outputs])
    # num_batches = len(total_x)

    rcl_per_condition_loss, kld_per_condition_loss = (
        torch.zeros(total_x.size()[-1] + 1),
        torch.zeros(total_x.size()[-1] + 1),
    )

    rcl_per_condition_loss = rcl_per_condition_loss.type_as(outputs[0]["rcl_per_elem"])
    kld_per_condition_loss = kld_per_condition_loss.type_as(outputs[0]["rcl_per_elem"])

    for output in outputs:
        rcl_per_element = output["rcl_per_elem"]
        kld_per_element = output["kld_per_elem"]
        mu_per_elem = output["mu_per_elem"]

        loss += output[f"{prefix}_loss"].detach()
        rcl_loss += output["recon_loss"].detach()
        kld_loss += output["kld_loss"].detach()
        for jj, ii in enumerate(torch.unique(output["cond_labels"])):
            this_cond_positions = output["cond_labels"] == ii
            batch_rcl = torch.cat(
                [
                    batch_rcl.type_as(rcl_per_element),
                    torch.sum(rcl_per_element[this_cond_positions], dim=0).view(1, -1),
                ],
                0,
            )
            batch_kld = torch.cat(
                [
                    batch_kld.type_as(kld_per_element),
                    torch.sum(kld_per_element[this_cond_positions], dim=0).view(1, -1),
                ],
                0,
            )
            batch_mu = torch.cat(
                [
                    batch_mu.type_as(mu_per_elem),
                    torch.sum(mu_per_elem[this_cond_positions], dim=0).view(1, -1),
                ],
                0,
            )

            all_rcl = torch.cat(
                [
                    all_rcl.type_as(rcl_per_element),
                    rcl_per_element[this_cond_positions],
                ],
                0,
            )
            all_kld = torch.cat(
                [
                    all_kld.type_as(kld_per_element),
                    kld_per_element[this_cond_positions],
                ],
                0,
            )
            all_mu = torch.cat(
                [all_mu.type_as(mu_per_elem), mu_per_elem[this_cond_positions]], 0
            )

            this_batch_size = rcl_per_element.shape[0]
            num_data_points += this_batch_size
            batch_length += 1

            this_cond_rcl = torch.sum(rcl_per_element[this_cond_positions])
            this_cond_kld = torch.sum(kld_per_element[this_cond_positions])

            rcl_per_condition_loss[jj] += this_cond_rcl.detach()
            kld_per_condition_loss[jj] += this_cond_kld.detach()

    # loss = loss / num_batches
    # rcl_loss = rcl_loss / num_batches
    # kld_loss = kld_loss / num_batches
    # rcl_per_condition_loss = rcl_per_condition_loss / num_batches
    # kld_per_condition_loss = kld_per_condition_loss / num_batches

    loss = loss / num_data_points
    rcl_loss = rcl_loss / num_data_points
    kld_loss = kld_loss / num_data_points
    rcl_per_condition_loss = rcl_per_condition_loss / num_data_points
    kld_per_condition_loss = kld_per_condition_loss / num_data_points

    # Save metrics averaged across all batches and dimension per condition
    for j in range(len(torch.unique(output["cond_labels"]))):
        dataframe["epoch"].append(current_epoch)
        dataframe["condition"].append(j)
        dataframe[f"total_{prefix}_losses"].append(loss.item())
        dataframe[f"total_{prefix}_ELBO"].append(rcl_loss.item() + kld_loss.item())
        dataframe[f"total_{prefix}_rcl"].append(rcl_loss.item())
        dataframe[f"total_{prefix}_klds"].append(kld_loss.item())
        dataframe[f"{prefix}_rcl"].append(rcl_per_condition_loss[j].item())
        dataframe[f"{prefix}_kld"].append(kld_per_condition_loss[j].item())

    stats = pd.DataFrame(dataframe)

    path = dir_path / f"stats_{prefix}.csv"
    if prefix == "train":
        print(f"====> {prefix}")
        print("====> Epoch: {} losses: {:.4f}".format(current_epoch, loss))
        print("====> RCL loss: {:.4f}".format(rcl_loss))
        print("====> KLD loss: {:.4f}".format(kld_loss))

    if path.exists():
        stats.to_csv(path, mode="a", header=False, index=False)
    else:
        stats.to_csv(path, header="column_names", index=False)

    if prefix == "test":

        # Save test KLD and variance of mu per dimension, condition
        # This is averaged across batch

        dataframe2 = {
            "dimension": [],
            "test_kld_per_dim": [],
            "condition": [],
            "mu_std_per_dim": [],
            "mu_mean_per_dim": [],
            "explained_variance": [],
        }

        for j in range(len(torch.unique(output["cond_labels"]))):

            # this_cond_per_batch_kld = batch_kld[
            #     j :: len(torch.unique(output["cond_labels"])), :
            # ]
            # this_cond_per_batch_mu = batch_mu[
            #     j :: len(torch.unique(output["cond_labels"])), :
            # ]

            this_cond_per_element_kld = all_kld[
                j :: len(torch.unique(output["cond_labels"])), :
            ]
            this_cond_per_element_mu = all_mu[
                j :: len(torch.unique(output["cond_labels"])), :
            ]

            # summed_kld = torch.sum(this_cond_per_batch_kld, dim=0) / batch_length
            # dim_var = torch.std(this_cond_per_batch_mu, dim=0)
            summed_kld = torch.sum(this_cond_per_element_kld, dim=0) / num_data_points
            dim_std = torch.std(this_cond_per_element_mu, dim=0)
            dim_mean = torch.mean(this_cond_per_element_mu, dim=0)

            summed_summed_kld = torch.sum(summed_kld)

            for k in range(len(summed_kld)):
                dataframe2["dimension"].append(k)
                dataframe2["condition"].append(
                    torch.unique(output["cond_labels"])[j].item()
                )
                dataframe2["test_kld_per_dim"].append(summed_kld[k].item())
                dataframe2["mu_std_per_dim"].append(dim_std[k].item())
                dataframe2["mu_mean_per_dim"].append(dim_mean[k].item())
                dataframe2["explained_variance"].append(
                    (summed_kld[k].item() / summed_summed_kld.item()) * 100
                )

        stats_per_dim = pd.DataFrame(dataframe2)

        path2 = dir_path / f"stats_per_dim_{prefix}.csv"
        if path2.exists():
            stats_per_dim.to_csv(path2, mode="a", header=False, index=False)
        else:
            stats_per_dim.to_csv(path2, header="column_names", index=False)

        # Get ranked Z dim list
        stats_per_dim_ranked = (
            stats_per_dim.loc[stats_per_dim["test_kld_per_dim"] > 0.05]
            .sort_values(by=["test_kld_per_dim"])
            .reset_index(drop=True)
        )

        ranked_z_dim_list = [i for i in stats_per_dim_ranked["dimension"][::-1]]

        # Save mu per element, dimension and conditon
        # Not averaged across batch
        dataframe3 = {
            "dimension": [],
            "mu": [],
            "element": [],
            "condition": [],
        }

        for j in range(len(torch.unique(output["cond_labels"]))):

            # this_cond_mu = batch_mu[j :: len(torch.unique(output["cond_labels"])), :]
            this_cond_per_element_mu = all_mu[
                j :: len(torch.unique(output["cond_labels"])), :
            ]

            for element in range(this_cond_per_element_mu.shape[0]):
                for dimension in range(this_cond_per_element_mu.shape[1]):
                    dataframe3["dimension"].append(dimension)
                    dataframe3["element"].append(element)
                    dataframe3["condition"].append(
                        torch.unique(output["cond_labels"])[j].item()
                    )
                    dataframe3["mu"].append(
                        this_cond_per_element_mu[element, dimension].item()
                    )

        mu_per_elem_and_dim = pd.DataFrame(dataframe3)

        path3 = dir_path / f"mu_per_elem_and_dim_{prefix}.csv"
        if path3.exists():
            mu_per_elem_and_dim.to_csv(path3, mode="a", header=False, index=False)
        else:
            mu_per_elem_and_dim.to_csv(path3, header="column_names", index=False)

        # Make correlation plot between top 20 latent dims
        mu_per_elem_and_dim = mu_per_elem_and_dim[["dimension", "mu", "element"]]
        table = pd.pivot_table(
            mu_per_elem_and_dim, values="mu", index=["element"], columns=["dimension"]
        )
        mu_corrs = table[ranked_z_dim_list[:20]].corr()

        plt.figure(figsize=(8, 8))
        sns.set_context("talk")
        sns.heatmap(
            mu_corrs.abs(),
            cmap="RdBu_r",
            vmin=-1,
            vmax=1,
            xticklabels=True,
            yticklabels=True,
            square=True,
            cbar_kws={"shrink": 0.82},
        )
        plt.savefig(dir_path / "latent_dim_corrs.png", dpi=300, bbox_inches="tight")

        path_all = dir_path / "stats_all.csv"
        all_stats = pd.DataFrame()
        for prefix in ["train", "val", "test"]:
            stats = pd.read_csv(dir_path / f"stats_{prefix}.csv")
            all_stats = all_stats.append(stats)

        if path_all.exists():
            all_stats.to_csv(path_all, mode="a", header=False, index=False)
        else:
            all_stats.to_csv(path_all, header="column_names", index=False)

    # return outputs


def find_optimizer(optimizer_name):
    """
    Given optimizer name, get it from torch.optim
    """
    available_optimizers = []
    for cls_name, cls in opt.__dict__.items():
        if inspect.isclass(cls):
            if issubclass(cls, opt.Optimizer):
                available_optimizers.append(cls_name)

    if optimizer_name in available_optimizers:
        optimizer_class = opt.__dict__[optimizer_name]
    else:
        raise KeyError(
            f"optimizer {optimizer_name} not available, "
            f"options are {available_optimizers}"
        )
    return optimizer_class


def find_lr_scheduler(scheduler_name):
    """
    Given scheduler name, get it from torch.optim.lr_scheduler
    """
    available_schedulers = []
    for cls_name, cls in opt.lr_scheduler.__dict__.items():
        if inspect.isclass(cls):
            if "LR" in cls_name and cls_name[0] != "_":
                available_schedulers.append(cls_name)

    if scheduler_name in available_schedulers:
        scheduler_class = opt.lr_scheduler.__dict__[scheduler_name]
    else:
        raise Exception(
            f"scheduler {scheduler_name} not available, "
            f"options are {available_schedulers}"
        )
    return scheduler_class


def q_batch(z, mus, logvars):
    """
    Compute a batch contribution to marginal q, where marginal q is

    q(z) = 1/N sum(q(z | x_n)).

    This function computes *unnormalized* contributions to that sum

    """
    numerator = (z - mus) ** 2 / (logvars.exp())
    numerator = -0.5 * numerator.sum(axis=1)
    numerator = torch.exp(numerator)

    # equivalent to product of all sigma-squares
    denominator = torch.sum(logvars, axis=1).exp()
    denominator *= math.pow(2 * math.pi, mus.shape[1])
    denominator = torch.sqrt(denominator)

    return (numerator / denominator).sum()


def marginal_kl(
    model,
    dataloader,
    prior_mean,
    prior_logvar,
    n_samples,
    x_label,
    c_label,
    verbose=True,
):

    prior = multivariate_normal(mean=prior_mean, cov=np.exp(prior_logvar))

    dataset_size = len(dataloader.dataset)

    with torch.no_grad():
        total_marginal_kl = 0
        if verbose:
            sample_iter = trange(n_samples, desc="all samples")
        else:
            sample_iter = range(n_samples)

        for i in sample_iter:
            sample_ix = random.randint(0, dataset_size)

            # import ipdb

            # ipdb.set_trace()
            sample = dataloader.dataset.datasets[sample_ix]

            this_x, this_c = to_device(
                sample[x_label].float().unsqueeze(0),
                sample[c_label].float().unsqueeze(0),
                model.device,
            )

            _, z_s, _, _, _, _, _, _ = model.forward(
                this_x,
                this_c,
            )

            total_q = 0
            log_p_z_s = prior.logpdf(z_s.cpu().numpy())

            if verbose:
                batch_iter = tqdm(
                    iter(dataloader),
                    total=len(dataloader),
                    leave=False,
                    desc=f"sample {i}",
                )
            else:
                batch_iter = dataloader

            for batch in batch_iter:
                this_x, this_c = to_device(
                    batch[x_label].float(),
                    batch[c_label].float(),
                    model.device,
                )
                _, mu, logvar, _, _, _, _, _ = model.forward(this_x, this_c)

                total_q += q_batch(z_s.double(), mu.double(), logvar.double())

            total_marginal_kl += torch.log(total_q / dataset_size) - log_p_z_s

        return total_marginal_kl / n_samples
