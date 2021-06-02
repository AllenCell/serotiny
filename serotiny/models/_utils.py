from typing import Sequence, Tuple, Union

import inspect
import math
import random
import torch
import torch.nn.functional as F
import torch.optim as opt
from torch import device, Tensor
from pytorch_lightning import LightningModule
from scipy.stats import multivariate_normal

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tqdm import trange, tqdm

def to_device(
    *args
):

    assert len(args) > 1
    target_device = args[-1]
    assert isinstance(target_device, (str, device))
    args = args[:-1]

    if len(args) > 1:
        return tuple(
            arg.to(target_device) for arg in args
        )
    else:
        return args[0].to(target_device)


def get_ranked_dims(
    stats,
    cutoff_kld_per_dim,
    max_num_shapemodes,
):
    stats = (
        stats.loc[stats["test_kld_per_dim"] > cutoff_kld_per_dim]
        .sort_values(by=["test_kld_per_dim"])
        .reset_index(drop=True)
    )

    ranked_z_dim_list = [i for i in stats["dimension"][::-1]]
    mu_std_list = [i for i in stats["mu_std_per_dim"][::-1]]
    mu_mean_list = [i for i in stats["mu_mean_per_dim"][::-1]]

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


def index_to_onehot(index, n_classes):
    index = index.long()

    onehot = torch.zeros(len(index), n_classes).type_as(index).float()
    onehot.scatter_(1, index, 1)

    return onehot


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

