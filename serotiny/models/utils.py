#!/usr/bin/env python3

from collections import OrderedDict
import torch
import torch.nn as nn
from torch.nn import functional as F
import pytorch_lightning as pl
import matplotlib.pyplot as plt
import numpy as np

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
        classes[class_index] + name,
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
    index = index.long().unsqueeze(1)

    onehot = torch.zeros(len(index), n_classes).type_as(index).float()
    onehot.scatter_(1, index, 1)

    return onehot
