import torch
from torch import nn
import pytorch_lightning as pl
from collections import OrderedDict
from torch.nn import functional as F
import matplotlib.pyplot as plt
import numpy as np
import torchvision.models as models
import seaborn as sns


class BasicNeuralNetwork(nn.Module):
    def __init__(self, num_channels=3, num_classes=5, dimensions=(176, 104)):
        super().__init__()
        self.network_name = "BasicNeuralNetwork"
        self.num_classes = num_classes
        self.layer_1 = torch.nn.Linear(
            dimensions[1] * dimensions[0] * num_channels, 
            128
        )
        self.layer_2 = torch.nn.Linear(128, 256)
        # TODO configure output dims as param
        self.layer_3 = torch.nn.Linear(256, num_classes)

        self.dropout = nn.Dropout(0.25)
        torch.nn.init.xavier_uniform_(self.layer_1.weight)
        torch.nn.init.xavier_uniform_(self.layer_2.weight)
        torch.nn.init.xavier_uniform_(self.layer_3.weight)

    def forward(self, x):
        batch_size, channels, width, height = x.size()
        x = x.view(batch_size, -1)
        x = self.dropout(x)
        x = self.layer_1(x)
        x = self.dropout(x)
        x = F.relu(x)
        x = self.layer_2(x)
        x = self.dropout(x)
        x = F.relu(x)
        x = self.layer_3(x)

        return x


class ResNet18Network(nn.Module):
    def __init__(self, num_channels=3, num_classes=5, dimensions=(176, 104)):
        super().__init__()
        self.network_name = "Resnet18"
        self.num_classes = num_classes
        self.feature_extractor_first_layer = nn.Sequential(
            OrderedDict(
                [
                    (
                        "conv_before_resnet",
                        nn.Conv2d(
                            num_channels,
                            3,
                            kernel_size=1,
                            stride=1,
                            padding=0,
                            bias=False,
                        ),
                    )
                ]
            )
        )
        # initialize weights of first layer
        torch.nn.init.xavier_uniform_(
            self.feature_extractor_first_layer.conv_before_resnet.weight
        )

        self.feature_extractor = models.resnet18(pretrained=True)
        # dont freeze the weights
        # self.feature_extractor.eval()

        # Classifier architecture to put on top of resnet18
        self.classifier = nn.Sequential(
            OrderedDict(
                [
                    (
                        "fc1",
                        nn.Linear(self.feature_extractor.fc.out_features, 100)
                    ),
                    ("relu", nn.ReLU()),
                    # TODO configure output number classes
                    ("fc2", nn.Linear(100, num_classes)),
                    # ("output", nn.Sigmoid()),
                ]
            )
        )

    def forward(self, x):
        # TODO rewritw this as self.model(x) so it works witn different models
        x = self.feature_extractor_first_layer(x)
        representations = self.feature_extractor(x)
        x = self.classifier(representations)
        return x


available_networks = {"basic": BasicNeuralNetwork, "resnet18": ResNet18Network}
available_optimizers = {"adam": torch.optim.Adam, "sgd": torch.optim.SGD}
available_schedulers = {
    "reduce_lr_plateau": torch.optim.lr_scheduler.ReduceLROnPlateau
}


class ClassificationModel(pl.LightningModule):
    def __init__(
        self,
        network,
        x_label="projection_image",
        y_label="mitotic_class",
        num_channels=3,
        num_classes=5,
        image_x=104,
        image_y=176,
        lr=1e-3,
        optimizer="adam",
        scheduler="reduce_lr_plateau",
        metrics=None,
    ):
        super().__init__()
        """Save hyperparameters"""
        # Can be accessed via checkpoint['hyper_parameters']
        self.save_hyperparameters()

        """Configs"""
        self.log_grads = True
        self.activations = False

        """model"""
        self.network = network

        if self.network.network_name == "Resnet18":
            self.activations = True

        hparams = {
            symbol: getattr(self.hparams, symbol)
            for symbol in dir(self.hparams)
        }
        print(f"hyperparameters - {hparams}")

        # Print out network
        self.example_input_array = torch.zeros(
            64,
            int(self.hparams.num_channels),
            image_y,
            image_x
            # 64, int(self.hparams.num_channels), 176, 104
        )

        if self.hparams.num_classes == 5:
            self.classes = ("M0", "M1/M2", "M3", "M4/M5", "M6/M7")
        elif self.hparams.num_classes == 2:
            self.classes = ("Mitotic", "Non_Mitotic")
        else:
            self.classes = ("0", "1", "2", "3", "4")

        """Metrics"""
        self.train_metrics = torch.nn.ModuleDict(
            {
                'accuracy': pl.metrics.Accuracy(),
                'precision': pl.metrics.Precision(
                    num_classes=5,
                    average='macro'
                ),
                'recall': pl.metrics.Recall(num_classes=5, average='macro'),
            }
        )
        self.val_metrics = torch.nn.ModuleDict(
            {
                'accuracy': pl.metrics.Accuracy(),
                'precision': pl.metrics.Precision(
                    num_classes=5,
                    average='macro'
                ), 
                'recall': pl.metrics.Recall(num_classes=5, average='macro'),
            }
        )
        self.test_metrics = torch.nn.ModuleDict(    
            {
                'accuracy': pl.metrics.Accuracy(),
                'precision': pl.metrics.Precision(
                    num_classes=5,
                    average='macro'
                ),
                'recall': pl.metrics.Recall(num_classes=5, average='macro'),
            }
        )

    def _generate_logs(self, preds, true, prefix, metrics):
        _, preds_batch = torch.max(preds, 1)
        return {
            # f"{prefix}_loss": loss,
            **{
                f"{prefix}_{key}": metrics[key](
                    preds_batch, true
                )
                for key in metrics.keys()
            },
        }

    def parse_batch(self, batch):
        x = batch[self.hparams.x_label]
        y = batch[self.hparams.y_label]
        # Return floats because this is expected dtype for nn.Loss
        return x.float(), y.long()

    def forward(self, x):
        output = self.network(x)
        return output

    def training_step_end(self, outputs):
        logs = self._generate_logs(
            outputs['preds'],
            outputs['target'],
            "train",
            self.train_metrics,
        )

        for key, value in logs.items():
            # TODO configure outout logging
            self.log(key, value, on_step=True, on_epoch=True, prog_bar=True)

        if outputs['batch_idx'] == 0:
            _, preds_batch = torch.max(outputs['preds'], 1)
            conf_mat = pl.metrics.functional.confusion_matrix(
                preds_batch,
                outputs['target'],
                num_classes=self.hparams.num_classes
            )

            fig, ax = plt.subplots(1, 1, figsize=(10, 10))
            sns.heatmap(
                conf_mat.cpu(),
                cmap="RdBu_r",
                annot=True, ax=ax,
                annot_kws={"size": 8}
            )
            self.logger.experiment.add_figure(
                "Confusion matrix (train)",
                fig,
                global_step=self.current_epoch,
            )

        return outputs

    def training_step(self, batch, batch_idx):
        x, y = self.parse_batch(batch)
        if self.current_epoch == 0:
            self.reference_image = x
            self.reference_label = y
        yhat = self(x)
        # TODO configure loss
        # This depends on if y is one hot or not, also on the last
        # layer of network
        loss = nn.CrossEntropyLoss()(yhat, y)

        self.log(
            "train_loss", loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True
        )

        return {
            'loss': loss,
            'preds': yhat,
            'target': y,
            'batch_idx': batch_idx
        }

    def validation_step_end(self, outputs):
        logs = self._generate_logs(
            outputs['preds'],
            outputs['target'],
            "val",
            self.val_metrics,
        )

        for key, value in logs.items():
            # TODO configure outout logging
            self.log(key, value, on_step=True, on_epoch=True, prog_bar=True)

        if outputs['batch_idx'] == 0:
            _, preds_batch = torch.max(outputs['preds'], 1)
            conf_mat = pl.metrics.functional.confusion_matrix(
                preds_batch, 
                outputs['target'], 
                num_classes=self.hparams.num_classes
            )

            fig, ax = plt.subplots(1, 1, figsize=(10, 10))
            sns.heatmap(
                conf_mat.cpu(), 
                cmap="RdBu_r", 
                annot=True, ax=ax, 
                annot_kws={"size": 8}
            )
            self.logger.experiment.add_figure(
                "Confusion matrix (val)",
                fig,
                global_step=self.current_epoch,
            )

        return outputs

    def validation_step(self, batch, batch_idx):

        x, y = self.parse_batch(batch)
        if batch_idx == 0:
            self.logger.experiment.add_figure(
                "predictions vs actuals (val).",
                self._plot_classes_preds(self.network, x, y),
                global_step=self.current_epoch,
            )
        yhat = self(x)
        loss = nn.CrossEntropyLoss()(yhat, y)
        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True)

        return {
            'val_loss': loss,
            'preds': yhat,
            'target': y,
            'batch_idx': batch_idx
        }

    def test_step_end(self, outputs):
        logs = self._generate_logs(
            outputs['preds'],
            outputs['target'],
            "test",
            self.test_metrics,
        )

        for key, value in logs.items():
            # TODO configure outout logging
            self.log(key, value, on_step=True, on_epoch=True, prog_bar=True)

        if outputs['batch_idx'] == 0:
            _, preds_batch = torch.max(outputs['preds'], 1)
            conf_mat = pl.metrics.functional.confusion_matrix(
                preds_batch, 
                outputs['target'], 
                num_classes=self.hparams.num_classes
            )

            fig, ax = plt.subplots(1, 1, figsize=(10, 10))
            sns.heatmap(
                conf_mat.cpu(), 
                cmap="RdBu_r", 
                annot=True, 
                ax=ax, 
                annot_kws={"size": 8}
            )
            self.logger.experiment.add_figure(
                "Confusion matrix (test)",
                fig,
                global_step=self.current_epoch,
            )

        return outputs

    def test_step(self, batch, batch_idx):
        x, y = self.parse_batch(batch)

        yhat = self(x)
        loss = nn.CrossEntropyLoss()(yhat, y)

        self.log("test_loss", loss)

        return {
            "test_loss": loss, 
            'preds': yhat, 
            'target': y, 
            'batch_idx': batch_idx
        }

    def validation_epoch_end(self, outputs):
        class_probs = []
        class_preds = []
        for pred in outputs:
            pred = pred["preds"]
            class_probs_batch = [F.softmax(el, dim=0) for el in pred]
            _, class_preds_batch = torch.max(pred, 1)
            class_probs.append(class_probs_batch)
            class_preds.append(class_preds_batch)
        test_probs = torch.cat([torch.stack(batch) for batch in class_probs])
        test_preds = torch.cat(class_preds)
        # plot all the pr curves
        for i in range(len(self.classes)):
            self._add_pr_curve_tensorboard(
                i, test_probs, test_preds, global_step=self.current_epoch
            )

    def test_epoch_end(self, outputs):
        class_probs = []
        class_preds = []
        for pred in outputs:
            pred = pred["preds"]
            class_probs_batch = [F.softmax(el, dim=0) for el in pred]
            _, class_preds_batch = torch.max(pred, 1)
            class_probs.append(class_probs_batch)
            class_preds.append(class_preds_batch)
        test_probs = torch.cat([torch.stack(batch) for batch in class_probs])
        test_preds = torch.cat(class_preds)
        # plot all the pr curves
        for i in range(len(self.classes)):
            self._add_pr_curve_tensorboard(
                i,
                test_probs,
                test_preds,
                global_step=self.current_epoch,
                name="_test",
            )

    def training_epoch_end(self, outputs):
        # Log computation graph
        #  the function is called after every epoch is completed
        if self.current_epoch == 0:
            self.logger.experiment.add_graph(
                ClassificationModel(
                    network=self.hparams.network,
                    x_label=self.hparams.x_label,
                    y_label=self.hparams.y_label,
                    num_channels=self.hparams.num_channels,
                    num_classes=self.hparams.num_classes,
                    image_x=self.hparams.image_x,
                    image_y=self.hparams.image_y,
                    lr=self.hparams.lr,
                    optimizer=self.hparams.optimizer,
                    scheduler=self.hparams.scheduler,
                    metrics=self.hparams.metrics,
                ),
                self.example_input_array.cuda(),
            )
        if self.activations is True:
            self.show_activations(self.reference_image)

    def _matplotlib_imshow(self, img, one_channel=False):
        if one_channel:
            img = img.mean(dim=0)
        img = img / 2 + 0.5  # unnormalize
        npimg = img.cpu().numpy()
        if one_channel:
            plt.imshow(npimg, cmap="Greys")
        else:
            plt.imshow(np.transpose(npimg, (1, 2, 0)))

    def _images_to_probs(self, net, images):
        """
        Generates predictions and corresponding probabilities from a trained
        network and a list of images
        """
        output = net(images)
        # convert output probabilities to predicted class
        _, preds_tensor = torch.max(output, 1)
        preds = np.squeeze(preds_tensor.cpu().numpy())
        return preds, [
            F.softmax(el, dim=0)[i].item() for i, el in zip(preds, output)
            ]

    def _plot_classes_preds(self, net, images, labels):
        """
        Generates matplotlib Figure using a trained network, along with images
        and labels from a batch, that shows the network's top prediction along
        with its probability, alongside the actual label, coloring this
        information based on whether the prediction was correct or not.
        Uses the "images_to_probs" function.
        """
        preds, probs = self._images_to_probs(net, images)

        labels = list(labels.cpu().numpy())
        labels = [int(i) for i in labels]
        # plot the images in the batch, along with predicted and true labels
        fig = plt.figure(figsize=(20, 15))
        for idx in np.arange(4):
            ax = fig.add_subplot(1, 4, idx + 1, xticks=[], yticks=[])
            self._matplotlib_imshow(images[idx], one_channel=True)
            ax.set_title(
                "{0}, {1:.1f}%\n(label: {2})".format(
                    self.classes[preds[idx]],
                    probs[idx] * 100.0,
                    self.classes[labels[idx]],
                ),
            )
        return fig

    # helper function
    def _add_pr_curve_tensorboard(
        self,
        class_index,
        test_probs,
        test_preds,
        global_step=0,
        name="_val",
    ):
        """
        Takes in a "class_index" and plots the corresponding
        precision-recall curve
        """
        tensorboard_preds = test_preds == class_index
        tensorboard_probs = test_probs[:, class_index]

        self.logger.experiment.add_pr_curve(
            self.classes[class_index] + name,
            tensorboard_preds,
            tensorboard_probs,
            global_step=global_step,
        )
        self.logger.experiment.close()

    def show_activations(self, x):
        # TODO configure if we want this or noot
        # logging reference image
        for i in range(4):
            self.logger.experiment.add_image(
                "input",
                torch.Tensor.cpu(x[i][0]),
                global_step=self.current_epoch,
                dataformats="HW",
            )

        first_conv_output = self.network.feature_extractor_first_layer(x)

        for i in range(4):  # Log 4 images per epoch
            self.logger.experiment.add_image(
                "first_conv",
                torch.Tensor.cpu(first_conv_output[i][0]),
                global_step=self.current_epoch,
                dataformats="HW",
            )

        # logging first convolution activations
        second_convolution = nn.Sequential(
            OrderedDict(
                [
                    ("conv1", self.network.feature_extractor.conv1),
                    ("bn1", self.network.feature_extractor.bn1),
                    ("relu", self.network.feature_extractor.relu),
                ]
            )
        )
        second_convolution_output = second_convolution(first_conv_output)
        # img_grid = torchvision.utils.make_grid(out)
        for i in range(4):  # Log 4 images per epoch
            self.logger.experiment.add_image(
                "second_conv",
                torch.Tensor.cpu(second_convolution_output[i][0]),
                global_step=self.current_epoch,
                dataformats="HW",
            )

        # logging classifier activations
        # logging first convolution activations
        basic_block = nn.Sequential(
            OrderedDict(
                [
                    ("maxpool", self.network.feature_extractor.maxpool),
                    ("layer1", self.network.feature_extractor.layer1),
                ]
            )
        )
        basic_block_output = basic_block(second_convolution_output)
        # img_grid_classifier = torchvision.utils.make_grid(out_classifier)
        for i in range(4):
            self.logger.experiment.add_image(
                "first_basic_block",
                torch.Tensor.cpu(basic_block_output[i][0]),
                global_step=self.current_epoch,
                dataformats="HW",
            )

    def configure_optimizers(self):
        # TODO configure optimizer as Adam or SGD etc, also other parameters
        # with it like shceduler
        if self.hparams.optimizer in available_optimizers:
            optimizer_class = available_optimizers[self.hparams.optimizer]
            optimizer = optimizer_class(self.parameters(), lr=self.hparams.lr)
        else:
            raise Exception(
                (
                    f"optimizer {self.hparams.optimizer} not available, "
                    f"options are {list(available_optimizers.keys())}"
                )
            )

        if self.hparams.scheduler in available_schedulers:
            scheduler_class = available_schedulers[self.hparams.scheduler]
            scheduler = scheduler_class(optimizer)
        else:
            raise Exception(
                (
                    f"scheduler {self.hparams.scheduler} not available, "
                    f"options are {list(available_schedulers.keys())}"
                )
            )

        return (
            {
                "optimizer": optimizer,
                "scheduler": scheduler,
                "monitor": "val_accuracy",
                "interval": "epoch",
                "frequency": 1,
                "strict": True,
            },
        )

    def on_after_backward(self):
        # example to inspect gradient information in tensorboard
        if (
            self.log_grads and self.trainer.global_step % 100 == 0
        ):  # don't make the tf file huge
            params = self.state_dict()
            for k, v in params.items():
                grads = v
                name = k
                self.logger.experiment.add_histogram(
                    tag=name, values=grads, global_step=self.trainer.global_step
                )


class MyPrintingCallback(pl.Callback):
    # TODO confugure a better callback than this, its currently empty
    def on_init_start(self, trainer):
        print("Starting to init trainer!")

    def on_init_end(self, trainer):
        print("Trainer is init now")

    def on_train_end(self, trainer, pl_module):
        print("Do something when training ends")
