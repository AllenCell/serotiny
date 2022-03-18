.. _projects:

serotiny projects
=================

One of the purposes of ``serotiny`` is to standardize the structure of ML projects
at our organization. For that, we created `a cookiecutter template <https://github.com/AllenCellModeling/serotiny-project-cookiecutter>`_
for ``serotiny`` projects, which makes it easier to get started with the right structure in place.

``serotiny`` ML-related commands recognize when they're run from within a serotiny
project and are thus able to make use and interact with each projects configuration,
as you'll see ahead.

We recommend familiarizing with `hydra <https://hydra.cc>`_ to better understand
the inner workings of serotiny projects. We rely heavily on ``hydra``'s functionality
to `instantiate Python objects from YAML configurations <https://hydra.cc/docs/advanced/instantiate_objects/overview/>`_, as well as its general
framework to configure applications in a modular fashion.

Project structure
*****************

Assuming you used `our cookiecutter template
<https://github.com/AllenCellModeling/serotiny-project-cookiecutter>`_,
to create a ``serotiny`` project as:

::

   $ cookiecutter https://github.com/AllenCellModeling/serotiny-project-cookiecutter
   project_name [My Project]: A Tiny Project
   package_name [a_tiny_project]:
   project_description [a simple project**: it's just a tiny project

A folder named ``a_tiny_project** will be created, with the following structure:

::

   .
   ├── a_tiny_project
   │   └── config
   │       ├── data
   │       ├── mlflow
   │       ├── model
   │       ├── trainer
   │       │   └── callbacks
   │       ├── train.yaml
   │       ├── test.yaml
   │       └── predict.yaml
   ├── MANIFEST.in
   └── setup.py


Configuration
*************

From the above structure we should note the following:

- There are three "top-level" YAML files: ``train.yaml``, ``test.yaml``, ``predict.yaml``

  - These stand for each of the available ML operations, and each correspond to a different
    CLI call and a ``hydra`` "config name"

- There are 5 configuration groups, each of which can be modularly and independently
  configured.

  - Under each of these configuration groups, you can add YAML files
    which correspond to a specific way in which to configure that group. E.g.
    under ``model`` you might have different architectures, under ``data** you
    may have different datasets, and so on.

- This modular form of configuration will make it trivial to swap different configuration
  blocks in and out.


**The** ``model`` **config group**
##########################


This is where the available models for training will be specified.
Each configuration file in this group should contain a ``hydra`` specification of a `(LightningModule)  <https://pytorch-lightning.readthedocs.io/en/latest/api/pytorch_lightning.core.LightningModule.html>`_ model class.

An (incomplete) example of a possible model config could be:

::

    _target_: serotiny.models.TabularVAE
    x_dim: 100
    latent_dim: 16
    hidden_layers: [32, 32, 32]
    x_label: "x"
    optimizer:
      _partial_: true
      _target_: torch.optim.Adam
      lr: 1e-3


**The** ``data`` **config group**
#################################

This is where the available datasets for training will be specified.
Each configuration file in this group should contain a ``hydra`` specification of a
`LightiningDatamodule <https://pytorch-lightning.readthedocs.io/en/latest/extensions/datamodules.html?highlight=datamodule>`_ datamodule class.

As of now, we only provide a
:py:class:`ManifestDatamodule class <serotiny.datamodule.ManifestDatamodule>` as
part of ``serotiny``, which is a class that works on top of what we call manifest
files. These are ``.csv`` files which describe a dataset, and for which we have
:py:mod:`loader classes <serotiny.io.dataframe.loaders>** which consume values
from the dataframe, either directly, or e.g. by reading file paths.

Alternatively, you can write and instantiate your own custom datamodule class instead.

One (incomplete) example of a datamodule instantiation could be:

::

    _target_: serotiny.datamodules.ManifestDatamodule

    path: /path/to/a/csv/file.csv
    batch_size: 64
    num_workers: 1
    loaders:
      x:
        _target_: serotiny.io.dataframe.loaders.LoadColumns
        startswith: feature_

    split_column: "split"


**The** ``trainer`` **config group**
####################################

Each configuration file in this group should contain a ``hydra`` specification of a `Trainer  <https://pytorch-lightning.readthedocs.io/en/latest/api/pytorch_lightning.trainer.trainer.Trainer.html>`_

This is the class that governs the training/testing/prediction process. See the
Pytorch Lightning docs for more on its available parameters and functionality.

**The** ``trainer/callbacks`` **config group**
##############################################

Each configuration file in this group should contain a list of ``hydra`` specifications of
`Callbacks <https://pytorch-lightning.readthedocs.io/en/latest/api/pytorch_lightning.trainer.trainer.Trainer.html>`_
you want to use during training/testing.

Callbacks are our recommended way of adding functionality to the train/eval loop,
like computing metrics and artifacts, using early-stopping, etc. For functionality
that is project specific, we recommend you implement these as part of your
newly created serotiny project. For general functionality, you can leverage
callbacks from the Pytorch Lightning community, and/or contribute them to
either ``serotiny** or Pytorch Lightning.

An example of a list of callbacks would be:

::

   - _target_: pytorch_lightning.callbacks.EarlyStopping
     monitor: val_loss
     patience: 5
     min_delta: 0.1

   - _target_: a_tiny_project.callbacks.YourCustomCallback
     param1: "a"
     param2: "b"
     more_params: [1,2,3]


**The** ``mlflow`` **config group**
###################################

This should contain a couple parameters to configure the usage of an MLFlow
server. At the very least, you should specify ``tracking_uri`` and set it
to the URL of your MLFlow server. Additionally, when running a training/testing
run, you'll have to specify the ``experiment_name`` and ``run_name`` either
here or in the command line. The way you do this depends on how you intend to
organize your ML runs, but one propose way is to have different config files with
different experiment names and a run name which is automatically computed from
other config values, using `OmegaConf's interpolation syntax <https://omegaconf.readthedocs.io/en/latest/usage.html#variable-interpolation>`_
