.. _getting_started:

Getting started
=================

One of ``serotiny``'s main purposes is to standardize the structure of ML projects
at our organization. For that, we created `a cookiecutter template <https://github.com/AllenCellModeling/serotiny-project-cookiecutter>`_
for ``serotiny`` projects, which makes it easier to get started with the right structure in place.

``serotiny`` has a command-line interface, with machine-learning related
commands which recognize when they are run from within a serotiny project and
are thus able to make use of and interact with each projects configuration, as
you will see ahead.

We recommend becoming familiar with `hydra <https://hydra.cc>`_ to better understand
the inner workings of serotiny projects. We rely heavily on ``hydra``'s functionality
to `instantiate Python objects from YAML configurations <https://hydra.cc/docs/advanced/instantiate_objects/overview/>`_,
as well as its general framework to configure applications in a modular fashion.

Project structure
*****************

Using `our cookiecutter template
<https://github.com/AllenCellModeling/serotiny-project-cookiecutter>`_,
you can create a ``serotiny`` project as:

::

   $ cookiecutter https://github.com/AllenCellModeling/serotiny-project-cookiecutter
   project_name [My Project]: A Tiny Project
   package_name [a_tiny_project]:
   project_description [a simple project]: it's just a tiny project

A folder named ``a_tiny_project`` will be created, with the following structure:

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

This folder structure is a valid, installable python package. That fact will be
useful later when you want to add custom models and datamodules to your project,
and refer to them in the configuration files, as you will see ahead.

To install your python project, you can run:

::

   $ pip install -e a_tiny_project


Configuration
*************

From the above structure we should note the following:

- There are three "top-level" YAML files: ``train.yaml``, ``test.yaml``, ``predict.yaml``

  - These stand for each of the available ML operations, and each correspond to a different
    CLI call and a ``hydra`` "config name"

- There are 5 configuration groups (``data``, ``model``, ``trainer``,
  ``trainer/callbacks``, ``mlflow``), each of which can be modularly and
  independently configured.

  - Under each of these configuration groups, you can add YAML files
    which correspond to a specific way in which to configure that group. E.g.
    under ``model`` you might have different architectures, under ``data`` you
    may have different datasets, and so on.

- This modular form of configuration will make it trivial to swap different configuration
  blocks in and out.


**The** ``model`` **config group**
##################################


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

If you developed your own architecture and want to use it, you can create a config
file for it too, where the value for the ``_target_`` key would be something like
``a_tiny_project.models.MyCustomModel`` - assuming your project is called
``a_tiny_project``, and that it has a module callled ``models`` which contains a
``LightningModule`` subclass, called ``MyCustomModel``.

**The** ``data`` **config group**
#################################

This is where the available datasets for training will be specified.
Each configuration file in this group should contain a ``hydra`` specification of a
`LightiningDatamodule <https://pytorch-lightning.readthedocs.io/en/latest/extensions/datamodules.html?highlight=datamodule>`_ datamodule class.

As of now, we only provide a
:py:class:`ManifestDatamodule class <serotiny.datamodules.ManifestDatamodule>` as
part of ``serotiny``, which is a class that works on top of what we call manifest
files. These are ``.csv`` files which describe a dataset, and for which we have
:py:mod:`loader classes <serotiny.io.dataframe.loaders>` which consume values
from the dataframe, either directly, or e.g. by reading file paths.

Alternatively, you can write and instantiate your own custom datamodule class instead,
or use one developed by someone else.

.. seealso::

   Pytorch Lightning has a collection of datamodules
   (and other useful building blocks) in a package called ``lightning-bolts`` which you
   can install via ``$ pip install lightning-bolts``. See `here <https://lightning-bolts.readthedocs.io/en/latest/>`_ for more info about it.

One (incomplete) example of a datamodule instantiation could be:

::

    _target_: serotiny.datamodules.ManifestDatamodule

    path: /path/to/a/csv/file.csv # this can either be the path to a .csv file
                                  # or a folder with 3 .csv files named
                                  # "train.csv", "valid.csv", "test.csv"
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
either ``serotiny`` or Pytorch Lightning.

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

This config should contain a couple parameters to configure the usage of an MLFlow
server. The three required parameters are: ``tracking_uri``, ``experiment_name``,
``run_name``.  In MLFlow, an "experiment" is a set
of "runs" - e.g. it might be a set of training runs for the same architecture
with different hyperparameters. Unlike MLFlow, ``serotiny`` expects run names
to be unique (under a given experiment name). We use this notion to, for example,
restart failed training runs.

Like any parameter in a ``serotiny`` call, the MLFlow parameters can either be set
in the config file or in the command-line. The value ``tracking_uri`` is likely
going to remain throughout your project, whereas the ``experiment_name`` and
``run_name`` might change with each different run. For that reason, we recommend
setting the ``tracking_uri`` argument in ``config/mlflow/default.yaml`` and
setting ``experiment_name`` and ``run_name`` in the command-line, by doing:

::

   $ serotiny train ... ++mlflow.experiment_name="Your experiment name" \
   ++mlflow.run_name="your run name"


Additionally, in scenarios such as parameter sweeps, it might be useful to
programatically set the ``run_name`` as a function of some other configuration
values. In that case, you can set ``run_name`` in the config file and use
`OmegaConf's interpolation syntax <https://omegaconf.readthedocs.io/en/latest/usage.html#variable-interpolation>`_
to construct a ``run_name`` string that is unique for each of your sweep values,
like:

::

   ...

   run_name = ${model.input_size}_${data.batch_size}_${seed}_${trainer.max_epochs}

   ...

.. important::

   Remember: ``serotiny`` expects ``run_name`` to be unique for each run under
   a given experiment, so if your interpolation logic doesn't generate different
   run names, you will end up always operating over the same run, which is not
   what you want.


Finally, all of the above assumes you have a running MLFlow server.
Check `the MLFlow docs <https://mlflow.org>`_ for more info on this.

To just test it out, you can try running:

::

   $ mlflow server --backend-store-uri /a/path/to/store/mlflow/data -p 1234

This will start an MLFlow server on port 1234, and store
its data on ``/a/path/to/store/mlflow/data``.


Train a model
*************

With the above configurations in place, you should be ready to train a model.
You can override configuration parameters in the CLI call, using
`hydra's overrides syntax <https://hydra.cc/docs/next/advanced/override_grammar/basic/>`_.

::

   $ serotiny train model=your_model_config_name data=your_data_config_name \
   ++seed=1337 ++trainer.max_epochs=10 ++trainer.enable_checkpointing=True \

Note the selection of the model and data configurations. If you omit these,
``serotiny`` will use the corresponding  ``default.yaml`` configurations.

Assuming appropriate configuration, you should see the results of your model
training on MLFlow

Load a trained model
********************

On MLFlow's dashboard, identify the run id for the model you just trained.
You can now load it e.g. in a Jupyter Notebook by doing:

::

   from serotiny.ml_ops.mlflow_utils import load_model_from_checkpoint

   model = load_model_from_checkpoint(THE_TRACKING_URI, THE_RUN_ID)
