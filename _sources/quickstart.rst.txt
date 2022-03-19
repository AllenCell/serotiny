Quickstart
==========

Create a ``serotiny``-project
*****************************

.. note ::

   ``serotiny`` is mostly meant to be used in the context of ``serotiny``-projects.
   For a more detailed explanation of what these are, see the :ref:`serotiny projects
   section <projects>` of the documentation.

Using `our cookiecutter template
<https://github.com/AllenCellModeling/serotiny-project-cookiecutter>`_,
create a ``serotiny`` project:

::

   $ cookiecutter https://github.com/AllenCellModeling/serotiny-project-cookiecutter

You will be prompted for a couple details about your project. Assuming you called
it "a tiny project", a folder named ``a_tiny_project`` will be created, with
the following structure:

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

This is an installable Python package. You can add modules to it, inside
``a_tiny_project``, according to your needs.

Setup the configs
*****************

Update the ``default.yaml`` files you'll find in the subdirectories of
``a_tiny_project/config``. They should contain commented examples of what
kinds of classes you can define there. These make use of hydra's functionality
to instantiate Python objects from a YAML configuration. See `here <https://hydra.cc/docs/next/advanced/instantiate_objects/overview/>`_


Get a MLFlow server running
***************************

Refer to `the MLFlow docs <https://mlflow.org>`_ for this. Once you have it
running, adjust the relevant configuration in
``a_tiny_project/config/mlflow/default.yaml``


Train a model
*************

You're ready to train a model! You can override configuration parameters in the
CLI call, using
`hydra's overrides syntax <https://hydra.cc/docs/next/advanced/override_grammar/basic/>`_.

::

   $ serotiny train

Assuming appropriate configuration, you should see the results of your model
training on MLFlow

Load a trained model
********************

Identify the run id for the model you just trained. You can now load it e.g. in
a Jupyter Notebook by doing:

::

   from serotiny.ml_ops.mlflow_utils import load_model_from_checkpoint

   model = load_model_from_checkpoint(THE_TRACKING_URI, THE_RUN_ID)
