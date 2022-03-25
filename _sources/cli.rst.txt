serotiny CLI
============

``serotiny`` comes with a CLI to make its use straightforward. The CLI has
multiple modules, for different functionalities. You can append ``--help``
to any module in the command-line, to obtain more information.

* ``serotiny``

  * ``train`` : model training
  * ``test`` : model testing/evaluation
  * ``predict`` : inference / prediction
  * ``dataframe`` : dataframe wrangling utils. See more :ref:`here <df_wrangling>`
  * ``image`` : utils to get info about image files
  * ``config`` : utils to interact with config files


ML operations: train, test, predict
***********************************

These commands run in the context of :ref:`serotiny projects <projects>`. In that
section you'll learn how to create and configure your ``serotiny`` projects. Once
you've done so, you can use these commands to train/test/predict with the models
you specified in your project. For example:

::

   $ serotiny train \
   model=a_model_config \
   data=a_datamodule_config \
   trainer=default \ # you can omit this line if you're using the default config
   mlflow=default \  # you can omit this line if you're using the default config
   ++model.latent_dim=11 \ # if you want to override a particular value, you can do so
   ++trainer.max_epochs=50 \
   ++seed=13

Note the arguments starting with ``++argument=...``. This is the syntax to use
when you want to override a config argument coming from your config files. You
can also use it to add additional arguments which aren't specified in your config
files (e.g. ``seed`` is by default not given in the config files). The reason
behind this initially intimidating syntax is because these commands are
``hydra`` scripts under the hood. That means you can use
`hydra's overrides syntax <https://hydra.cc/docs/next/advanced/override_grammar/basic/>`_
to change config parameters, as well as leverage any additional hydra functionality
like:

- Multi-runs and launchers

  - Just add the ``-m`` flag and the intended sweeps in the command line

- Tab completion (only on Bash)

  - Run ``eval "$(serotiny train -sc install=serotiny_bash)"``. You now have tab
    completion in serotiny projects in that shell session! (Useful when you have
    many config options for each config group - see the
    :ref:`serotiny projects section <projects>` to understand what this means)

- ...

Debugging ML operations
***********************

For the commands in the above section, you can add an argument
``++make_notebook=/path/to/destination/notebook.ipynb`` to the call, with a path
to a destination notebook file which will be created and populated with cells
that mimic the behavior of the respective CLI call, so that you can tinker with
them and run them one by one.
