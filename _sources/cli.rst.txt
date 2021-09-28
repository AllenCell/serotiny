serotiny CLI
============

:code:`serotiny` comes with a CLI to make it's use straightforward. The CLI has
multiple modules, for different functionalities. You can append `--help`
to any module in the command-line, to obtain more information.

* :code:`serotiny`

  * :code:`model` : model training and inference

    * :code:`train` : model training

    * :code:`predict` : inference / prediction

  * :code:`dataframe` : dataframe wrangling utils

    * :code:`transform` : dataframe transform operations

    * :code:`merge`: merge multiple dataframes

    * :code:`partition` : break one dataframe into chunks

  * :code:`image`

    * :code:`transform` : define and apply image transform pipelines

    * (Coming soon) :code:`feature_extraction`

  * :code:`utils`


Passing config dicts in CLI
***************************

Some of the CLI scripts take in dictionaries as input parameters. However,
for complex enough scenarios, it can be quite cumbersome to enter a dictionary
in the command line. For that reason, wherever a config dict is accepted, it
is also possible to pass it a path to a YAML file which contains the desired
config:
::

   $ serotiny model train --model=model.yaml ...

If you don't want to have a file per config dict, and prefer having all your
configurations in a single file, you can also specify a field in your YAML
file to be passed. Use "dot-notation" to specify nested fields:
::

   $ serotiny model train --model=config.yaml:train.model ...

Finally, you might be in one of the two situations above, but still want to
override or add an argument to your configuration. You can do that by adding
"dot-notation" overrides at the end of your command:
::

   $ serotiny model train --model=config.yaml:train.model ... model.latent_dimensions=10 model.input_channels=4
