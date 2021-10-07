Dataframe wrangling
===================

Wrangling dataframes is a common need. For us this mostly involves transforming
manifest files - partitioning, merging, creating train-val-test splits, filtering
rows and/or columns, ...

We implemented a CLI tool for this. It exposes to the CLI all the transforms defined
in the :py:mod:`dataframe transforms module <serotiny.data.dataframe.transforms>`, but
importantly it also allows the user to chain them in a single CLI call.

Example calling a single transform:

::
   serotiny dataframe transform \
       filter_columns my_input.csv --columns='["cellid","crop_seg","crop_raw"]' \
       --output_path my_output.csv


Example chaining multiple transforms:

::
   serotiny dataframe transform \
       filter_rows my_input.csv cellid '[1,2,3,4]' --exclude  - \
       filter_columns ... --columns='["cellid","crop_seg","crop_raw"]' - \
       split_dataframe ... --train_frac=0.7 --return_splits=false - \
       --output_path my_output.csv

Note the ``-`` at the end of the steps. That signifies we're done providing arguments
to the step that precedes it. Note also the ``...`` as the first argument to the
second and third steps. That is a placeholder, letting the CLI know which of the
input arguments to be provided with the result of the previous step
