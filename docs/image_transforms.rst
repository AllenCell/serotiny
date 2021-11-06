Applying image transforms
=========================

A common step in an image-centric workflow is transforming the input images
(resizing, padding, registering, normalizing, applying dillations and
other filters...).

For this, we provide a CLI utility that enables complex and flexible image
transformations to be applied to a dataset of images. Our starting point is
a manifest file (e.g. a ``.csv`` file) which contains (at least) a column
with paths to the images in the dataset.

The general structure of the CLI call is:

::

   serotiny image transform \
       --input_manifest=my_manifest.csv \
       --outputs=example.yaml:crop_and_normalize.outputs \
       --output_path=output_manifest.csv \
       --transforms_to_apply=example.yaml:crop_and_normalize.transforms \ # we will dive into this YAML file ahead
       --index_col="cell_id" # the index col is used for the image filenames
       # --merge_col=... \ # optional. to merge multiple manifests (horizontally)
       # --include_cols=... \ # optional. to include other columns in the output manifest
       # --n_workers=10 \ # optional. process multiple images in parallel
       # --verbose=False \ # optional. produce command line output


Configuring ``transforms_to_apply``
***************************************

``transforms_to_apply`` is where the whole transform pipeline is specified.
It can rely on ``serotiny``'s :doc:`dynamic import functionality <dynamic_imports>`
to provide very flexible transform pipelines. As described in the :doc:`CLI page <cli>`,
a config can either be provided directly in the command line, or it can be given
as a YAML file, or part of one. In any of those cases, the ``transforms_to_apply``
config is a dictionary, containing the several transforms needed to obtain the output image. Logically,
the order matters, and the last step describes the final result.

Each step can either depend on a path column given in the input dataframe (note: your input
dataframe can input several path columns which can all be used here), or it can depend on the
result of one (or multiple) intermediate transform(s). To enable this behaviour, each
transform must be given a name. Additionally, each transform also contains a ``steps``
field, which is the sequence of steps that make up that that transform, in order. This
can either be a dictionary or a list, and in both cases the order matters.

A transform must specify its ``input`` field as a dictionary, where each key is
either a column in the dataframe or the name of a previous transform, and each
corresponding value is the list of channels to use from the input given by its key

A special keyword ``^unpack`` can be used, in cases where a step depends on
multiple input images and the underlying function expects multiple input arguments
(e.g. ``numpy.concatenate``). This corresponds to applying the \* operator to
the image list.

In general, steps are configured equivalently for every input image. However if
a given step requires per-image parameters, a special argument ``^individual_args``
can be given, which is a dict containing as keys the function's required keyword
arguments, and as values the corresponding columns in the dataframe which contain the
needed data.

Specifying ``outputs``
**********************

``outputs`` is the parameter that lets the tool know what outputs and channels to store
from the transforms applied. It allows multiple outputs to be stored, which is convenient
if we are interested in different outputs which share a common initial transformation.
This parameter is simply a dictionary where each key is the name of a transform in the pipeline,
(i.e. corresponds to a key in the ``transforms_to_apply`` dictionary) and the corresponding
value is a list of the channels to store for that key.

Example transform pipeline (with comments)
******************************************


.. code-block:: yaml
  crop_and_normalize:
    transforms: # in the example command above, this is the field passed to the `transforms_to_apply` parameter
      cropped: # name of this transform. we use it to refer to its output as well
        input:
          img_path: [dna_raw, membrane_raw, structure_raw, dna_seg, membrane_seg, structure_seg]
        steps: # specifying the steps for this transform as a dict. in this case, a single step named "crop"
          crop:
            # a CropCenter transform is instantiated, with cropz, cropx, cropy set to 50,
            # and with center_of_mass being retrieved from the dataframe column with the
            # same name.
            ^init: serotiny.data.image.CropCenter
            cropz: 50
            cropx: 50
            cropy: 50
            ^individual_args:
              center_of_mass: center_of_mass

      normalized_raw:
        input: # here the input is the result of a previous transform, and a subset of its channels
          cropped: [0, 1, 2]
        steps: # here we specified the `steps` field as a list
          - ^init: serotiny.data.image.MinMaxNormalize
            clip_min: 0.05
            clip_max: 0.95
            clip_quantile: True

      masked:
        input: # this transform takes in the result of two previous transforms
          normalized_raw: # if no list of channels is given, all of the channels are used
          cropped: [3, 4, 5]
        steps:
          - ^bind: numpy.multiply
            ^unpack: True # using `^unpack` because the underlying function expects multiple inputs


      concated:
        input: # again, a transform which takes in the result of two previous transforms
          masked:
          cropped: [3, 4, 5]
        steps:
          - ^bind: numpy.concatenate
            axis: 0

    # specifying the outputs. this is the field passed to the `outputs` parameter,
    # in the example command at the top of the page
    outputs:
      # we're specifying that the result of `concated` will be saved, with the given
      # channel names, in the given order
      concated: [dna_raw, membrane_raw, structure_raw, dna_seg, membrane_seg, structure_seg]

