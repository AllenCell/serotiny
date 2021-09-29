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
       --input_manifests=my_manifest.csv \ # note the plural. passing a list of files is also supported
       --output_path=output_manifest.csv \
       --output_channel_names='[channel1, channel2, channel3]' \
       --transforms_to_apply=transforms.yaml \ # we will dive into this YAML file ahead
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
config is a list, containing the several steps needed to obtain the output image. Logically,
the order matters, and the last step describes the final result.

Each step can either depend on a path column given in the input dataframe (note: your input
dataframe can input several path columns which can all be used here), or it can depend on the
result of an intermediate step. To enable this behaviour, each step must be given a name.
Additionally, each step also contains a ``transforms`` field, which is the list of transforms
to be applied in that step, in order.

A step that depends on a path column must specify:
* ``path_col`` - the column that contains the path from which to read the input image
* ``channels`` - the list of channels to use

A step that depends on outputs of previous steps must specify:
* ``input_imgs`` - the list of names of previous steps to use

An additional key ``unpack`` can be used, in cases where the step depends on
multiple input images and the underlying function expects multiple input arguments
(e.g. ``numpy.concatenate``). This corresponds to applying the \* operator to
the image list.

In general, transforms configured equivalently for every input image. However if
a given transform requires per-image parameters, a special argument ``^individual_args``
can be given, which is a dict containing as keys the function's required keyword
arguments, and as values the corresponding columns in the dataframe which contain the
needed data.

Example transform pipeline (with comments)
******************************************


.. code-block:: yaml

    - name: segmentation
      path_col: crop_seg # dataframe has a column "crop_seg"
      channels: ["crop_seg"] # the images have a "crop_seg" channel
      transforms:
        # a CropCenter transform is instantiated, with cropz, cropx, cropy set to 50,
        # and with center_of_mass being retrieved from the dataframe column with the
        # same name.
        - ^init: serotiny.io.transforms.CropCenter
          cropz: 50
          cropx: 50
          cropy: 50
          ^individual_args:
            center_of_mass: center_of_mass
        # followed by a binary_dilation operation. note how external functions
        # are entirely usable here.
        - ^bind: scipy.ndimage.morphology.binary_dilation
          structure:
            ^invoke: numpy.ones
            shape: [1,8,8,8]

    - name: raw
      path_col: crop_raw
      channels: ["crop_raw"]
      transforms:
        # same CropCenter transform as above
        - ^init: serotiny.io.transforms.CropCenter
          cropz: 50
          cropx: 50
          cropy: 50
          ^individual_args:
            center_of_mass: center_of_mass
        # followed by a quantile normalization transform
        - ^init: serotiny.io.transforms.MinMaxNormalize
          clip_min: 0.05
          clip_max: 0.95
          clip_quantile: True

    - name: masked
      input_imgs: [raw, segmentation] # this step depends on the outputs of the previous two steps
      unpack: True # use unpack because the underlying function expects multiple inputs
      transforms:
        # simply multiplication of the inputs
        - ^bind: numpy.multiply

    # this is the final step, so its result will be the output
    - name: concated
      input_imgs: [masked, segmentation] # another step which depends on previous ones
      transforms:
        - ^bind: numpy.concatenate
          axis: 0
