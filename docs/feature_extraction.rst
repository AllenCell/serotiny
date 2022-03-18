Extraction features from images
===============================

Often we need to extract features for images, either for direct use in downstream
tasks (e.g. computing spherical harmonic coefficients), or as a intermediate step
for image processing (e.g. computing quantiles, bounding-boxes, etc.).

For this, we provide a CLI utility, similar in spirit to the image transform one,
that enables a set of features to be extracted from a dataset of images and stored
in a manifest. As usual, the starting point is
a manifest file (e.g. a ``.csv`` file) which contains (at least) a column
with paths to the images in the dataset.

The general structure of the CLI call is:

::


   serotiny image extract_features \
       --input_manifest=my_manifest.csv \
       --output_path=my_computed_features.csv \
       --features_to_extract=example.yaml:bounding_box.features_to_extract \ # explained ahead
       --path_col=the_input_image_col \
       --index_col=CellId \
       --include_cols='[]' \ # list of columns from the input manifest to include in the output
       # --image_reader=aicsimageio.readers.ome_tiff_reader.OmeTiffReader \ # optional. avoids an initial io operation to select what type of reader to use.
       # --return_merged=True \ # optional. whether to merge the results with the input
       # --backend="multiprocessing" \ # optional. select which parallel backend to use
       # --write_every_n_rows=100 \ # optional. number of iterations to save to disk at once
       # --n_workers=1 \ # optional. number of workers to parallelize the task
       # --verbose=False \ # optional. whether to produce some cli output
       # --skip_if_exists=False \ # optional. whether to try to retrieve results from an unfinished run and continue from there
       # --debug=False \ # optional. whether to return errors in csv instead of raising and breaking



Configuring ``features_to_extract``
***************************************

``features_to_extract`` is a dictionary where each key
is the name of a feature to extract, and will be used as the corresponding
column name in the output manifest. The value for each key is a config dict,
(which is loaded using ``serotiny``'s :doc:`dynamic import functionality <dynamic_imports>`),
specifying a feature extraction callable. these need to conform to a certain API
(for now, at least). You can find existing ones in the
:py:mod:`image feature extraction module <serotiny.data.image.feature_extraction>`

Example feature extraction
******************************************


.. code-block:: yaml

  bounding_box:
    features_to_extract:
      bbox:
        ^bind: serotiny.data.image.dillated_bbox
        channel: "membrane_seg"
        structuring_element: [8,8,8]
      center_of_mass:
        ^bind: serotiny.data.image.center_of_mass
        channel: "membrane_seg"
