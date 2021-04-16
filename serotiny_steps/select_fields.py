#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
import fire

from serotiny.io.data import load_csv

###############################################################################

log = logging.getLogger(__name__)

###############################################################################


def select_fields(
    dataset_path,
    output_path,
    fields,
):
    """
    Select some columns from a dataset
    """
    dataset = load_csv(dataset_path, [])
    manifest = dataset[fields]
    manifest.to_csv(output_path, index=False)


if __name__ == "__main__":
    # example command:
    # python -m serotiny.steps.select_fields \
    #     --dataset_path "data/projection.csv" \
    #     --output_path "data/filtered.csv" \
    #     --fields "['CellId', 'CellIndex', 'FOVId', 'CellImage3DPath', \
    # 'CellImage2DAllProjectionsPath', 'CellImage2DYXProjectionPath', \
    # 'SourceReadPath', 'NucleusSegmentationReadPath', \
    # 'MembraneSegmentationReadPath', 'ChannelIndexDNA', \
    # 'ChannelIndexMembrane', 'ChannelIndexStructure', \
    # 'ChannelIndexBrightfield', 'ChannelIndexNucleusSegmentation', \
    # 'ChannelIndexMembraneSegmentation', 'ChosenMitoticClass', \
    # 'Chosen2DProjectionPath']"

    fire.Fire(select_fields)
