#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import fire
import logging

import pandas as pd
from pathlib import Path

###############################################################################

log = logging.getLogger(__name__)

###############################################################################


def select_fields(
    dataset_path,
    output_path,
    fields,
):

    dataset = load_csv(dataset_path, [])
    manifest = dataset[fields]
    manifest.to_csv(output_path, index=False)

    print(result)
    return result


if __name__ == "__main__":
    # example command:
    # python -m serotiny.steps.apply_projection \
    #     --dataset_path "data/manifest_merged.csv" \
    #     --output_path "data/manifest_filtered.csv" \
    #     --fields "['ChosenMitoticClass']"

    fire.Fire(select_fields)
