# -*- coding: utf-8 -*-

"""Top-level package for mitotic-classifier."""

__author__ = "Ryan Spangler, Ritvik Vasan"
__email__ = "ryan.spangler@alleninstitute.org, ritvik.vasan@alleninstitute.org"
__version__ = "0.0.1"

import os

os.environ["DASK_DISTRIBUTED__WORKER__DAEMON"] = "False"
os.environ["MKL_THREADING_LAYER"] = "GNU"


def get_module_version():
    return __version__
