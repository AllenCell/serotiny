# -*- coding: utf-8 -*-

"""
An agglomeration of ML primitives, utils, and complete models, used at AICS, to
model cell images and data.
"""

__author__ = "Ryan Spangler, Ritvik Vasan"
__email__ = "ryan.spangler@alleninstitute.org, ritvik.vasan@alleninstitute.org"
__version__ = "0.0.1"

import os

os.environ["DASK_DISTRIBUTED__WORKER__DAEMON"] = "False"
os.environ["MKL_THREADING_LAYER"] = "GNU"


def get_module_version():
    """
    simply return the version of this module
    """
    return __version__
