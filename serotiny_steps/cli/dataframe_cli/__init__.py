from serotiny_steps.merge_data import merge_data
from serotiny_steps.partition_data import partition_data
from .transform import DataframeTransformCLI

class DataframeCLI:
    merge = merge_data
    partition = partition_data
    transform = DataframeTransformCLI
