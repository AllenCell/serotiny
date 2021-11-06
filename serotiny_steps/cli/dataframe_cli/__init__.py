from ..utils.base_cli import BaseCLI

class DataframeCLI(BaseCLI):
    @classmethod
    def merge(cls):
        """
        Load a list of csv's, merge them, then write back out to csv.
        """
        from serotiny_steps.merge_data import merge_data
        return cls._decorate(merge_data)

    @classmethod
    def partition(cls):
        """
        Split a dataframe into N partitions of output data
        """
        from serotiny_steps.partition_data import partition_data
        return cls._decorate(partition_data)

    @classmethod
    def transform(cls):
        """
        Apply a transform (or chain of transforms) to a dataframe
        """
        from .transform import DataframeTransformCLI
        return DataframeTransformCLI
