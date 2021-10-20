from .dummy import DummyDatamodule
from .manifest_datamodule import ManifestDatamodule
from .patch import PatchDatamodule
from .split import SplitDatamodule
# from .single_graph_datamodule import SingleGraphDatamodule
# from .multiple_graph_datamodule import MultipleGraphDatamodule

__all__ = [
    "DummyDatamodule",
    "ManifestDatamodule",
    "PatchDatamodule",
    "SplitDatamodule",
    # "SingleGraphDatamodule",
    # "MultipleGraphDatamodule"
]
