import glob
import tarfile
from pathlib import Path

import fire

import pandas as pd
from sklearn.preprocessing import LabelEncoder

from serotiny.io.quilt import download_quilt_data


def cellpath2dict(path):
    """
    Convert a given cell path to a dict
    """
    cell = path.split("/")[-1]
    cell = cell.split(".")[0]
    cell = cell.split("_")
    return {cell[i * 2]: cell[i * 2 + 1] for i in range(len(cell) // 2)}


def _make_aics_mnist_manifest(dataset_path):
    cells = []
    for split in ["train", "test"]:
        _split_path = str((Path(dataset_path) / split) / "*")
        for structure_path in glob.glob(_split_path):
            _struct_path = str(Path(structure_path) / "*")
            structure = structure_path.split("/")[-1]
            for cell_img in glob.glob(_struct_path):
                cells.append(
                    dict(
                        cellpath2dict(cell_img),
                        structure=structure,
                        split=split,
                        path=str(Path(cell_img).resolve()),
                    )
                )

    return pd.DataFrame(cells)


def make_aics_mnist_dataset(data_dir):
    """
    Download AICS MNIST dataset, create the corresponding manifest, and store
    it in a given path
    """
    data_dir = Path(data_dir)

    if not (data_dir / "aics_mnist_rgb.tar.gz").exists():
        download_quilt_data(
            package="aics/aics_mnist",
            bucket="s3://allencell",
            data_save_loc=data_dir,
        )

    if not (data_dir / "aics_mnist_rgb").exists():
        with tarfile.open(data_dir / "aics_mnist_rgb.tar.gz") as f:
            f.extractall(data_dir)

    if not (data_dir / "aics_mnist_rgb.csv").exists():
        manifest = _make_aics_mnist_manifest(data_dir / "aics_mnist_rgb")
        manifest["structure_encoded"] = LabelEncoder().fit_transform(
            manifest["structure"]
        )
        manifest.to_csv(data_dir / "aics_mnist_rgb.csv", index=False)


if __name__ == "__main__":
    fire.Fire(make_aics_mnist_dataset)
