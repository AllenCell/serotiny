from typing import Optional

import fire

from sklearn.model_selection import train_test_split

from serotiny.io.dataframe import load_csv


def make_split_col(
    manifest_path: str,
    train_frac: float,
    val_frac: Optional[float] = None,
):
    manifest = load_csv(manifest_path, required_fields=[])

    train_ix, val_test_ix = train_test_split(manifest.index.tolist(),
                                             train_size=train_frac)
    if val_frac is not None:
        val_frac = val_frac / (1 - train_frac)
    else:
        # by default use same size for val and test
        val_frac = 0.5

    val_ix, test_ix = train_test_split(val_test_ix, train_size=val_frac)

    manifest.loc[train_ix, "split"] = "train"
    manifest.loc[val_ix, "split"] = "test"
    manifest.loc[test_ix, "split"] = "validation"

    return manifest

def main(
    manifest_path: str,
    output_path: str,
    train_frac: float,
    val_frac: Optional[float] = None,
):
    manifest = make_split_col(manifest_path, train_frac, val_frac)
    manifest.to_csv(output_path)


if __name__ == "__main__":
    fire.Fire(main)
