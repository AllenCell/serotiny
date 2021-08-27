import fire

import joblib
from sklearn.decomposition import PCA

from serotiny.io.dataframe import load_csv, filter_columns


def _fit_pca(
    dataset,
    output_path: str,
    n_components: int,
    filter_options: dict,
    compression: int = 3,
):
    cols = filter_columns(dataset.columns.tolist(), **filter_options)
    dataset = dataset[cols]
    pca = PCA(n_components).fit(dataset)

    joblib.dump(pca, output_path, compress=compression)


def fit_pca(
    dataset_path: str,
    output_path: str,
    n_components: int,
    filter_options: dict,
    compression: int = 3,
):
    dataset = load_csv(dataset_path)
    _fit_pca(dataset, output_path, n_components, filter_options, compression)


if __name__ == "__main__":
    fire.Fire(fit_pca)
