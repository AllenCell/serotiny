from serotiny_steps.fit_pca import fit_pca as _fit_pca
from serotiny_steps.make_aics_mnist_dataset import make_aics_mnist_dataset as _make_mnist


class MiscCLI:
    fit_pca = _fit_pca
    make_aics_mnist_dataset = _make_mnist
