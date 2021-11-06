from ..utils.base_cli import BaseCLI

class MiscCLI(BaseCLI):
    @classmethod
    def fit_pca(cls):
        from serotiny_steps.fit_pca import fit_pca
        return cls._decorate(fit_pca)

    @classmethod
    def make_aics_mnist_dataset(cls):
        from serotiny_steps.make_aics_mnist_dataset import make_aics_mnist_dataset
        return cls._decorate(make_aics_mnist_dataset)
