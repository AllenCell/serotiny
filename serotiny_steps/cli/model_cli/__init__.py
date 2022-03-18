from ..utils.base_cli import BaseCLI

class ModelCLI(BaseCLI):
    @classmethod
    def train(cls):
        """
        Train a model given its configuration.
        """
        from serotiny_steps.train_model import train_model
        return cls._decorate(train_model)

    @classmethod
    def predict(cls):
        """
        Apply a trained model to some data
        """
        from serotiny_steps.apply_model import apply_model
        return cls._decorate(apply_model)
