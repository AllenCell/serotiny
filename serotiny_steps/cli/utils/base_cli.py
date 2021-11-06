class BaseCLI:
    @classmethod
    def _decorate(cls, func):
        from .omegaconf_decorator import omegaconf_decorator
        return omegaconf_decorator(func)
