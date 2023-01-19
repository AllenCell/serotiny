__version__ = "0.0.9"

# to force init of omegaconf resolvers
import serotiny.config.resolvers as _cfg_resolvers
from hydra.core.utils import setup_globals

setup_globals()
