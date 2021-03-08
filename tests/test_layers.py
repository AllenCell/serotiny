from ..networks.layers._2d.basic import BasicLayer
from ..networks.layers._2d.pad import PadLayer
from ..networks.layers._2d.down_residual import DownResidualLayer
from ..networks.layers._2d.up_residual import UpResidualLayer
from ..networks._2d.cbvae_encoder import CBVAEEncoder
from ..networks._2d.cbvae_decoder import CBVAEDecoder

from ..networks.layers._3d.basic import BasicLayer
from ..networks.layers._3d.pad import PadLayer
from ..networks.layers._3d.down_residual import DownResidualLayer
from ..networks.layers._3d.up_residual import UpResidualLayer
from ..networks._3d.cbvae_encoder import CBVAEEncoder
from ..networks._3d.cbvae_decoder import CBVAEDecoder


def test_layers():
    BasicLayer(5, 5)
