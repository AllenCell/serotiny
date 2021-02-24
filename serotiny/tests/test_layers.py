from ..library.layers._2d.basic import BasicLayer
from ..library.layers._2d.pad import PadLayer
from ..library.layers._2d.down_residual import DownResidualLayer
from ..library.layers._2d.up_residual import UpResidualLayer
from ..library.networks._2d.cbvae_encoder import CBVAEEncoder
from ..library.networks._2d.cbvae_decoder import CBVAEDecoder

from ..library.layers._3d.basic import BasicLayer
from ..library.layers._3d.pad import PadLayer
from ..library.layers._3d.down_residual import DownResidualLayer
from ..library.layers._3d.up_residual import UpResidualLayer
from ..library.networks._3d.cbvae_encoder import CBVAEEncoder
from ..library.networks._3d.cbvae_decoder import CBVAEDecoder

def test_layers():
    BasicLayer(5, 5)
