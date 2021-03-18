from serotiny.networks.layers._2d.basic import BasicLayer
from serotiny.networks.layers.pad import PadLayer
from serotiny.networks.layers._2d.down_residual import DownResidualLayer
from serotiny.networks.layers._2d.up_residual import UpResidualLayer
from serotiny.networks.vae.cbvae_encoder import CBVAEEncoder
from serotiny.networks.vae.cbvae_decoder import CBVAEDecoder

from serotiny.networks.layers._3d.basic import BasicLayer
from serotiny.networks.layers._3d.down_residual import DownResidualLayer
from serotiny.networks.layers._3d.up_residual import UpResidualLayer


def test_layers():
    BasicLayer(5, 5)
