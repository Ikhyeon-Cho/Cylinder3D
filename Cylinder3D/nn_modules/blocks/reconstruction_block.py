import spconv
import torch.nn as nn
from Cylinder3D.nn_modules.sparse.conv_modules import Spconv3x1x1SigmoidBatchNorm
from Cylinder3D.nn_modules.sparse.conv_modules import Spconv1x3x1SigmoidBatchNorm
from Cylinder3D.nn_modules.sparse.conv_modules import Spconv1x1x3SigmoidBatchNorm
from Cylinder3D.nn_modules.blocks.base_block import BaseBlock


class ReconBlock(BaseBlock):
    """
    Reconstruction block with directional convolutions

    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        indice_key: Key for sparse convolution indices
    """

    def __init__(self, in_channels, out_channels, indice_key=None):
        super().__init__()

        # r, θ, z directional convolutions
        self.recon_r = Spconv3x1x1SigmoidBatchNorm(
            in_channels, out_channels,
            indice_key=indice_key + "bef"
        )

        self.recon_theta = Spconv1x3x1SigmoidBatchNorm(
            in_channels, out_channels,
            indice_key=indice_key + "bef"
        )
        self.recon_z = Spconv1x1x3SigmoidBatchNorm(
            in_channels, out_channels,
            indice_key=indice_key + "bef"
        )

    def forward(self, x):
        # Process in r, θ, z directions
        shortcut_r = self.recon_r(x)
        shortcut_theta = self.recon_theta(x)
        shortcut_z = self.recon_z(x)

        # Combine directional features
        combined_features = (shortcut_r.features +
                             shortcut_theta.features +
                             shortcut_z.features)

        # Create new tensor for attention
        shortcut_r.features = combined_features * x.features

        return shortcut_r
