import spconv
import torch.nn as nn
from Cylinder3D.nn_modules.sparse.conv_modules import Spconv3x3LeakyReLUBatchNorm
from Cylinder3D.nn_modules.sparse.conv_modules import Spconv3x1LeakyReLUBatchNorm
from Cylinder3D.nn_modules.sparse.conv_modules import Spconv1x3LeakyReLUBatchNorm
from Cylinder3D.nn_modules.blocks.base_block import BaseBlock


class UpBlock(BaseBlock):
    """
    Upsampling block with skip connections

    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        indice_key: Key for sparse convolution indices
        up_key: Key for upsampling indices
    """

    def __init__(self, in_channels, out_channels, indice_key=None, up_key=None):
        super().__init__()

        # Initial convolution -> transform encoder features
        self.transform = Spconv3x3LeakyReLUBatchNorm(
            in_channels, out_channels,
            indice_key=indice_key + "new_up"
        )

        # Upsampling layer
        self.upsampling = spconv.SparseInverseConv3d(
            out_channels, out_channels,
            kernel_size=3,
            indice_key=up_key,
            bias=False
        )
        self.refinement = nn.Sequential(
            Spconv1x3LeakyReLUBatchNorm(
                out_channels, out_channels,
                indice_key=indice_key),
            Spconv3x1LeakyReLUBatchNorm(
                out_channels, out_channels,
                indice_key=indice_key),
            Spconv3x3LeakyReLUBatchNorm(
                out_channels, out_channels,
                indice_key=indice_key),
        )

    def forward(self, x, skip):

        x = self.transform(x)                    # Initial convolution
        x = self.upsampling(x)                   # Upsampling
        x.features = x.features + skip.features  # Skip connection
        x = self.refinement(x)                   # Refinement

        return x
