import spconv
import torch.nn as nn
from Cylinder3D.nn_modules.sparse.conv_modules import Spconv1x3LeakyReLUBatchNorm
from Cylinder3D.nn_modules.sparse.conv_modules import Spconv3x1LeakyReLUBatchNorm
from Cylinder3D.nn_modules.blocks.base_block import BaseBlock


class AsymmResBlock(BaseBlock):
    """
    The very first block of U-Net with complementary convolution patterns

    Architecture:
        - Two branches learning complementary residual features
        - skipA: θz-plane → rz-plane features
        - skipB: rz-plane → θz-plane features
        - Residual learning through branch addition

    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        indice_key: Key for sparse convolution indices
    """

    def __init__(self, in_channels, out_channels, indice_key):
        super().__init__()

        # Branch 1: 1x3 -> 3x1 convolutions (θz-plane → rz-plane)
        self.skip_connection = nn.Sequential(
            Spconv1x3LeakyReLUBatchNorm(
                in_channels, out_channels,
                indice_key=indice_key + "bef"),
            Spconv3x1LeakyReLUBatchNorm(
                out_channels, out_channels,
                indice_key=indice_key + "bef"),
        )

        # Branch 2: 3x1 -> 1x3 convolutions
        self.residual = nn.Sequential(
            Spconv3x1LeakyReLUBatchNorm(
                in_channels, out_channels,
                indice_key=indice_key + "bef"),
            Spconv1x3LeakyReLUBatchNorm(
                out_channels, out_channels,
                indice_key=indice_key + "bef"),
        )

    def forward(self, x):
        """
        Forward pass through asymmetric residual block

        x: SparseConvTensor
            - features: [N, in_channels]  # N: number of active voxels
            - indices:  [N, 4]            # (batch, r, θ, z)
        """
        # Residual connection
        x1 = self.skip_connection(x)
        x2 = self.residual(x)
        x2.features += x1.features

        return x1


class DownsamplingAsymmResBlock(BaseBlock):  # with pooling
    """
    Asymmetric Residual Block with optional downsampling for U-Net

    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        dropout_rate: Dropout probability
        pooling: Whether to use pooling
        height_pooling: Whether to pool in height dimension
        indice_key: Key for sparse convolution indices

    Pooling Modes:
        1. Regular (height_pooling=False):
            stride=(2,2,1) -> reduces r,θ dimensions
            [D,H,W] -> [D/2,H/2,W]

        2. Height pooling (height_pooling=True):
            stride=(2,2,2) -> reduces all dimensions
            [D,H,W] -> [D/2,H/2,W/2]
    """

    def __init__(self, in_channels, out_channels, dropout_rate,
                 do_pooling=True, do_height_pooling=False, indice_key=None):
        super().__init__()

        self.skip_connection = nn.Sequential(
            Spconv3x1LeakyReLUBatchNorm(
                in_channels, out_channels,
                indice_key=indice_key + "bef"),
            Spconv1x3LeakyReLUBatchNorm(
                out_channels, out_channels,
                indice_key=indice_key + "bef"),
        )

        self.residual = nn.Sequential(
            Spconv1x3LeakyReLUBatchNorm(
                in_channels, out_channels,
                indice_key=indice_key + "bef"),
            Spconv3x1LeakyReLUBatchNorm(
                out_channels, out_channels,
                indice_key=indice_key + "bef"),
        )

        # Pooling layer
        if do_pooling:
            if do_height_pooling:
                # reduce all dimensions [D,H,W] -> [D/2,H/2,W/2]
                self.pooling = spconv.SparseConv3d(
                    out_channels, out_channels,
                    kernel_size=3,
                    stride=2,  # (2,2,2)
                    padding=1,
                    indice_key=indice_key,
                    bias=False
                )
            else:
                # reduce r,θ dimensions [D,H,W] -> [D/2,H/2,W]
                self.pooling = spconv.SparseConv3d(
                    out_channels, out_channels,
                    kernel_size=3,
                    stride=(2, 2, 1),
                    padding=1,
                    indice_key=indice_key,
                    bias=False
                )
        self.has_pooling_layer = do_pooling

    def forward(self, x):
        # Residual connection
        x1 = self.skip_connection(x)
        x2 = self.residual(x)
        x2.features += x1.features

        # Optional pooling
        if self.has_pooling_layer:
            return self.pooling(x2), x2
        else:
            return x2
