"""
Author: Ikhyeon Cho
Link: https://github.com/Ikhyeon-Cho
File: asymmetric_sparse_conv.py
Date: 2024/12/16 10:00

Asymmetric 3D Sparse Convolution Network for Cylinder3D
Reference: https://github.com/xinge008/Cylinder3D
"""

import torch
import torch.nn as nn
import spconv
import numpy as np
from Cylinder3D.nn_modules.blocks import AsymmResBlock, DownsamplingAsymmResBlock, UpBlock, ReconBlock


class SparseVoxelSegmentor(nn.Module):
    """
    Cylindrical Sparse Voxel Segmentation Network

    Args:
        voxel_dim: Output tensor shape [D, H, W]
        num_input_features: Number of input point-wise features (C)
        num_classes: Number of output classes
        height: Height dimension
        init_size: Initial feature size (C_init)
    """

    def __init__(self, voxel_dim, num_input_features=128, num_classes=20, height=32, init_size=32):
        """
        Architecture:

        [Input Sparse Points (N, C)]
                ↓
        1. Context Network (ResContextBlock)
           - Feature extraction: C → C_init
                ↓
        2. Encoder (ResBlocks with pooling)
           - 4 stages with progressive downsampling
           - Features: C_init → 2C → 4C → 8C → 16C
           - Resolution: (D, H, W) → (D/4, H/8, W/8)
                ↓
        3. Decoder (UpBlocks with skip connections)
           - 4 stages of upsampling with skip connections
           - Features: 16C → 8C → 4C → 2C
           - Resolution: (D/4, H/8, W/8) → (D, H, W)
                ↓
        4. Reconstruction (ReconBlock)
           - Multi-directional feature refinement (XYZ)
           - Attention mechanism
                ↓
        5. Classification Head
           - Final semantic prediction
           - Output: (B, num_classes, D, H, W)
        """
        super().__init__()

        self.voxel_dim = np.array(voxel_dim)

        # 1. Initial feature extraction
        self.resblock = AsymmResBlock(
            in_channels=num_input_features,
            out_channels=init_size,
            indice_key="pre"
        )

        # 2. Encoder Network (4 stages)
        CH = init_size
        self.encoder = nn.ModuleList([
            # Stage 1: C_init → 2C, height pooling
            DownsamplingAsymmResBlock(CH, 2*CH,
                                      dropout_rate=0.2,
                                      do_height_pooling=True,
                                      indice_key="down2"),
            # Stage 2: 2C → 4C, height pooling
            DownsamplingAsymmResBlock(2*CH, 4*CH,
                                      dropout_rate=0.2,
                                      do_height_pooling=True,
                                      indice_key="down3"),
            # Stage 3: 4C → 8C, spatial pooling
            DownsamplingAsymmResBlock(4*CH, 8*CH,
                                      dropout_rate=0.2,
                                      do_pooling=True,
                                      do_height_pooling=False,
                                      indice_key="down4"),
            # Stage 4: 8C → 16C, spatial pooling
            DownsamplingAsymmResBlock(8*CH, 16*CH,
                                      dropout_rate=0.2,
                                      do_pooling=True,
                                      do_height_pooling=False,
                                      indice_key="down5")
        ])

        # 3. Decoder Network (4 stages)
        self.decoder = nn.ModuleList([
            # Stage 1: 16C → 16C
            UpBlock(16*CH, 16*CH,
                    indice_key="up0", up_key="down5"),
            # Stage 2: 16C → 8C
            UpBlock(16*CH, 8*CH,
                    indice_key="up1", up_key="down4"),
            # Stage 3: 8C → 4C
            UpBlock(8*CH, 4*CH,
                    indice_key="up2", up_key="down3"),
            # Stage 4: 4C → 2C
            UpBlock(4*CH, 2*CH,
                    indice_key="up3", up_key="down2")
        ])

        # 4. Reconstruction Network
        self.reconblock = ReconBlock(2*CH, 2*CH, indice_key="recon")

        # 5. Classification Head
        self.classifier = spconv.SubMConv3d(
            in_channels=4*CH,  # Concatenated features
            out_channels=num_classes,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True,
            indice_key="logit"
        )

    def forward(self, voxel_features, coords, batch_size):
        """
        Forward pass through the network

        Flow:
            1. Create sparse tensor from input points
            2. Context feature extraction
            3. Encoder with skip connections
            4. Decoder with feature fusion
            5. Reconstruction and classification

        Args:
            voxel_features: (N, C) Point features
            coords: (N, 4) Point coordinates (batch_idx, x, y, z)
            batch_size: Number of samples in batch

        Returns:
            y: Dense prediction tensor [B, num_classes, D, H, W]
        """

        # Create initial sparse tensor
        x = spconv.SparseConvTensor(features=voxel_features,
                                    indices=coords.int(),
                                    spatial_shape=self.voxel_dim,
                                    batch_size=batch_size)

        # 1. Feature extraction
        x = self.resblock(x)
        # 2. Encoder network
        skip_features = []
        for i, encoder in enumerate(self.encoder):
            x, skip = encoder(x)
            skip_features.append(skip)

        # 3. Decoder network
        for i, decoder in enumerate(self.decoder):
            x = decoder(x, skip_features[-i-1])

        # 4. Reconstruction
        recon_features = self.reconblock(x)

        # 5. Feature fusion
        fused_features = torch.cat(
            (recon_features.features, x.features),
            dim=1
        )
        recon_features.features = fused_features

        # 6. Classification
        logits = self.classifier(recon_features)
        y = logits.dense()

        return y


if __name__ == "__main__":

    net = SparseVoxelSegmentor(voxel_dim=[480, 360, 32],
                               num_input_features=128,
                               num_classes=20,
                               height=32,
                               init_size=32)
    for key in net.state_dict().keys():
        print(key)
    print(len(net.state_dict().keys()))
    print(net)