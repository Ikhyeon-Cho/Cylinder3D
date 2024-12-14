"""
Author: Ikhyeon Cho
Link: https://github.com/Ikhyeon-Cho
File: model.py
Date: 2024/12/14 17:03
"""

from Cylinder3D.nn_modules.cylindrical_point_encoder import CylindricalPointEncoder
from Cylinder3D.nn_modules.cylindrical_voxel_segmentor import SparseVoxelSegmentor
import torch.nn as nn


class Cylinder3D(nn.Module):
    """
    Cylinder3D Network for 3D LiDAR Semantic Segmentation

    Architecture:
        1. Point Feature Extraction
        2. Sparse Voxel Segmentation
    """

    def __init__(self, voxel_dim, num_classes, cfg: dict):
        super().__init__()

        # Point feature extraction module
        self.point_encoder = CylindricalPointEncoder(input_dim=9,
                                                     feat_projection_dim=cfg['feature_compression'],
                                                     out_feature_dim=256)

        # Sparse voxel segmentation module
        self.voxel_segmentor = SparseVoxelSegmentor(
            voxel_dim=voxel_dim,
            num_input_features=16,
            num_classes=num_classes,
            height=32,
            init_size=32
        )

    def forward(self, feats, coords, batch_size):
        """
        Forward pass

        Args:
            feats: (N, C) Point features
            coords: (N, 4) Point coordinates
            batch_size: Number of samples in batch

        Returns:
            voxel_semantics: (B, num_classes, D, H, W) Semantic segmentation logits
        """
        # 1. Extract point features
        point_features, point_coords = self.point_encoder(feats, coords)

        # 2. Voxel-based segmentation
        voxel_semantics = self.voxel_segmentor(
            point_features, point_coords, batch_size)

        return voxel_semantics


if __name__ == "__main__":

    net = Cylinder3D(voxel_dim=16, num_classes=19,
                     cfg={'feature_compression': 16})
    for key in net.state_dict().keys():
        print(key)
        break

    # load new_state_dict.pth with some layers
    import torch
    pretrained_weights = torch.load('./new_state_dict.pth')
    net.load_state_dict(pretrained_weights, strict=False)