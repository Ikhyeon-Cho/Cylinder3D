import torch
import torch.nn as nn
import torch.nn.init as init
import torch_scatter
from Cylinder3D.nn_modules.point.mlp import PointMLP


class CylindricalPointEncoder(nn.Module):
    def __init__(self,
                 input_dim,
                 hidden_dims=[64, 128, 256],
                 out_feature_dim=64,
                 feat_projection_dim=None):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.BatchNorm1d(input_dim),

            nn.Linear(input_dim, hidden_dims[0]),
            nn.BatchNorm1d(hidden_dims[0]),
            nn.ReLU(),

            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.BatchNorm1d(hidden_dims[1]),
            nn.ReLU(),

            nn.Linear(hidden_dims[1], hidden_dims[2]),
            nn.BatchNorm1d(hidden_dims[2]),
            nn.ReLU(),

            nn.Linear(hidden_dims[2], out_feature_dim)
        )

        if feat_projection_dim is not None:
            self.mlp_feature_projection = nn.Sequential(
                nn.Linear(out_feature_dim, feat_projection_dim),
                nn.ReLU())
        else:
            self.mlp_feature_projection = None

        self.initialize_weights()

    def initialize_weights(self):
        """Initialize feature compression layers"""
        if self.mlp_feature_projection is not None:
            for m in self.mlp_feature_projection.modules():
                if isinstance(m, nn.Linear):
                    init.kaiming_normal_(
                        m.weight, mode='fan_out', nonlinearity='relu')
                    if m.bias is not None:
                        init.constant_(m.bias, 0.01)

    def forward(self, feats, coords):
        # shuffle the points
        shuffled_ind = torch.randperm(feats.shape[0])
        feats = feats[shuffled_ind]
        coords = coords[shuffled_ind]

        # Get unique coordinates
        unq_coords, unq_inv, unq_cnt = torch.unique(coords,
                                                    return_inverse=True,
                                                    return_counts=True,
                                                    dim=0)
        unq_coords = unq_coords.type(torch.int64)

        # Point-wise feature extraction
        feats = self.mlp(feats)
        feats_max_pooled, _ = torch_scatter.scatter_max(feats, unq_inv, dim=0)

        if self.mlp_feature_projection is not None:
            feats_max_pooled = self.mlp_feature_projection(feats_max_pooled)

        return feats_max_pooled, unq_coords


if __name__ == "__main__":

    net = CylindricalPointEncoder(input_dim=3,
                                  hidden_dims=[64, 128, 256],
                                  out_feature_dim=64,
                                  feat_projection_dim=16)

    print("Model state_dict:\n")
    for key in net.state_dict().keys():
        print(key)

    print(len(net.state_dict().keys()))