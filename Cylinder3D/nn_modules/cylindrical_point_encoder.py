import torch
import torch.nn as nn
import torch.nn.init as init
import torch_scatter
from Cylinder3D.nn_modules.point.mlp import PointMLP


class CylindricalPointEncoder(nn.Module):
    def __init__(self,
                 input_dim=9,
                 hidden_dims=[64, 128, 256],
                 out_feature_dim=256,
                 feature_transform_dim=None):
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

        if feature_transform_dim is not None:
            self.feature_transform = nn.Sequential(
                nn.Linear(out_feature_dim, feature_transform_dim),
                nn.ReLU())
        else:
            self.feature_transform = None

        self.initialize_weights()

    def initialize_weights(self):
        """Initialize feature compression layers"""
        if self.feature_transform is not None:
            for m in self.feature_transform.modules():
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

        if self.feature_transform is not None:
            feats_max_pooled = self.feature_transform(feats_max_pooled)

        return feats_max_pooled, unq_coords


if __name__ == "__main__":

    net = CylindricalPointEncoder(input_dim=9,
                                  hidden_dims=[64, 128, 256],
                                  out_feature_dim=256,
                                  feature_transform_dim=16)

    print("Model state_dict:\n")
    for key in net.state_dict().keys():
        print(key)

    print(len(net.state_dict().keys()))
    print(net)
