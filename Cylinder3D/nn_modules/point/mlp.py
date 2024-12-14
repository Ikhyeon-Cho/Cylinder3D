import torch.nn as nn
import torch.nn.init as init


class PointMLP(nn.Module):
    def __init__(self,
                 point_feature_dim,
                 hidden_dims=[64, 128, 256],
                 out_feature_dim=256):
        """
        Point Processing model with MLP layers

        Args:
            point_features: Input point feature dimension
            hidden_dims: List of hidden layer dimensions
            out_features: Processed point feature dimension
        """
        super(PointMLP, self).__init__()

        self.mlp = nn.Sequential(
            nn.BatchNorm1d(point_feature_dim),

            nn.Linear(point_feature_dim, hidden_dims[0]),
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

        self.initialize_weights()

    def initialize_weights(self):
        """
        Carefully initialize all layers:
        - Linear layers: Kaiming initialization for ReLU networks
        - BatchNorm: Weight=1, Bias=0 for better initial behavior
        """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # Kaiming initialization for ReLU networks
                init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    # Initialize bias with small constant for better gradient flow
                    init.constant_(m.bias, 0.01)

            elif isinstance(m, nn.BatchNorm1d):
                # Standard initialization for BatchNorm
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)

        # Special initialization for the last layer
        if isinstance(self.mlp[-1], nn.Linear):
            # Xavier initialization for the last layer
            # Helps with initial predictions by keeping variance in check
            init.xavier_normal_(self.mlp[-1].weight, gain=0.01)
            if self.mlp[-1].bias is not None:
                init.constant_(self.mlp[-1].bias, 0)

    def forward(self, feats):
        """
        Forward pass

        Args:
            feats: (N, C) Point features

        Returns:
            processed_feats: (N, out_features) Processed point features
        """
        processed_feats = self.mlp(feats)
        return processed_feats


if __name__ == "__main__":

    model = PointMLP(point_feature_dim=3,
                     hidden_dims=[64, 128, 256],
                     out_feature_dim=256)
    import torch
    feats = torch.randn(10, 3)
    processed_feats = model(feats)
    print(feats.shape)
    print(processed_feats.shape)
