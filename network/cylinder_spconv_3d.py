# -*- coding:utf-8 -*-
# author: Xinge
# @file: cylinder_spconv_3d.py

from torch import nn

REGISTERED_MODELS_CLASSES = {}


def register_model(cls, name=None):
    global REGISTERED_MODELS_CLASSES
    if name is None:
        name = cls.__name__
    assert name not in REGISTERED_MODELS_CLASSES, f"exist class: {REGISTERED_MODELS_CLASSES}"
    REGISTERED_MODELS_CLASSES[name] = cls
    return cls


def get_model_class(name):
    global REGISTERED_MODELS_CLASSES
    assert name in REGISTERED_MODELS_CLASSES, f"available class: {REGISTERED_MODELS_CLASSES}"
    return REGISTERED_MODELS_CLASSES[name]


@register_model
class cylinder_asym(nn.Module):
    def __init__(self,
                 cylin_model,
                 segmentator_spconv,
                 sparse_shape,
                 ):
        super().__init__()
        self.name = "cylinder_asym"

        self.cylinder_3d_generator = cylin_model

        self.cylinder_3d_spconv_seg = segmentator_spconv

        self.sparse_shape = sparse_shape

    def forward(self, train_pt_fea_ten, train_vox_ten, batch_size):

        coords, features_3d = self.cylinder_3d_generator(
            train_pt_fea_ten, train_vox_ten)

        spatial_features = self.cylinder_3d_spconv_seg(
            features_3d, coords, batch_size)

        return spatial_features


if __name__ == "__main__":
    import torch
    from network.segmentator_3d_asymm_spconv import Asymm_3d_spconv
    from network.cylinder_fea_generator import cylinder_fea

    output_shape = [256, 256, 32]
    use_norm = True
    num_input_features = 128
    init_size = 16
    num_class = 20
    fea_dim = 128
    out_fea_dim = 128

    cylinder_3d_spconv_seg = Asymm_3d_spconv(
        output_shape=output_shape,
        use_norm=use_norm,
        num_input_features=num_input_features,
        init_size=init_size,
        nclasses=num_class)

    cy_fea_net = cylinder_fea(grid_size=output_shape,
                              fea_dim=fea_dim,
                              out_pt_fea_dim=out_fea_dim,
                              fea_compre=num_input_features)
    
    model = cylinder_asym(cylin_model=cy_fea_net, segmentator_spconv=cylinder_3d_spconv_seg, sparse_shape=output_shape)
    
    # dummy data
    N = 1000
    feats = torch.randn(N, 9)
    coords = torch.randint(0, 256, (N, 3))
    batch_idx_tensor = torch.full((N, 1), 0, dtype=torch.long)
    coords = torch.cat([batch_idx_tensor, coords], dim=1)
    batch_size = 1

    # forward pass
    voxel_semantics = model(feats, coords, batch_size)
    print(voxel_semantics.shape)
