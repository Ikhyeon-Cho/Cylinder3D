# read statc dict
import torch
import yaml

with open('configs/semantickitti.yaml', 'r') as file:
    config = yaml.safe_load(file)

model_load_path = config['TRAIN']['model_load_path']
state_dict = torch.load(model_load_path)

for key in state_dict.keys():
    print(key)

# new_state_dict = state_dict.copy()

# for key in state_dict.keys():
#     new_key = key
#     if key.startswith('point_encoder.feature_projection.'):
#         new_key = key.replace(
#             'point_encoder.feature_projection.', 'point_encoder.feature_transform.')
#     # elif key.startswith('cylinder_3d_generator.fea_compression.'):
#         # new_key = key.replace(
#         # 'cylinder_3d_generator.fea_compression.', 'point_encoder.mlp_feature_projection.')

#     new_state_dict[new_key] = state_dict[key]

# print("New state dict:")
# for key in new_state_dict.keys():
#     print(key)


def adapt_state_dict(state_dict):
    new_state_dict = {}

    # Create mapping rules
    key_mapping = {
        'point_encoder.feature_projection': 'point_encoder.feature_transform',
        'cylinder_3d_spconv_seg.downCntx': 'voxel_segmentor.resblock',
        'cylinder_3d_spconv_seg.resBlock': 'voxel_segmentor.encoder',
        'cylinder_3d_spconv_seg.upBlock': 'voxel_segmentor.decoder',
        'cylinder_3d_spconv_seg.ReconNet': 'voxel_segmentor.reconblock',
        'cylinder_3d_spconv_seg.logits': 'voxel_segmentor.classifier'
    }

    for old_key, param in state_dict.items():
        new_key = old_key

        # Handle point encoder keys
        if old_key.startswith('cylinder_3d_generator.PPmodel.'):
            new_key = old_key.replace(
                'cylinder_3d_generator.PPmodel.', 'point_encoder.mlp.')
        elif old_key.startswith('cylinder_3d_generator.fea_compression.'):
            new_key = old_key.replace(
                'cylinder_3d_generator.fea_compression.', 'point_encoder.feature_transform.')

        # Handle voxel segmentor keys
        for old_prefix, new_prefix in key_mapping.items():
            if old_key.startswith(old_prefix):
                # Special handling for resBlock2-5 to encoder.0-3
                if 'resBlock' in old_prefix:
                    # Convert resBlock2-5 to 0-3
                    block_num = int(old_key.split('.')[1][-1]) - 2
                    remaining = '.'.join(old_key.split('.')[2:])
                    new_key = f'{new_prefix}.{block_num}.{remaining}'
                    break
                # Special handling for upBlock0-3 to decoder.0-3
                elif 'upBlock' in old_prefix:
                    # Extract upBlock number
                    block_num = int(old_key.split('.')[1][-1])
                    remaining = '.'.join(old_key.split('.')[2:])
                    new_key = f'{new_prefix}.{block_num}.{remaining}'
                    break
                else:
                    new_key = old_key.replace(old_prefix, new_prefix)
                    break

        # Additional transformations for specific layer names
        new_key = new_key.replace('conv1', 'skip_connection.0.conv')
        new_key = new_key.replace('bn0', 'skip_connection.0.bn')
        new_key = new_key.replace('conv1_2', 'skip_connection.1.conv')
        new_key = new_key.replace('bn0_2', 'skip_connection.1.bn')
        new_key = new_key.replace('conv2', 'residual.0.conv')
        new_key = new_key.replace('bn1', 'residual.0.bn')
        new_key = new_key.replace('conv3', 'residual.1.conv')
        new_key = new_key.replace('bn2', 'residual.1.bn')
        new_key = new_key.replace('pool', 'pooling')

        # Handle reconstruction network keys
        if 'ReconNet' in old_key:
            new_key = new_key.replace('conv1', 'recon_r')
            new_key = new_key.replace('conv1_2', 'recon_theta')
            new_key = new_key.replace('conv1_3', 'recon_z')

        new_state_dict[new_key] = param

    return new_state_dict

new_state_dict = adapt_state_dict(state_dict)
torch.save(new_state_dict, './new_state_dict.pth')
