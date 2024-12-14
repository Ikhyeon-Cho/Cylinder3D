import torch
import yaml

# model_load_path = './new_state_dict.pth'
model_load_path = '/workspace/SSC-toolbox/Cylinder3D/configs/model_save_backup.pt'
state_dict = torch.load(model_load_path)

for key in state_dict.keys():
    print(key)