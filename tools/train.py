"""
Author: Ikhyeon Cho
Link: https://github.com/Ikhyeon-Cho
File: train.py
Date: 2024/12/14 16:22

Re-implementation of Cylinder3D.
Reference: https://github.com/xinge008/Cylinder3D
"""

import argparse
from Cylinder3D.model import Cylinder3D
from Cylinder3D.dataset import Cylinder3DDataset
from Cylinder3D.trainer import Cylinder3DTrainer
from utils.train import machine, seed, checkpoint
from utils.config import yaml_tools
from utils.system import time
import os
from torch.utils.data import DataLoader
import torch


def log_dirname(logger_yaml: dict):
    log_time = time.now(timezone=logger_yaml['timezone'])
    log_dirname = os.path.join(
        logger_yaml['log_dir'], f'Cylinder3D_SemanticKITTI_{log_time}')
    return log_dirname


def main(args):

    config_path = args.cfg
    print(f'============ Training routine: "{config_path}" ============')

    # 0. Set seed, device
    print('=> Setting seed and device...')
    seed.seed_all(42)
    device = machine.get_device()

    # 1. Load configs
    print('=> Loading configs...')
    yaml = yaml_tools.load_yaml(config_path)
    DATASET_CFG = yaml['DATASET']
    MODEL_CFG = yaml['MODEL']
    TRAIN_CFG = yaml['TRAIN']
    LOGGER_CFG = yaml['LOGGER']

    # 2. Load dataset
    print('=> Loading dataset...')
    DATASET_ROOT = args.dset_root
    if DATASET_ROOT is None:
        DATASET_ROOT = DATASET_CFG['root_dir']

    train_dataset = Cylinder3DDataset(
        DATASET_ROOT, cfg=DATASET_CFG, phase='train')
    val_dataset = Cylinder3DDataset(
        DATASET_ROOT, cfg=DATASET_CFG, phase='valid')
    cfg = {
        'batch_size': TRAIN_CFG['batch_size'],
        'num_workers': TRAIN_CFG['num_workers'],
        'collate_fn': train_dataset.collate_fn,
        'drop_last': True
    }
    dataloader_dict = {
        'train': DataLoader(train_dataset, **cfg, shuffle=False),
        'val': DataLoader(val_dataset, **cfg, shuffle=False),
    }

    # 3. Load model
    print('=> Loading network architecture...')
    model = Cylinder3D(num_classes=DATASET_CFG['num_class'],
                       voxel_dim=DATASET_CFG['voxel_dim'],
                       cfg=MODEL_CFG)

    # 4. Train model
    print('=> Training model...')
    trainer = Cylinder3DTrainer(model=model,
                                data=dataloader_dict,
                                cfg=TRAIN_CFG,
                                device=device,
                                log_dir=log_dirname(LOGGER_CFG))

    trainer.train()
    print('=> Training routine completed...')


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", default="configs/semantickitti.yaml")
    parser.add_argument("--dset_root", default=None)

    args = parser.parse_args()
    main(args)
