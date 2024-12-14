"""
Author: Ikhyeon Cho
Link: https://github.com/Ikhyeon-Cho
File: dataset.py
"""

from Cylinder3D.datasets.cylindrical_kitti import CylindricalKITTI


class Cylinder3DDataset:

    def __init__(self, dataset_root: str, phase: str, cfg: dict):
        self.dataset_type = cfg['type'].lower()
        self.dataset = self._get_dataset(dataset_root, phase, cfg)

    def _get_dataset(self, dataset_root: str, phase: str, cfg: dict):
        if self.dataset_type == 'kitti':
            return CylindricalKITTI(
                kitti_root=dataset_root,
                voxel_dim=cfg['voxel_dim'],
                phase=phase
            )
        elif self.dataset_type == 'nuscenes':
            raise NotImplementedError(
                "NuScenes dataset is not implemented yet")
        else:
            raise ValueError(f"Dataset type {self.dataset_type} not supported")

    def __getitem__(self, index):
        return self.dataset[index]

    def __len__(self):
        return len(self.dataset)

    def collate_fn(self, batch):
        return self.dataset.collate_fn(batch)
