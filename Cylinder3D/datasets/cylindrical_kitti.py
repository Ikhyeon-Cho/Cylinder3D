from semantic_kitti_dataset.kitti import Segmentation
import numpy as np
import numba as nb
import torch


def cartesian2Polar(points: np.ndarray) -> np.ndarray:
    # r= root(x^2 + y^2)
    # θ= arctan(y/x)
    # z = z
    # intensity = intensity (if exists)
    r = np.sqrt(points[:, 0] ** 2 + points[:, 1] ** 2)
    theta = np.arctan2(points[:, 1], points[:, 0])
    if points.shape[1] > 3:  # Has intensity
        return np.stack((r, theta, points[:, 2], points[:, 3]), axis=1)
    return np.stack((r, theta, points[:, 2]), axis=1)


def polar2Cartesian(points_polar: np.ndarray) -> np.ndarray:
    # x = r*cos(θ)
    # y = r*sin(θ)
    # z = z
    # additional channels remain unchanged
    x = points_polar[0] * np.cos(points_polar[1])
    y = points_polar[0] * np.sin(points_polar[1])
    if points_polar.shape[0] > 3:  # Has additional channels
        return np.concatenate(([x, y, points_polar[2]], points_polar[3:]))
    return np.array([x, y, points_polar[2]])


def coord_quantization(points: np.ndarray, voxel_dim: np.ndarray,
                       min_bound=None, max_bound=None):
    if min_bound is None:
        min_bound = np.min(points, axis=0)
    if max_bound is None:
        max_bound = np.max(points, axis=0)
    crop_range = max_bound - min_bound
    voxel_resolution = crop_range / (voxel_dim - 1)
    quantized_coordinates = (np.floor(
        (np.clip(points, min_bound, max_bound) - min_bound) / voxel_resolution)).astype(int)

    return quantized_coordinates, voxel_resolution


class CylindricalKITTI(Segmentation):
    def __init__(self,
                 kitti_root: str,
                 voxel_dim,
                 phase: str,
                 rotate_aug=False,
                 flip_aug=False,
                 scale_aug=False,
                 transform_aug=False,
                 empty_voxel_label=0,
                 return_test=False,
                 fixed_volume_space=True,
                 min_voxel_coords=[0, -np.pi, -4],
                 max_voxel_coords=[50, np.pi, 2],
                 trans_std=[0.1, 0.1, 0.1],
                 min_rad=-np.pi / 4,
                 max_rad=np.pi / 4):

        super().__init__(kitti_root, phase)

        # Cylinder specific parameters
        self.voxel_dim = np.asarray(voxel_dim)

        self.fixed_volume_space = fixed_volume_space
        self.min_voxel_coords = np.asarray(min_voxel_coords)
        self.max_voxel_coords = np.asarray(max_voxel_coords)

        # Augmentation parameters
        self.rotate_aug = rotate_aug
        self.flip_aug = flip_aug
        self.scale_aug = scale_aug
        self.transform = transform_aug
        self.trans_std = trans_std
        self.noise_rotation = np.random.uniform(min_rad, max_rad)
        # Dataset parameters
        self.empty_voxel_label = empty_voxel_label
        self.return_test = return_test

    def __getitem__(self, index):

        item_dict = {  # To be returned
            "feats": None,
            "coords": None,
            "labels": None,
            "labels_voxel": None,
        }
        # Load KITTI scan-label pair
        kitti_sample = super().__getitem__(index)
        kitti_points = kitti_sample['points']  # (N, 4)
        kitti_labels = kitti_sample['labels']  # (N,)

        # Data augmentation
        kitti_points = self._augment_points(kitti_points)

        #######################################################################
        # xyz -> polar_xyz -> coordinate quantization -> point-voxel features #
        #######################################################################

        # Convert to polar coordinates: r, θ, z
        points = cartesian2Polar(kitti_points)

        # Coordinate quantization -> integer coordinates
        points_quantized, voxel_resolution = self._voxelize_points(points)
        point_coords = points_quantized[:, :3].astype(np.int32)

        # point-voxel features for PTnet
        voxel_centers = (point_coords.astype(np.float32) + 0.5) * \
            voxel_resolution + self.min_voxel_coords
        offset_to_center = points[:, :3] - voxel_centers
        point_features = np.concatenate(
            (offset_to_center, points[:, :3], kitti_points[:, :2], points[:, 3:]), axis=1)

        ###################################################
        # Process labels for voxels using majority voting #
        ###################################################

        voxel_label_pair = np.concatenate(
            [point_coords, kitti_labels[:, None]], axis=1)
        voxel_label_pair = voxel_label_pair[np.lexsort(
            (point_coords[:, 0],
             point_coords[:, 1],
             point_coords[:, 2])), :]

        # Majority voting for voxel labels
        processed_label = np.ones(
            self.voxel_dim, dtype=np.uint8) * self.empty_voxel_label
        processed_label = nb_process_label(
            np.copy(processed_label), voxel_label_pair)

        item_dict["feats"] = torch.from_numpy(point_features).float()
        item_dict["coords"] = torch.from_numpy(point_coords).long()
        item_dict["labels"] = kitti_labels
        item_dict["labels_voxel"] = torch.from_numpy(processed_label).long()

        return item_dict

    def collate_fn(self, batch):
        feats = []
        coords = []
        labels = []
        labels_voxel = []

        # Iterate over batch
        for batch_idx, item in enumerate(batch):
            feats.append(item["feats"])
            labels.append(item["labels"])
            labels_voxel.append(item["labels_voxel"])

            # Add batch index to coordinates
            N = len(item["feats"])
            batch_coords = item["coords"]
            batch_idx_tensor = torch.full((N, 1), batch_idx, dtype=torch.long)
            coords.append(torch.cat([batch_idx_tensor, batch_coords], dim=1))

        # Concatenate everything
        feats = torch.cat(feats, dim=0)           # [B*N, C]
        coords = torch.cat(coords, dim=0)         # [B*N, 4] (B, x, y, z)
        labels = torch.cat(labels, dim=0)         # [B*N]
        labels_voxel = torch.stack(labels_voxel)  # [B, H, W, D]

        return {
            "feats": feats,
            "coords": coords,
            "labels": labels,
            "labels_voxel": labels_voxel
        }

    def __rotate_points(self, points):
        "Random data augmentation by rotation"
        rotate_rad = np.deg2rad(np.random.random() * 90) - np.pi / 4
        c, s = np.cos(rotate_rad), np.sin(rotate_rad)
        j = np.matrix([[c, s], [-s, c]])
        points[:, :2] = np.dot(points[:, :2], j)
        return points

    def __flip_points(self, points):
        "Random data augmentation by flip x, y, or x+y"
        flip_type = np.random.choice(4, 1)
        if flip_type == 1:
            points[:, 0] = -points[:, 0]
        elif flip_type == 2:
            points[:, 1] = -points[:, 1]
        elif flip_type == 3:
            points[:, :2] = -points[:, :2]
        return points

    def __scale_points(self, points):
        "Random data augmentation by scaling"
        noise_scale = np.random.uniform(0.95, 1.05)
        points[:, 0] = noise_scale * points[:, 0]
        points[:, 1] = noise_scale * points[:, 1]
        return points

    def __transform_points(self, points):
        "Random data augmentation by translation"
        noise_translate = np.array([np.random.normal(0, self.trans_std[0], 1),
                                    np.random.normal(0, self.trans_std[1], 1),
                                    np.random.normal(0, self.trans_std[2], 1)]).T
        points[:, 0:3] += noise_translate
        return points

    def _augment_points(self, points):
        "Data augmentation"
        if self.rotate_aug:
            points = self.__rotate_points(points)
        if self.flip_aug:
            points = self.__flip_points(points)
        if self.scale_aug:
            points = self.__scale_points(points)
        if self.transform:
            points = self.__transform_points(points)
        return points

    def _voxelize_points(self, points: np.ndarray) -> tuple:
        """Quantize points into a cylindrical grid

        Returns:
            quantized_coordinates: (N,3) Quantized coordinates
            voxel_resolution: (3,) Voxel resolution
        """

        # Coordinate quantization: r, θ, z
        point_coords = points[:, :3]
        if self.fixed_volume_space:
            min_bound = self.min_voxel_coords
            max_bound = self.max_voxel_coords
        else:
            min_bound = np.min(point_coords, axis=0)
            max_bound = np.max(point_coords, axis=0)

        quantized_coordinates, voxel_resolution = coord_quantization(
            point_coords, self.voxel_dim, min_bound, max_bound)

        # concatenate quantized coordinates with additional channels
        quantized_points = np.concatenate(
            (quantized_coordinates, points[:, 3:]), axis=1)

        return quantized_points, voxel_resolution

    def _get_voxel2point_mapping(self, voxel_dim: np.ndarray,
                                 voxel_resolution: np.ndarray,
                                 min_bound: np.ndarray):
        """Get the center points of each voxel"""

        voxel_indices = np.indices(self.voxel_dim)
        voxel_centers = voxel_indices * \
            voxel_resolution.reshape(-1, 1, 1, 1) + \
            min_bound.reshape(-1, 1, 1, 1)

        return voxel_centers


@nb.jit('u1[:,:,:](u1[:,:,:],i8[:,:])', nopython=True, cache=True, parallel=False)
def nb_process_label(processed_label, sorted_label_voxel_pair):
    label_size = 256
    counter = np.zeros((label_size,), dtype=np.uint16)
    counter[sorted_label_voxel_pair[0, 3]] = 1
    cur_sear_ind = sorted_label_voxel_pair[0, :3]
    for i in range(1, sorted_label_voxel_pair.shape[0]):
        cur_ind = sorted_label_voxel_pair[i, :3]
        if not np.all(np.equal(cur_ind, cur_sear_ind)):
            processed_label[cur_sear_ind[0], cur_sear_ind[1],
                            cur_sear_ind[2]] = np.argmax(counter)
            counter = np.zeros((label_size,), dtype=np.uint16)
            cur_sear_ind = cur_ind
        counter[sorted_label_voxel_pair[i, 3]] += 1
    processed_label[cur_sear_ind[0], cur_sear_ind[1],
                    cur_sear_ind[2]] = np.argmax(counter)
    return processed_label


if __name__ == "__main__":

    dataset = CylindricalKITTI(kitti_root="/data/semanticKITTI/dataset",
                               voxel_dim=[100, 100, 100],
                               phase="train")
    data = dataset[0]
    print(data["coords"].shape)
    print(data["feats"].shape)
    print(data["labels"].shape)
    print(data["labels_voxel"].shape)
    print()

    from torch.utils.data import DataLoader
    cfg = {
        "batch_size": 10,
        "shuffle": True,
        "collate_fn": CylindricalKITTI.collate_fn
    }
    dataloader = DataLoader(dataset, **cfg)
    for i, sample_dict in enumerate(dataloader):
        print(f"batch size: {dataloader.batch_size}")
        print(f"coords shape: {sample_dict['coords'].shape}")
        print(f"feats shape: {sample_dict['feats'].shape}")
        print(f"labels shape: {sample_dict['labels'].shape}")
        print(f"labels_voxel shape: {sample_dict['labels_voxel'].shape}")

        break
