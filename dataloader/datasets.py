from semantic_kitti_pytorch.datasets import SemanticKITTI_Segmentation
import numpy as np
import numba as nb


def cartesian2polar(points: np.ndarray) -> np.ndarray:
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


@nb.jit('u1[:,:,:](u1[:,:,:],i8[:,:])', nopython=True, cache=True, parallel=False)
def process_voxel_labels(voxel_labels: np.ndarray, label_voxel_pairs: np.ndarray) -> np.ndarray:
    """Process labels for voxels using majority voting

    Args:
        voxel_labels: (H,W,D) Empty label grid initialized with ignore_label
        label_voxel_pairs: (N,4) array of [voxel_x, voxel_y, voxel_z, label]
    """
    label_size = 256  # Max number of label classes
    counter = np.zeros((label_size,), dtype=np.uint16)
    counter[label_voxel_pairs[0, 3]] = 1
    current_voxel = label_voxel_pairs[0, :3]

    # Count labels for each voxel and assign most frequent
    for i in range(1, label_voxel_pairs.shape[0]):
        voxel_idx = label_voxel_pairs[i, :3]
        if not np.all(np.equal(voxel_idx, current_voxel)):
            # Assign most frequent label to current voxel
            voxel_labels[current_voxel[0], current_voxel[1],
                         current_voxel[2]] = np.argmax(counter)
            # Reset for next voxel
            counter = np.zeros((label_size,), dtype=np.uint16)
            current_voxel = voxel_idx
        counter[label_voxel_pairs[i, 3]] += 1

    # Handle last voxel
    voxel_labels[current_voxel[0], current_voxel[1],
                 current_voxel[2]] = np.argmax(counter)
    return voxel_labels


class CylindricalKITTIDataset(SemanticKITTI_Segmentation):
    def __init__(self,
                 kitti_root: str,
                 voxel_dim,
                 phase: str,
                 rotate_aug=False,
                 flip_aug=False,
                 scale_aug=False,
                 transform_aug=False,
                 ignore_label=255,
                 return_test=False,
                 fixed_volume_space=False,
                 min_volume_space=[0, -np.pi, -4],
                 max_volume_space=[50, np.pi, 2],
                 trans_std=[0.1, 0.1, 0.1],
                 min_rad=-np.pi / 4,
                 max_rad=np.pi / 4):

        super().__init__(kitti_root, phase)

        # Cylinder specific parameters
        self.voxel_dim = np.asarray(voxel_dim)
        self.fixed_volume_space = fixed_volume_space
        self.max_volume_space = max_volume_space
        self.min_volume_space = min_volume_space

        # Augmentation parameters
        self.rotate_aug = rotate_aug
        self.flip_aug = flip_aug
        self.scale_aug = scale_aug
        self.transform = transform_aug
        self.trans_std = trans_std
        self.noise_rotation = np.random.uniform(min_rad, max_rad)
        # Dataset parameters
        self.ignore_label = ignore_label
        self.return_test = return_test

    def __getitem__(self, index):

        item_dict = {  # To be returned
            "feats": None,
            "coords": None,
            "labels": None,
        }

        # Load KITTI scan-label pair
        kitti_sample = super().__getitem__(index)
        points = kitti_sample['points']
        if self.phase != 'test':
            labels = kitti_sample['labels']
        else:
            labels = np.zeros_like(points[:, 3])

        ########################################
        ###### xyz -> polar_xyz -> 

        # Data augmentation
        points = self._augment_points(points)

        # Convert to polar coordinates
        xyz = points[:, :3]
        intensity = points[:, 3]
        polar_xyz = cartesian2polar(xyz)

        # Coordinate quantization: r, theta, z
        quantized_coordinates, intervals, min_bound, cylind2cart_lookup = self._voxelize_points(
            polar_xyz)

        processed_label = np.ones(self.voxel_dim, dtype=np.uint8)
        processed_label *= self.ignore_label

        label_voxel_pair = np.concatenate(
            [quantized_coordinates, labels[:, None]], axis=1)
        label_voxel_pair = label_voxel_pair[np.lexsort(
            (quantized_coordinates[:, 0],
             quantized_coordinates[:, 1],
             quantized_coordinates[:, 2])), :]

        processed_label = nb_process_label(
            np.copy(processed_label), label_voxel_pair)

        # center data on each voxel for PTnet
        voxel_centers = (quantized_coordinates.astype(np.float32) + 0.5) * \
            intervals + min_bound
        offset_to_center = polar_xyz - voxel_centers
        point_features = np.concatenate(
            (offset_to_center, polar_xyz, xyz[:, :2]), axis=1)

        item_dict["labels"] = processed_label
        item_dict["coords"] = cylind2cart_lookup
        item_dict["feats"] = point_features

        return item_dict

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

    def _voxelize_points(self, points_polar: np.ndarray):
        """Convert points to voxel grid coordinates"""

        coordinate_quantization = self.__coordinate_quantization(points_polar)
        quantized_coordinates, min_bound, max_bound, intervals = coordinate_quantization

        # Get cylindrical to cartesian lookup table
        dim_array = np.ones(len(self.voxel_dim) + 1, int)
        dim_array[0] = -1
        cylind2cart_lookup = np.indices(
            self.voxel_dim) * intervals.reshape(dim_array) + min_bound.reshape(dim_array)
        cylind2cart_lookup = polar2Cartesian(cylind2cart_lookup)

        return quantized_coordinates, intervals, min_bound, cylind2cart_lookup

    def __coordinate_quantization(self, points_polar):
        "Coordinate quantization: r, θ, z"
        if self.fixed_volume_space:
            max_bound = self.max_volume_space
            min_bound = self.min_volume_space
        else:
            max_bound_r = np.percentile(points_polar[:, 0], 100, axis=0)
            min_bound_r = np.percentile(points_polar[:, 0], 0, axis=0)
            max_bound = np.concatenate(
                ([max_bound_r], np.max(points_polar[:, 1:], axis=0)))
            min_bound = np.concatenate(
                ([min_bound_r], np.min(points_polar[:, 1:], axis=0)))

        # Get grid index
        crop_range = max_bound - min_bound
        intervals = crop_range / (self.voxel_dim - 1)
        if (intervals == 0).any():
            raise ValueError("Zero interval!")

        quantized_coordinates = (np.floor(
            (np.clip(points_polar, min_bound, max_bound) - min_bound) / intervals)).astype(int)
        return quantized_coordinates, min_bound, max_bound, intervals

    def _process_labels(self, grid_indices: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """Process point labels into voxel grid labels

        Args:
            grid_indices: (N,3) Voxel indices for each point
            labels: (N,) Label for each point
        Returns:
            processed_labels: (H,W,D) Voxel grid with majority labels
        """
        # Initialize empty label grid
        processed_labels = np.ones(
            self.voxel_dim, dtype=np.uint8) * self.ignore_label

        # Combine grid indices and labels
        label_voxel_pairs = np.concatenate(
            [grid_indices, labels[:, None]], axis=1)

        # Sort by voxel indices for processing
        label_voxel_pairs = label_voxel_pairs[np.lexsort(
            (grid_indices[:, 0], grid_indices[:, 1], grid_indices[:, 2])), :]

        # Process labels using numba-accelerated function
        processed_labels = process_voxel_labels(
            processed_labels, label_voxel_pairs)

        return processed_labels

    def rotation_points_single_angle(self, points, angle, axis=0):
        # points: [N, 3]
        rot_sin = np.sin(angle)
        rot_cos = np.cos(angle)
        if axis == 1:
            rot_mat_T = np.array(
                [[rot_cos, 0, -rot_sin], [0, 1, 0], [rot_sin, 0, rot_cos]],
                dtype=points.dtype)
        elif axis == 2 or axis == -1:
            rot_mat_T = np.array(
                [[rot_cos, -rot_sin, 0], [rot_sin, rot_cos, 0], [0, 0, 1]],
                dtype=points.dtype)
        elif axis == 0:
            rot_mat_T = np.array(
                [[1, 0, 0], [0, rot_cos, -rot_sin], [0, rot_sin, rot_cos]],
                dtype=points.dtype)
        else:
            raise ValueError("axis should in range")

        return points @ rot_mat_T


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

    dataset = CylindricalKITTIDataset(kitti_root="/data/semanticKITTI/dataset",
                                      voxel_dim=[100, 100, 100],
                                      phase="train")
    for i in range(10):
        item = dataset[i]
        print(item["coords"].shape)
        print(item["feats"].shape)
        print(item["labels"].shape)
        break
