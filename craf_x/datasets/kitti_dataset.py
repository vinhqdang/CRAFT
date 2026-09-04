import os
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from ..config import CRAFXConfig

# KITTI 3D object detection tracks exactly these three classes; everything
# else ('Van', 'Truck', 'Person_sitting', 'Tram', 'Misc', 'DontCare') is
# dropped, matching craf_x's 3-channel KITTI heatmap head.
KITTI_CLASSES = {"Car": 0, "Pedestrian": 1, "Cyclist": 2}


class CRAFXKittiDataset(Dataset):
    """
    Wrapper for the KITTI 3D Object Detection Dataset.
    Single camera and single LiDAR sweep.

    If `data_root/split` contains a real KITTI layout (image_2/, velodyne/,
    calib/, label_2/), samples are parsed from disk: the image is read and
    resized, the LiDAR point cloud is voxelized into a `config.bev_channels`-
    agnostic 4-channel BEV raster (occupancy, max height, mean intensity,
    point-count density) over `x_range`/`y_range`/`z_range`, and 3D box
    labels (camera frame) are projected via the calibration matrices into
    the same BEV grid to build the heatmap/regression targets.

    If that layout isn't found, falls back to a small dummy dataset
    returning correctly-shaped random tensors, so pipelines can still be
    exercised without real data (matches the other dataset wrappers).

    Caveats of this template implementation:
    - No velocity ground truth exists in KITTI (single frame, no tracking),
      so the 'V' target is always zero.
    - The regression head only predicts (dx, dy, z, w, l, h) — no yaw/heading
      channel — so `rotation_y` is parsed but not used as a training target.
    - The CCP contrastive mask 'm' is set to all-ones: on real, synchronized,
      non-adversarial sensor data, camera and LiDAR are assumed consistent
      everywhere in their shared BEV footprint.
    """

    def __init__(
        self,
        data_root: str,
        split: str = "training",
        config: Optional[CRAFXConfig] = None,
        image_size: Optional[int] = None,
        x_range: Tuple[float, float] = (0.0, 70.4),
        y_range: Tuple[float, float] = (-40.0, 40.0),
        z_range: Tuple[float, float] = (-3.0, 1.0),
    ):
        self.data_root = data_root
        self.split = split
        self.config = config or CRAFXConfig()
        self.image_size = image_size or self.config.bev_h
        self.x_range = x_range
        self.y_range = y_range
        self.z_range = z_range

        self._image_dir = os.path.join(data_root, split, "image_2")
        self._velodyne_dir = os.path.join(data_root, split, "velodyne")
        self._calib_dir = os.path.join(data_root, split, "calib")
        self._label_dir = os.path.join(data_root, split, "label_2")

        self.is_real = os.path.isdir(self._image_dir) and os.path.isdir(self._velodyne_dir)
        if self.is_real:
            self.sample_indices = sorted(
                os.path.splitext(f)[0] for f in os.listdir(self._image_dir) if f.endswith(".png")
            )
            if not self.sample_indices:
                warnings.warn(f"No .png images found under {self._image_dir}. Falling back to dummy mode.")
                self.is_real = False

        if not self.is_real:
            self.sample_indices = ["000000", "000001", "000002"]  # mocked indices

    def __len__(self):
        return len(self.sample_indices)

    def __getitem__(self, idx):
        sample_id = self.sample_indices[idx]

        if not self.is_real:
            return self._dummy_sample(sample_id)

        image = self._read_image(sample_id)
        points = self._read_velodyne(sample_id)
        pointcloud = self._points_to_bev(points)
        calib = self._read_calib(sample_id)
        labels = self._read_labels(sample_id)
        targets = self._labels_to_targets(labels, calib)
        m = torch.ones(1, self.config.bev_h, self.config.bev_w)

        return {
            "image": image,
            "pointcloud": pointcloud,
            "m": m,
            "targets": targets,
            "idx": sample_id,
        }

    def _dummy_sample(self, sample_id: str) -> Dict:
        image = torch.randn(3, self.image_size, self.image_size)
        pointcloud = torch.randn(4, self.config.bev_h, self.config.bev_w)
        m = torch.randint(0, 2, (1, self.config.bev_h, self.config.bev_w)).float()

        targets = {
            "H": torch.zeros(len(KITTI_CLASSES), self.config.bev_h, self.config.bev_w),
            "B": torch.zeros(6, self.config.bev_h, self.config.bev_w),
            "V": torch.zeros(2, self.config.bev_h, self.config.bev_w),
        }

        return {
            "image": image,
            "pointcloud": pointcloud,
            "m": m,
            "targets": targets,
            "idx": sample_id,
        }

    def _read_image(self, sample_id: str) -> torch.Tensor:
        path = os.path.join(self._image_dir, f"{sample_id}.png")
        img = Image.open(path).convert("RGB").resize((self.image_size, self.image_size))
        array = np.asarray(img, dtype=np.float32) / 255.0  # (H, W, 3)
        return torch.from_numpy(array).permute(2, 0, 1).contiguous()

    def _read_velodyne(self, sample_id: str) -> np.ndarray:
        path = os.path.join(self._velodyne_dir, f"{sample_id}.bin")
        return np.fromfile(path, dtype=np.float32).reshape(-1, 4)

    def _read_calib(self, sample_id: str) -> Dict[str, np.ndarray]:
        path = os.path.join(self._calib_dir, f"{sample_id}.txt")
        calib = {}
        with open(path, "r") as f:
            for line in f:
                if ":" not in line:
                    continue
                key, values = line.split(":", 1)
                values = values.strip()
                if not values:
                    continue
                calib[key.strip()] = np.array([float(v) for v in values.split()])

        r0_rect = np.eye(4)
        r0_rect[:3, :3] = calib["R0_rect"].reshape(3, 3)

        tr_velo_to_cam = np.eye(4)
        tr_velo_to_cam[:3, :] = calib["Tr_velo_to_cam"].reshape(3, 4)

        return {"R0_rect": r0_rect, "Tr_velo_to_cam": tr_velo_to_cam}

    def _read_labels(self, sample_id: str) -> List[Dict]:
        path = os.path.join(self._label_dir, f"{sample_id}.txt")
        if not os.path.isfile(path):
            return []

        objects = []
        with open(path, "r") as f:
            for line in f:
                fields = line.strip().split()
                if len(fields) < 15:
                    continue
                obj_type = fields[0]
                if obj_type not in KITTI_CLASSES:
                    continue
                height, width, length = (float(v) for v in fields[8:11])
                x, y, z = (float(v) for v in fields[11:14])
                rotation_y = float(fields[14])
                objects.append(
                    {
                        "class": obj_type,
                        "dimensions": (height, width, length),
                        "location_cam": (x, y, z),
                        "rotation_y": rotation_y,
                    }
                )
        return objects

    def _camera_to_velodyne(self, point_cam: np.ndarray, calib: Dict[str, np.ndarray]) -> np.ndarray:
        """Camera-frame (x, y, z) -> velodyne-frame (x, y, z), per the KITTI devkit convention."""
        point_cam_h = np.array([point_cam[0], point_cam[1], point_cam[2], 1.0])
        cam_to_velo = np.linalg.inv(calib["Tr_velo_to_cam"]) @ np.linalg.inv(calib["R0_rect"])
        point_velo_h = cam_to_velo @ point_cam_h
        return point_velo_h[:3]

    def _to_grid_indices(self, x: float, y: float) -> Optional[Tuple[int, int]]:
        x_min, x_max = self.x_range
        y_min, y_max = self.y_range
        if not (x_min <= x < x_max and y_min <= y < y_max):
            return None
        row = int((x - x_min) / (x_max - x_min) * self.config.bev_h)
        col = int((y - y_min) / (y_max - y_min) * self.config.bev_w)
        row = min(max(row, 0), self.config.bev_h - 1)
        col = min(max(col, 0), self.config.bev_w - 1)
        return row, col

    def _points_to_bev(self, points: np.ndarray) -> torch.Tensor:
        """
        Voxelizes a raw (N, 4) [x, y, z, intensity] LiDAR sweep into a
        4-channel BEV raster: occupancy, normalized max height, mean
        intensity, and a log-scaled point-count density.
        """
        bev = np.zeros((4, self.config.bev_h, self.config.bev_w), dtype=np.float32)
        counts = np.zeros((self.config.bev_h, self.config.bev_w), dtype=np.float32)
        height_sum_max = np.full((self.config.bev_h, self.config.bev_w), -np.inf, dtype=np.float32)
        intensity_sum = np.zeros((self.config.bev_h, self.config.bev_w), dtype=np.float32)

        z_min, z_max = self.z_range
        for x, y, z, intensity in points:
            cell = self._to_grid_indices(x, y)
            if cell is None or not (z_min <= z <= z_max):
                continue
            row, col = cell
            counts[row, col] += 1
            intensity_sum[row, col] += intensity
            height_sum_max[row, col] = max(height_sum_max[row, col], z)

        occupied = counts > 0
        bev[0][occupied] = 1.0
        normalized_height = np.zeros_like(height_sum_max)
        normalized_height[occupied] = (height_sum_max[occupied] - z_min) / (z_max - z_min)
        bev[1] = np.clip(normalized_height, 0.0, 1.0)
        bev[2][occupied] = intensity_sum[occupied] / counts[occupied]
        bev[3] = np.log1p(counts) / np.log1p(counts.max()) if counts.max() > 0 else counts

        return torch.from_numpy(bev)

    def _labels_to_targets(self, labels: List[Dict], calib: Dict[str, np.ndarray]) -> Dict[str, torch.Tensor]:
        heatmap = torch.zeros(len(KITTI_CLASSES), self.config.bev_h, self.config.bev_w)
        regression = torch.zeros(6, self.config.bev_h, self.config.bev_w)
        velocity = torch.zeros(2, self.config.bev_h, self.config.bev_w)  # no velocity GT in KITTI

        z_min, z_max = self.z_range
        for obj in labels:
            x, y, z = self._camera_to_velodyne(np.array(obj["location_cam"]), calib)
            cell = self._to_grid_indices(x, y)
            if cell is None:
                continue
            row, col = cell
            class_idx = KITTI_CLASSES[obj["class"]]

            x_min, x_max = self.x_range
            y_min, y_max = self.y_range
            cell_x_size = (x_max - x_min) / self.config.bev_h
            cell_y_size = (y_max - y_min) / self.config.bev_w
            dx = ((x - x_min) % cell_x_size) / cell_x_size
            dy = ((y - y_min) % cell_y_size) / cell_y_size
            z_norm = float(np.clip((z - z_min) / (z_max - z_min), 0.0, 1.0))
            height, width, length = obj["dimensions"]

            heatmap[class_idx, row, col] = 1.0
            regression[:, row, col] = torch.tensor([dx, dy, z_norm, width, length, height])

        return {"H": heatmap, "B": regression, "V": velocity}
