import json
import os
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from ..config import CRAFXConfig

# CADC's Scale-AI-provided 3D annotations cover ten labels (Car, Truck, Bus,
# Bicycle, Horse_and_Buggy, Pedestrian, Pedestrian_With_Object, Animal,
# Garbage_Containers_on_Wheels, Traffic_Guidance_Objects); most of the long
# tail is extremely rare (Horse_and_Buggy: 75 instances total across the
# whole dataset, Animal fewer still). Following CRAFXKittiDataset's
# precedent of keeping only well-populated classes, we keep the five with
# real support (Car 281941, Pedestrian 62851, Truck 20411, Bus 4867,
# Bicycle 785 instances dataset-wide per the CADC paper's Table VI/VII) and
# drop the rest, matching craf_x's fixed-width heatmap head.
CADC_CLASSES = {"Car": 0, "Truck": 1, "Bus": 2, "Pedestrian": 3, "Bicycle": 4}
CADC_NUM_CLASSES = 5

# Per-date road condition, from the CADC devkit's own
# cadc_dataset_route_stats.csv ("Road snow cover" column): every 2018_03_06
# and 2018_03_07 drive is "None" (bare road), every 2019_02_27 drive in our
# downloaded subset is "Covered" (snow-covered road). This is a real,
# dataset-provided severity split -- unlike Snowy Scenes, which required
# deriving severity ourselves from semantic-segmentation class fractions
# since it ships no such per-drive label at all.
_DATE_CATEGORY = {
    "2018_03_06": "bare",
    "2018_03_07": "bare",
    "2019_02_27": "covered",
}


class CRAFXCADCDataset(Dataset):
    """
    Wrapper for the Canadian Adverse Driving Conditions (CADC) dataset
    (Pitropov et al., IJRR 2021), read from an already-extracted local
    directory tree (unlike CRAFXSnowyScenesDataset, which reads lazily from
    a zip -- CADC's downloaded subset is small enough, and its per-drive
    folder structure awkward enough for zip-relative paths, that plain
    filesystem access is simpler here).

    Expected layout, matching `scripts/download_cadc.sh`'s output (itself
    matching the official devkit's own extraction of `labeled.zip` and
    `3d_ann.json`, see https://github.com/mpitropov/cadc_devkit):

        data_root/<date>/<drive>/labeled/image_0{camera_id}/data/<frame:010d>.png
        data_root/<date>/<drive>/labeled/lidar_points/data/<frame:010d>.bin
        data_root/<date>/<drive>/3d_ann.json

    (`labeled.zip`'s own internal listing has no wrapping folder per the
    CADC paper's Fig. 7, but this repo's own download/extract of it produces
    a `labeled/` subfolder -- confirmed against the actual downloaded and
    extracted files, not assumed from the paper diagram alone.)

    LiDAR points are (N, 4) float32 [x, y, z, intensity] -- identical raw
    layout to KITTI and Snowy Scenes, confirmed against the devkit's own
    `lidar_utils.py` (`np.fromfile(..., dtype=np.float32).reshape((-1, 4))`).
    3D annotation positions/yaw in `3d_ann.json` are already in the LiDAR
    frame (confirmed against `run_demo_tracklets.py`, which builds each
    cuboid's transform directly from `cuboid['position']`/`['yaw']` with no
    additional camera<->LiDAR step) -- no extrinsic transform needed for
    labels, same as Snowy Scenes and unlike KITTI.

    `sample_indices` entries are formatted `f"{category}_{date}_{drive}_{frame:010d}"`
    with `category` in `{"bare", "covered"}` (per-date road condition, from
    the devkit's own `cadc_dataset_route_stats.csv`), specifically so that
    `conformal_monitor.real_snow_stream.category_indices`/`category_subset`
    -- written for Snowy Scenes' `{category}_{frame}` convention -- work
    against this dataset completely unchanged.
    """

    def __init__(
        self,
        data_root: str,
        config: Optional[CRAFXConfig] = None,
        camera_id: int = 0,
        image_size: Optional[int] = None,
        x_range: Tuple[float, float] = (0.0, 70.4),
        y_range: Tuple[float, float] = (-40.0, 40.0),
        z_range: Tuple[float, float] = (-3.0, 1.0),
    ):
        self.data_root = data_root
        self.config = config or CRAFXConfig()
        self.camera_id = camera_id
        self.image_size = image_size or self.config.bev_h
        self.x_range = x_range
        self.y_range = y_range
        self.z_range = z_range

        self._frame_index: List[Tuple[str, str, int]] = []  # (date, drive, frame_num)
        self.sample_indices: List[str] = []
        self._ann_cache: Dict[Tuple[str, str], list] = {}

        if not os.path.isdir(data_root):
            warnings.warn(f"{data_root} does not exist. Dataset will be empty.")
            return

        for date in sorted(os.listdir(data_root)):
            date_dir = os.path.join(data_root, date)
            if not os.path.isdir(date_dir) or date not in _DATE_CATEGORY:
                continue
            category = _DATE_CATEGORY[date]

            for drive in sorted(os.listdir(date_dir)):
                drive_dir = os.path.join(date_dir, drive)
                image_dir = os.path.join(drive_dir, "labeled", f"image_0{camera_id}", "data")
                lidar_dir = os.path.join(drive_dir, "labeled", "lidar_points", "data")
                ann_path = os.path.join(drive_dir, "3d_ann.json")
                if not (os.path.isdir(image_dir) and os.path.isdir(lidar_dir) and os.path.isfile(ann_path)):
                    continue

                n_frames = len([f for f in os.listdir(image_dir) if f.endswith(".png")])
                for frame_num in range(n_frames):
                    self._frame_index.append((date, drive, frame_num))
                    self.sample_indices.append(f"{category}_{date}_{drive}_{frame_num:010d}")

        if not self._frame_index:
            warnings.warn(f"No CADC drives found under {data_root}. Dataset will be empty.")

    def __len__(self):
        return len(self._frame_index)

    def _annotations_for(self, date: str, drive: str) -> list:
        key = (date, drive)
        if key not in self._ann_cache:
            ann_path = os.path.join(self.data_root, date, drive, "3d_ann.json")
            with open(ann_path, "r") as f:
                self._ann_cache[key] = json.load(f)
        return self._ann_cache[key]

    def __getitem__(self, idx):
        date, drive, frame_num = self._frame_index[idx]
        frame_str = f"{frame_num:010d}"
        drive_dir = os.path.join(self.data_root, date, drive)

        image = self._read_image(drive_dir, frame_str)
        points = self._read_lidar(drive_dir, frame_str)
        pointcloud = self._points_to_bev(points)

        cuboids = self._annotations_for(date, drive)[frame_num]["cuboids"]
        targets = self._cuboids_to_targets(cuboids)
        m = torch.ones(1, self.config.bev_h, self.config.bev_w)

        return {
            "image": image,
            "pointcloud": pointcloud,
            "m": m,
            "targets": targets,
            "idx": self.sample_indices[idx],
        }

    def _read_image(self, drive_dir: str, frame_str: str) -> torch.Tensor:
        path = os.path.join(drive_dir, "labeled", f"image_0{self.camera_id}", "data", f"{frame_str}.png")
        img = Image.open(path).convert("RGB").resize((self.image_size, self.image_size))
        array = np.asarray(img, dtype=np.float32) / 255.0
        return torch.from_numpy(array).permute(2, 0, 1).contiguous()

    def _read_lidar(self, drive_dir: str, frame_str: str) -> np.ndarray:
        path = os.path.join(drive_dir, "labeled", "lidar_points", "data", f"{frame_str}.bin")
        return np.fromfile(path, dtype=np.float32).reshape(-1, 4)

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
        """Same 4-channel (occupancy, max height, mean intensity, log point density) scheme as CRAFXKittiDataset."""
        bev = np.zeros((4, self.config.bev_h, self.config.bev_w), dtype=np.float32)
        counts = np.zeros((self.config.bev_h, self.config.bev_w), dtype=np.float32)
        height_max = np.full((self.config.bev_h, self.config.bev_w), -np.inf, dtype=np.float32)
        intensity_sum = np.zeros((self.config.bev_h, self.config.bev_w), dtype=np.float32)

        z_min, z_max = self.z_range
        for x, y, z, intensity in points:
            cell = self._to_grid_indices(x, y)
            if cell is None or not (z_min <= z <= z_max):
                continue
            row, col = cell
            counts[row, col] += 1
            intensity_sum[row, col] += intensity
            height_max[row, col] = max(height_max[row, col], z)

        occupied = counts > 0
        bev[0][occupied] = 1.0
        normalized_height = np.zeros_like(height_max)
        normalized_height[occupied] = (height_max[occupied] - z_min) / (z_max - z_min)
        bev[1] = np.clip(normalized_height, 0.0, 1.0)
        bev[2][occupied] = intensity_sum[occupied] / counts[occupied]
        bev[3] = np.log1p(counts) / np.log1p(counts.max()) if counts.max() > 0 else counts

        return torch.from_numpy(bev)

    def _cuboids_to_targets(self, cuboids: List[Dict]) -> Dict[str, torch.Tensor]:
        num_classes = self.config.num_classes
        heatmap = torch.zeros(num_classes, self.config.bev_h, self.config.bev_w)
        regression = torch.zeros(6, self.config.bev_h, self.config.bev_w)
        velocity = torch.zeros(2, self.config.bev_h, self.config.bev_w)  # CADC labels carry no velocity GT

        z_min, z_max = self.z_range
        x_min, x_max = self.x_range
        y_min, y_max = self.y_range
        cell_x_size = (x_max - x_min) / self.config.bev_h
        cell_y_size = (y_max - y_min) / self.config.bev_w

        for obj in cuboids:
            label = obj.get("label")
            if label not in CADC_CLASSES or not (0 <= CADC_CLASSES[label] < num_classes):
                continue
            pos = obj["position"]
            x, y, z = pos["x"], pos["y"], pos["z"]
            cell = self._to_grid_indices(x, y)
            if cell is None:
                continue
            row, col = cell

            dx = ((x - x_min) % cell_x_size) / cell_x_size
            dy = ((y - y_min) % cell_y_size) / cell_y_size
            z_norm = float(np.clip((z - z_min) / (z_max - z_min), 0.0, 1.0))
            dims = obj["dimensions"]

            heatmap[CADC_CLASSES[label], row, col] = 1.0
            regression[:, row, col] = torch.tensor([dx, dy, z_norm, dims["x"], dims["y"], dims["z"]])

        return {"H": heatmap, "B": regression, "V": velocity}
