import io
import re
import zipfile
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from ..config import CRAFXConfig

_LABEL_MAP_LINE = re.compile(r'^\s*(\d+)\s*:\s*"([^"]*)"')

# label_map.yaml currently defines ids 0-28 (SemanticKITTI-style classes plus
# a "snow" class); object_labels reuses this same id space. Callers that
# need num_classes before constructing the dataset (e.g. to build the
# CRAFXConfig passed into it) can use this constant; the dataset also parses
# the archive's actual label_map.yaml into `self.classes` for verification.
SNOWY_SCENES_NUM_CLASSES = 29


def _parse_label_map(yaml_text: str) -> Dict[int, str]:
    """
    Tiny hand-rolled parser for the dataset's `label_map.yaml`, which is a
    flat `<int>: "<name>"` list under an `object_labels:` key -- avoids
    pulling in a YAML dependency for a one-line-per-entry format.
    """
    classes = {}
    for line in yaml_text.splitlines():
        m = _LABEL_MAP_LINE.match(line)
        if m:
            classes[int(m.group(1))] = m.group(2)
    return classes


class CRAFXSnowyScenesDataset(Dataset):
    """
    Wrapper for the Snowy Scenes multimodal dataset (Ngo, Raisuddin, et al.,
    "Snowy Scenes: A Multimodal Multitask Dataset Toward Snow-Tonomous
    Vehicles"; distributed as a single `ROADVIEW5k.zip` archive).

    Reads lazily straight out of the zip archive via `zipfile` -- the
    archive is ~49GB compressed / ~93GB uncompressed, so it is read
    in-place rather than extracted to disk.

    Archive layout (per split in {'train', 'val', 'test'}):
        ROADVIEW5k/<split>/images/<frame_id>.png          -- RGB camera
        ROADVIEW5k/<split>/velodyne/<frame_id>.bin          -- (N,4) float32 [x,y,z,intensity], LiDAR frame
        ROADVIEW5k/<split>/object_labels/<frame_id>.txt     -- one 3D box per line:
                                                                `class_id x y z d1 d2 d3 yaw`,
                                                                already in the LiDAR frame (no
                                                                camera<->LiDAR transform needed, unlike KITTI)
        ROADVIEW5k/<split>/labels/<frame_id>.bin            -- per-point semantic segmentation
                                                                (1 byte/point, aligned with velodyne) --
                                                                NOT used here: craf_x's head has no
                                                                segmentation output.
        ROADVIEW5k/label_map.yaml                           -- object_labels class_id -> name

    Reuses the same BEV-voxelization scheme as `CRAFXKittiDataset` for the
    point cloud, and the same target convention: heatmap peak + (dx, dy, z,
    d1, d2, d3) regression at the object's cell (no yaw channel -- the head
    doesn't predict one), zero velocity (the format carries none), and an
    all-ones CCP mask (real, synchronized, non-adversarial data).
    """

    def __init__(
        self,
        zip_path: str,
        split: str = "train",
        config: Optional[CRAFXConfig] = None,
        image_size: Optional[int] = None,
        x_range: Tuple[float, float] = (0.0, 100.0),
        y_range: Tuple[float, float] = (-50.0, 50.0),
        z_range: Tuple[float, float] = (-3.0, 3.0),
    ):
        self.zip_path = zip_path
        self.split = split
        self.config = config or CRAFXConfig()
        self.image_size = image_size or self.config.bev_h
        self.x_range = x_range
        self.y_range = y_range
        self.z_range = z_range

        self._zf: Optional[zipfile.ZipFile] = None  # opened lazily per-process, see _zip()
        self._root = "ROADVIEW5k"
        self._split_dir = f"{self._root}/{split}"

        # Use a throwaway handle for this one-time listing rather than
        # self._zip(), so self._zf stays None after construction. DataLoader
        # workers on Linux are forked (not spawned/pickled), so they inherit
        # process memory directly -- if self._zf were already open here, every
        # worker would inherit the *same* underlying file descriptor and
        # race on its shared read offset, corrupting reads (BadZipFile).
        # Leaving it None means each process's own first _zip() call (from
        # __getitem__) opens an independent handle.
        with zipfile.ZipFile(zip_path) as zf:
            names = zf.namelist()
            label_map_path = f"{self._root}/label_map.yaml"
            self.classes = _parse_label_map(zf.read(label_map_path).decode("utf-8")) if label_map_path in names else {}

        prefix = f"{self._split_dir}/velodyne/"
        self.sample_indices = sorted(
            n[len(prefix):-len(".bin")] for n in names if n.startswith(prefix) and n.endswith(".bin")
        )

    def _zip(self) -> zipfile.ZipFile:
        # DataLoader workers each need their own ZipFile handle (a
        # zipfile.ZipFile is not safely shared across processes).
        if self._zf is None:
            self._zf = zipfile.ZipFile(self.zip_path)
        return self._zf

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_zf"] = None  # don't pickle an open file handle across worker processes
        return state

    def __len__(self):
        return len(self.sample_indices)

    def __getitem__(self, idx):
        frame_id = self.sample_indices[idx]

        image = self._read_image(frame_id)
        points = self._read_velodyne(frame_id)
        pointcloud = self._points_to_bev(points)
        labels = self._read_object_labels(frame_id)
        targets = self._labels_to_targets(labels)
        m = torch.ones(1, self.config.bev_h, self.config.bev_w)

        return {
            "image": image,
            "pointcloud": pointcloud,
            "m": m,
            "targets": targets,
            "idx": frame_id,
        }

    def _read_image(self, frame_id: str) -> torch.Tensor:
        data = self._zip().read(f"{self._split_dir}/images/{frame_id}.png")
        img = Image.open(io.BytesIO(data)).convert("RGB").resize((self.image_size, self.image_size))
        array = np.asarray(img, dtype=np.float32) / 255.0
        return torch.from_numpy(array).permute(2, 0, 1).contiguous()

    def _read_velodyne(self, frame_id: str) -> np.ndarray:
        data = self._zip().read(f"{self._split_dir}/velodyne/{frame_id}.bin")
        return np.frombuffer(data, dtype=np.float32).reshape(-1, 4)

    def _read_object_labels(self, frame_id: str) -> List[Dict]:
        path = f"{self._split_dir}/object_labels/{frame_id}.txt"
        try:
            text = self._zip().read(path).decode("utf-8")
        except KeyError:
            return []

        objects = []
        for line in text.splitlines():
            fields = line.split()
            if len(fields) < 8:
                continue
            class_id = int(fields[0])
            x, y, z, d1, d2, d3, yaw = (float(v) for v in fields[1:8])
            objects.append({"class_id": class_id, "location": (x, y, z), "dimensions": (d1, d2, d3), "yaw": yaw})
        return objects

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

    def _labels_to_targets(self, labels: List[Dict]) -> Dict[str, torch.Tensor]:
        num_classes = self.config.num_classes
        heatmap = torch.zeros(num_classes, self.config.bev_h, self.config.bev_w)
        regression = torch.zeros(6, self.config.bev_h, self.config.bev_w)
        velocity = torch.zeros(2, self.config.bev_h, self.config.bev_w)  # format carries no velocity

        z_min, z_max = self.z_range
        x_min, x_max = self.x_range
        y_min, y_max = self.y_range
        cell_x_size = (x_max - x_min) / self.config.bev_h
        cell_y_size = (y_max - y_min) / self.config.bev_w

        for obj in labels:
            x, y, z = obj["location"]
            cell = self._to_grid_indices(x, y)
            if cell is None or not (0 <= obj["class_id"] < num_classes):
                continue
            row, col = cell

            dx = ((x - x_min) % cell_x_size) / cell_x_size
            dy = ((y - y_min) % cell_y_size) / cell_y_size
            z_norm = float(np.clip((z - z_min) / (z_max - z_min), 0.0, 1.0))
            d1, d2, d3 = obj["dimensions"]

            heatmap[obj["class_id"], row, col] = 1.0
            regression[:, row, col] = torch.tensor([dx, dy, z_norm, d1, d2, d3])

        return {"H": heatmap, "B": regression, "V": velocity}
