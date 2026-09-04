import os

import numpy as np
import pytest
import torch
from PIL import Image

from craf_x.config import CRAFXConfig
from craf_x.datasets.kitti_dataset import CRAFXKittiDataset, KITTI_CLASSES

# Calibration chosen so the camera<->velodyne axis mapping is exact and
# hand-verifiable: with R0_rect = I and
#   Tr_velo_to_cam = [[0,-1,0,0],[0,0,-1,0],[1,0,0,0]]
# a velodyne point (vx, vy, vz) maps to camera point (cx, cy, cz) =
# (-vy, -vz, vx), and the inverse (camera -> velodyne) recovers
# (vx, vy, vz) = (cz, -cx, -cy).
CALIB_TEXT = (
    "R0_rect: 1 0 0 0 1 0 0 0 1\n"
    "Tr_velo_to_cam: 0 -1 0 0 0 0 -1 0 1 0 0 0\n"
)

# A Car at velodyne-frame (x=10, y=2, z=-1) -> camera-frame (x=-2, y=1, z=10).
VELO_LOCATION = (10.0, 2.0, -1.0)
CAM_LOCATION = (-2.0, 1.0, 10.0)
DIMENSIONS_HWL = (1.5, 1.6, 4.0)  # height, width, length


def _write_kitti_sample(root: str, sample_id: str = "000000"):
    split_dir = os.path.join(root, "training")
    for sub in ("image_2", "velodyne", "calib", "label_2"):
        os.makedirs(os.path.join(split_dir, sub), exist_ok=True)

    # Solid mid-gray image so the resized/normalized pixel mean is predictable.
    img = Image.new("RGB", (64, 64), color=(128, 128, 128))
    img.save(os.path.join(split_dir, "image_2", f"{sample_id}.png"))

    points = np.array(
        [
            [VELO_LOCATION[0], VELO_LOCATION[1], VELO_LOCATION[2], 0.5],
            [5.0, -10.0, 0.0, 0.2],
            [30.0, 20.0, -2.0, 0.8],
        ],
        dtype=np.float32,
    )
    points.tofile(os.path.join(split_dir, "velodyne", f"{sample_id}.bin"))

    with open(os.path.join(split_dir, "calib", f"{sample_id}.txt"), "w") as f:
        f.write(CALIB_TEXT)

    label_line = (
        f"Car 0.00 0 0.00 0.00 0.00 50.00 50.00 "
        f"{DIMENSIONS_HWL[0]:.2f} {DIMENSIONS_HWL[1]:.2f} {DIMENSIONS_HWL[2]:.2f} "
        f"{CAM_LOCATION[0]:.2f} {CAM_LOCATION[1]:.2f} {CAM_LOCATION[2]:.2f} 0.00\n"
    )
    with open(os.path.join(split_dir, "label_2", f"{sample_id}.txt"), "w") as f:
        f.write(label_line)


@pytest.fixture
def kitti_root(tmp_path):
    root = str(tmp_path / "kitti")
    _write_kitti_sample(root)
    return root


def _expected_cell(config: CRAFXConfig, x_range, y_range):
    row = int((VELO_LOCATION[0] - x_range[0]) / (x_range[1] - x_range[0]) * config.bev_h)
    col = int((VELO_LOCATION[1] - y_range[0]) / (y_range[1] - y_range[0]) * config.bev_w)
    return row, col


def test_dataset_detects_real_layout(kitti_root):
    dataset = CRAFXKittiDataset(data_root=kitti_root)
    assert dataset.is_real is True
    assert len(dataset) == 1


def test_falls_back_to_dummy_mode_when_layout_missing(tmp_path):
    dataset = CRAFXKittiDataset(data_root=str(tmp_path / "does_not_exist"))
    assert dataset.is_real is False
    assert len(dataset) == 3  # the mocked-index fallback


def test_real_image_is_actually_read(kitti_root):
    config = CRAFXConfig(bev_h=16, bev_w=16)
    dataset = CRAFXKittiDataset(data_root=kitti_root, config=config)
    sample = dataset[0]

    assert sample["image"].shape == (3, config.bev_h, config.bev_h)
    assert sample["image"].mean().item() == pytest.approx(128 / 255, abs=1e-3)


def test_real_pointcloud_bev_reflects_actual_points(kitti_root):
    config = CRAFXConfig(bev_h=16, bev_w=16)
    x_range, y_range = (0.0, 70.4), (-40.0, 40.0)
    dataset = CRAFXKittiDataset(data_root=kitti_root, config=config, x_range=x_range, y_range=y_range)
    sample = dataset[0]

    bev = sample["pointcloud"]
    assert bev.shape == (4, config.bev_h, config.bev_w)

    row, col = _expected_cell(config, x_range, y_range)
    assert bev[0, row, col] == 1.0  # occupancy at the known point's cell
    expected_height_norm = (VELO_LOCATION[2] - (-3.0)) / (1.0 - (-3.0))
    assert bev[1, row, col].item() == pytest.approx(expected_height_norm, abs=1e-4)

    # A cell far from any point should be empty.
    assert bev[0, 0, 0] == 0.0


def test_real_labels_project_to_correct_bev_cell_and_class(kitti_root):
    config = CRAFXConfig(bev_h=16, bev_w=16)
    x_range, y_range = (0.0, 70.4), (-40.0, 40.0)
    dataset = CRAFXKittiDataset(data_root=kitti_root, config=config, x_range=x_range, y_range=y_range)
    sample = dataset[0]

    targets = sample["targets"]
    assert targets["H"].shape == (len(KITTI_CLASSES), config.bev_h, config.bev_w)
    assert targets["B"].shape == (6, config.bev_h, config.bev_w)
    assert targets["V"].shape == (2, config.bev_h, config.bev_w)
    assert torch.count_nonzero(targets["V"]) == 0  # no velocity GT in KITTI

    row, col = _expected_cell(config, x_range, y_range)
    assert targets["H"][KITTI_CLASSES["Car"], row, col] == 1.0
    # No heatmap activation for the other classes anywhere.
    for cls, cls_idx in KITTI_CLASSES.items():
        if cls != "Car":
            assert torch.count_nonzero(targets["H"][cls_idx]) == 0

    height, width, length = DIMENSIONS_HWL
    regression = targets["B"][:, row, col]
    assert regression[3].item() == pytest.approx(width)
    assert regression[4].item() == pytest.approx(length)
    assert regression[5].item() == pytest.approx(height)


def test_match_mask_is_all_ones_for_real_data(kitti_root):
    dataset = CRAFXKittiDataset(data_root=kitti_root)
    sample = dataset[0]
    assert torch.all(sample["m"] == 1.0)
