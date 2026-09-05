import json
import os

import numpy as np
import pytest
import torch
from PIL import Image
from torch.utils.data import DataLoader

from craf_x.config import CRAFXConfig
from craf_x.datasets.cadc_dataset import CADC_NUM_CLASSES, CRAFXCADCDataset

# A "Car" at LiDAR-frame (x=20, y=-5, z=-1), dims (4.5, 2.0, 1.6).
OBJECT_LOCATION = (20.0, -5.0, -1.0)
OBJECT_DIMS = (4.5, 2.0, 1.6)
OBJECT_LABEL = "Car"


def _write_frame(drive_dir: str, frame_num: int, image_size=(32, 32), color=(64, 64, 64)):
    frame_str = f"{frame_num:010d}"
    img_dir = os.path.join(drive_dir, "labeled", "image_00", "data")
    lidar_dir = os.path.join(drive_dir, "labeled", "lidar_points", "data")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(lidar_dir, exist_ok=True)

    Image.new("RGB", image_size, color=color).save(os.path.join(img_dir, f"{frame_str}.png"))
    points = np.array(
        [
            [OBJECT_LOCATION[0], OBJECT_LOCATION[1], OBJECT_LOCATION[2], 0.6],
            [1.0, 1.0, 0.0, 0.1],
        ],
        dtype=np.float32,
    )
    points.tofile(os.path.join(lidar_dir, f"{frame_str}.bin"))


def _build_cadc_tree(root: str, date: str, drive: str, frames_cuboids):
    """`frames_cuboids`: list of cuboid-lists, one per frame (frame index = list index)."""
    drive_dir = os.path.join(root, date, drive)
    for frame_num, cuboids in enumerate(frames_cuboids):
        _write_frame(drive_dir, frame_num)
    with open(os.path.join(drive_dir, "3d_ann.json"), "w") as f:
        json.dump([{"cuboids": cuboids} for cuboids in frames_cuboids], f)


def _car_cuboid():
    return {
        "label": OBJECT_LABEL,
        "position": {"x": OBJECT_LOCATION[0], "y": OBJECT_LOCATION[1], "z": OBJECT_LOCATION[2]},
        "dimensions": {"x": OBJECT_DIMS[0], "y": OBJECT_DIMS[1], "z": OBJECT_DIMS[2]},
        "yaw": 0.0,
        "stationary": False,
        "camera_used": 0,
        "attributes": {},
        "points_count": 11,
    }


@pytest.fixture
def cadc_root(tmp_path):
    root = str(tmp_path / "cadcd")
    # 2018_03_06 -> "bare" road category (2 frames: one with a Car, one empty)
    _build_cadc_tree(root, "2018_03_06", "0001", [[_car_cuboid()], []])
    # 2019_02_27 -> "covered" road category (1 frame with a Car)
    _build_cadc_tree(root, "2019_02_27", "0002", [[_car_cuboid()]])
    return root


def _expected_cell(config, x_range, y_range):
    row = int((OBJECT_LOCATION[0] - x_range[0]) / (x_range[1] - x_range[0]) * config.bev_h)
    col = int((OBJECT_LOCATION[1] - y_range[0]) / (y_range[1] - y_range[0]) * config.bev_w)
    return row, col


def test_dataset_indexes_all_drives_and_frames(cadc_root):
    dataset = CRAFXCADCDataset(data_root=cadc_root)
    assert len(dataset) == 3  # 2 frames from 0001 + 1 frame from 0002


def test_sample_indices_carry_correct_category_prefix(cadc_root):
    dataset = CRAFXCADCDataset(data_root=cadc_root)
    bare_ids = [s for s in dataset.sample_indices if s.startswith("bare_")]
    covered_ids = [s for s in dataset.sample_indices if s.startswith("covered_")]
    assert len(bare_ids) == 2
    assert len(covered_ids) == 1
    assert bare_ids[0] == "bare_2018_03_06_0001_0000000000"
    assert covered_ids[0] == "covered_2019_02_27_0002_0000000000"


def test_category_indices_from_real_snow_stream_module_works_unchanged(cadc_root):
    # Regression/integration test: conformal_monitor.real_snow_stream's
    # category_indices/category_subset (written for Snowy Scenes'
    # "{category}_{frame}" convention) must work against CRAFXCADCDataset
    # completely unchanged, since that's the whole point of the
    # "{category}_{date}_{drive}_{frame}" sample_indices format.
    import sys

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "papers", "conformal-snow-icra2027"))
    from conformal_monitor.real_snow_stream import category_indices

    dataset = CRAFXCADCDataset(data_root=cadc_root)
    bare_idx = category_indices(dataset, "bare")
    covered_idx = category_indices(dataset, "covered")
    assert len(bare_idx) == 2
    assert len(covered_idx) == 1


def test_image_is_read_correctly(cadc_root):
    config = CRAFXConfig(bev_h=16, bev_w=16, num_classes=CADC_NUM_CLASSES)
    dataset = CRAFXCADCDataset(data_root=cadc_root, config=config)
    sample = dataset[0]
    assert sample["image"].shape == (3, config.bev_h, config.bev_h)
    assert sample["image"].mean().item() == pytest.approx(64 / 255, abs=1e-3)


def test_pointcloud_and_labels_project_to_same_known_cell(cadc_root):
    config = CRAFXConfig(bev_h=16, bev_w=16, num_classes=CADC_NUM_CLASSES)
    x_range, y_range, z_range = (0.0, 100.0), (-50.0, 50.0), (-3.0, 3.0)
    dataset = CRAFXCADCDataset(
        data_root=cadc_root, config=config, x_range=x_range, y_range=y_range, z_range=z_range
    )
    idx = dataset.sample_indices.index("bare_2018_03_06_0001_0000000000")
    sample = dataset[idx]

    row, col = _expected_cell(config, x_range, y_range)

    bev = sample["pointcloud"]
    assert bev.shape == (4, config.bev_h, config.bev_w)
    assert bev[0, row, col] == 1.0
    expected_height_norm = (OBJECT_LOCATION[2] - z_range[0]) / (z_range[1] - z_range[0])
    assert bev[1, row, col].item() == pytest.approx(expected_height_norm, abs=1e-4)

    targets = sample["targets"]
    assert targets["H"].shape == (CADC_NUM_CLASSES, config.bev_h, config.bev_w)
    assert targets["H"][0, row, col] == 1.0  # Car -> class index 0
    assert torch.count_nonzero(targets["H"]) == 1

    regression = targets["B"][:, row, col]
    assert regression[3].item() == pytest.approx(OBJECT_DIMS[0])
    assert regression[4].item() == pytest.approx(OBJECT_DIMS[1])
    assert regression[5].item() == pytest.approx(OBJECT_DIMS[2])

    assert torch.count_nonzero(targets["V"]) == 0  # CADC labels carry no velocity GT


def test_frame_with_no_cuboids_gives_empty_heatmap(cadc_root):
    config = CRAFXConfig(bev_h=16, bev_w=16, num_classes=CADC_NUM_CLASSES)
    dataset = CRAFXCADCDataset(data_root=cadc_root, config=config)
    idx = dataset.sample_indices.index("bare_2018_03_06_0001_0000000001")
    sample = dataset[idx]
    assert torch.count_nonzero(sample["targets"]["H"]) == 0


def test_unrecognized_label_is_dropped_not_crashed(cadc_root):
    # A label outside CADC_CLASSES (e.g. "Horse_and_Buggy", genuinely rare
    # in the real dataset) must be silently skipped, not raise.
    drive_dir = os.path.join(cadc_root, "2018_03_06", "0003")
    _write_frame(drive_dir, 0)
    with open(os.path.join(drive_dir, "3d_ann.json"), "w") as f:
        json.dump([{"cuboids": [{**_car_cuboid(), "label": "Horse_and_Buggy"}]}], f)

    config = CRAFXConfig(bev_h=16, bev_w=16, num_classes=CADC_NUM_CLASSES)
    dataset = CRAFXCADCDataset(data_root=cadc_root, config=config)
    idx = dataset.sample_indices.index("bare_2018_03_06_0003_0000000000")
    sample = dataset[idx]
    assert torch.count_nonzero(sample["targets"]["H"]) == 0


def test_match_mask_is_all_ones(cadc_root):
    dataset = CRAFXCADCDataset(data_root=cadc_root)
    sample = dataset[0]
    assert torch.all(sample["m"] == 1.0)


def test_dataloader_batching(cadc_root):
    config = CRAFXConfig(bev_h=16, bev_w=16, num_classes=CADC_NUM_CLASSES)
    dataset = CRAFXCADCDataset(data_root=cadc_root, config=config)
    loader = DataLoader(dataset, batch_size=3, shuffle=False)
    batch = next(iter(loader))
    assert batch["image"].shape == (3, 3, 16, 16)
    assert batch["targets"]["H"].shape == (3, CADC_NUM_CLASSES, 16, 16)


def test_missing_data_root_warns_and_is_empty(tmp_path):
    with pytest.warns(UserWarning):
        dataset = CRAFXCADCDataset(data_root=str(tmp_path / "does_not_exist"))
    assert len(dataset) == 0
