import io
import pickle
import zipfile

import numpy as np
import pytest
import torch
from PIL import Image

from craf_x.config import CRAFXConfig
from craf_x.datasets.snowy_scenes_dataset import CRAFXSnowyScenesDataset, SNOWY_SCENES_NUM_CLASSES

LABEL_MAP_YAML = """object_labels:
0: "unlabeled"
2: "car"
7: "person"
"""

# A "car" (class 2) at LiDAR-frame (x=20, y=-5, z=-1), dims (4.5, 2.0, 1.6), yaw unused.
OBJECT_LOCATION = (20.0, -5.0, -1.0)
OBJECT_DIMS = (4.5, 2.0, 1.6)
OBJECT_CLASS = 2


def _build_snowy_scenes_zip(path: str, frame_id: str = "falling_000001"):
    points = np.array(
        [
            [OBJECT_LOCATION[0], OBJECT_LOCATION[1], OBJECT_LOCATION[2], 0.6],
            [1.0, 1.0, 0.0, 0.1],
        ],
        dtype=np.float32,
    )

    img = Image.new("RGB", (32, 32), color=(64, 64, 64))
    img_bytes = io.BytesIO()
    img.save(img_bytes, format="PNG")

    label_line = (
        f"{OBJECT_CLASS} {OBJECT_LOCATION[0]} {OBJECT_LOCATION[1]} {OBJECT_LOCATION[2]} "
        f"{OBJECT_DIMS[0]} {OBJECT_DIMS[1]} {OBJECT_DIMS[2]} 0.0\n"
    )

    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("ROADVIEW5k/label_map.yaml", LABEL_MAP_YAML)
        zf.writestr(f"ROADVIEW5k/train/images/{frame_id}.png", img_bytes.getvalue())
        zf.writestr(f"ROADVIEW5k/train/velodyne/{frame_id}.bin", points.tobytes())
        zf.writestr(f"ROADVIEW5k/train/object_labels/{frame_id}.txt", label_line)
        # A second frame with no objects, to exercise the empty-label path.
        zf.writestr("ROADVIEW5k/train/images/falling_000002.png", img_bytes.getvalue())
        zf.writestr("ROADVIEW5k/train/velodyne/falling_000002.bin", points.tobytes())


@pytest.fixture
def snowy_zip(tmp_path):
    path = str(tmp_path / "ROADVIEW5k.zip")
    _build_snowy_scenes_zip(path)
    return path


def _expected_cell(config, x_range, y_range):
    row = int((OBJECT_LOCATION[0] - x_range[0]) / (x_range[1] - x_range[0]) * config.bev_h)
    col = int((OBJECT_LOCATION[1] - y_range[0]) / (y_range[1] - y_range[0]) * config.bev_w)
    return row, col


def test_dataset_lists_frames_and_parses_label_map(snowy_zip):
    dataset = CRAFXSnowyScenesDataset(zip_path=snowy_zip, split="train")
    assert len(dataset) == 2
    assert dataset.classes == {0: "unlabeled", 2: "car", 7: "person"}


def test_image_is_read_correctly(snowy_zip):
    config = CRAFXConfig(bev_h=16, bev_w=16, num_classes=SNOWY_SCENES_NUM_CLASSES)
    dataset = CRAFXSnowyScenesDataset(zip_path=snowy_zip, split="train", config=config)
    sample = dataset[0]
    assert sample["image"].shape == (3, config.bev_h, config.bev_h)
    assert sample["image"].mean().item() == pytest.approx(64 / 255, abs=1e-3)


def test_pointcloud_and_labels_project_to_same_known_cell(snowy_zip):
    config = CRAFXConfig(bev_h=16, bev_w=16, num_classes=SNOWY_SCENES_NUM_CLASSES)
    x_range, y_range, z_range = (0.0, 100.0), (-50.0, 50.0), (-3.0, 3.0)
    dataset = CRAFXSnowyScenesDataset(
        zip_path=snowy_zip, split="train", config=config, x_range=x_range, y_range=y_range, z_range=z_range
    )
    sample = dataset[dataset.sample_indices.index("falling_000001")]

    row, col = _expected_cell(config, x_range, y_range)

    bev = sample["pointcloud"]
    assert bev.shape == (4, config.bev_h, config.bev_w)
    assert bev[0, row, col] == 1.0
    expected_height_norm = (OBJECT_LOCATION[2] - z_range[0]) / (z_range[1] - z_range[0])
    assert bev[1, row, col].item() == pytest.approx(expected_height_norm, abs=1e-4)

    targets = sample["targets"]
    assert targets["H"].shape == (SNOWY_SCENES_NUM_CLASSES, config.bev_h, config.bev_w)
    assert targets["H"][OBJECT_CLASS, row, col] == 1.0
    assert torch.count_nonzero(targets["H"]) == 1  # exactly one activation anywhere

    regression = targets["B"][:, row, col]
    assert regression[3].item() == pytest.approx(OBJECT_DIMS[0])
    assert regression[4].item() == pytest.approx(OBJECT_DIMS[1])
    assert regression[5].item() == pytest.approx(OBJECT_DIMS[2])

    assert torch.count_nonzero(targets["V"]) == 0  # no velocity in this format


def test_frame_with_no_object_labels_gives_empty_heatmap(snowy_zip):
    config = CRAFXConfig(bev_h=16, bev_w=16, num_classes=SNOWY_SCENES_NUM_CLASSES)
    dataset = CRAFXSnowyScenesDataset(zip_path=snowy_zip, split="train", config=config)
    sample = dataset[dataset.sample_indices.index("falling_000002")]
    assert torch.count_nonzero(sample["targets"]["H"]) == 0


def test_match_mask_is_all_ones(snowy_zip):
    dataset = CRAFXSnowyScenesDataset(zip_path=snowy_zip, split="train")
    sample = dataset[0]
    assert torch.all(sample["m"] == 1.0)


def test_dataset_is_picklable_for_dataloader_workers(snowy_zip):
    dataset = CRAFXSnowyScenesDataset(zip_path=snowy_zip, split="train")
    dataset[0]  # force the lazy zip handle open
    restored = pickle.loads(pickle.dumps(dataset))
    assert restored._zf is None
    assert len(restored) == len(dataset)
    assert restored[0]["idx"] == dataset[0]["idx"]
