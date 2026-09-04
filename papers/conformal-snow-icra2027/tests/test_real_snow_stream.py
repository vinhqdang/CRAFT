import io
import zipfile

import numpy as np
import pytest
from PIL import Image

from craf_x.config import CRAFXConfig
from craf_x.datasets.snowy_scenes_dataset import CRAFXSnowyScenesDataset, SNOWY_SCENES_NUM_CLASSES
from conformal_monitor.real_snow_stream import RealSnowOnsetStream, category_indices, category_subset

LABEL_MAP_YAML = """object_labels:
0: "unlabeled"
2: "car"
"""


def _write_frame(zf, split, frame_id):
    img = Image.new("RGB", (16, 16), color=(10, 10, 10))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    zf.writestr(f"ROADVIEW5k/{split}/images/{frame_id}.png", buf.getvalue())

    points = np.array([[5.0, 0.0, 0.0, 0.3]], dtype=np.float32)
    zf.writestr(f"ROADVIEW5k/{split}/velodyne/{frame_id}.bin", points.tobytes())
    zf.writestr(f"ROADVIEW5k/{split}/object_labels/{frame_id}.txt", "2 5.0 0.0 0.0 4.0 2.0 1.5 0.0\n")


@pytest.fixture
def two_category_zip(tmp_path):
    path = str(tmp_path / "ROADVIEW5k.zip")
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("ROADVIEW5k/label_map.yaml", LABEL_MAP_YAML)
        for i in range(4):
            _write_frame(zf, "train", f"accumulated_{i:06d}")
        for i in range(3):
            _write_frame(zf, "train", f"falling_{i:06d}")
    return path


def test_category_indices_and_subset_filter_correctly(two_category_zip):
    dataset = CRAFXSnowyScenesDataset(zip_path=two_category_zip, split="train")
    accum_idx = category_indices(dataset, "accumulated")
    falling_idx = category_indices(dataset, "falling")

    assert len(accum_idx) == 4
    assert len(falling_idx) == 3
    assert set(accum_idx).isdisjoint(falling_idx)
    assert all(dataset.sample_indices[i].startswith("accumulated_") for i in accum_idx)

    accum_subset = category_subset(dataset, "accumulated")
    assert len(accum_subset) == 4


def test_real_onset_stream_switches_at_onset_frame(two_category_zip):
    dataset = CRAFXSnowyScenesDataset(zip_path=two_category_zip, split="train")
    nominal = category_subset(dataset, "accumulated")
    degraded = category_subset(dataset, "falling")

    stream = RealSnowOnsetStream(nominal, degraded, onset_frame=3, scene_length=6)
    assert len(stream) == 6
    assert stream.onset_frame == 3

    for t in range(3):
        assert stream[t]["idx"].startswith("accumulated_")
    for t in range(3, 6):
        assert stream[t]["idx"].startswith("falling_")


def test_real_onset_stream_wraps_with_replacement(two_category_zip):
    dataset = CRAFXSnowyScenesDataset(zip_path=two_category_zip, split="train")
    nominal = category_subset(dataset, "accumulated")  # length 4
    degraded = category_subset(dataset, "falling")  # length 3

    stream = RealSnowOnsetStream(nominal, degraded, onset_frame=2, scene_length=10)
    # degraded portion is indices [2, 10), length 8, wrapping a length-3 dataset
    assert stream[2]["idx"] == stream[5]["idx"]  # (5-2)%3 == (2-2)%3 == 0


def test_real_onset_stream_rejects_invalid_construction(two_category_zip):
    dataset = CRAFXSnowyScenesDataset(zip_path=two_category_zip, split="train")
    nominal = category_subset(dataset, "accumulated")
    degraded = category_subset(dataset, "falling")

    with pytest.raises(ValueError):
        RealSnowOnsetStream(nominal, degraded, onset_frame=6, scene_length=6)  # onset_frame must be < scene_length

    empty = category_subset(dataset, "highway")  # no highway frames in this fixture
    with pytest.raises(ValueError):
        RealSnowOnsetStream(empty, degraded, onset_frame=1, scene_length=3)


def test_real_onset_stream_rejects_out_of_range_index(two_category_zip):
    dataset = CRAFXSnowyScenesDataset(zip_path=two_category_zip, split="train")
    nominal = category_subset(dataset, "accumulated")
    degraded = category_subset(dataset, "falling")
    stream = RealSnowOnsetStream(nominal, degraded, onset_frame=2, scene_length=5)
    with pytest.raises(IndexError):
        stream[5]


def test_real_frames_have_correct_shapes(two_category_zip):
    config = CRAFXConfig(bev_h=8, bev_w=8, num_classes=SNOWY_SCENES_NUM_CLASSES)
    dataset = CRAFXSnowyScenesDataset(zip_path=two_category_zip, split="train", config=config)
    nominal = category_subset(dataset, "accumulated")
    degraded = category_subset(dataset, "falling")
    stream = RealSnowOnsetStream(nominal, degraded, onset_frame=2, scene_length=4)

    sample = stream[0]
    assert sample["image"].shape == (3, config.bev_h, config.bev_h)
    assert sample["pointcloud"].shape == (4, config.bev_h, config.bev_w)
    assert sample["targets"]["H"].shape == (SNOWY_SCENES_NUM_CLASSES, config.bev_h, config.bev_w)
