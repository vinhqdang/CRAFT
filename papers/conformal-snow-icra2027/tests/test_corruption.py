import torch
import pytest
from torch.utils.data import Dataset

from craf_x.datasets.nuscenes_mock import NuScenesMockDataset
from conformal_monitor.corruption import (
    WeatherOnsetStream,
    apply_camera_snow_corruption,
    apply_lidar_snow_corruption,
)


class _DeterministicMockDataset(Dataset):
    """
    Fixed-content stand-in for NuScenesMockDataset, which draws fresh
    torch.randn tensors on every __getitem__ call (no seeding) and so can't
    be used where a test needs the *same* base sample twice.
    """

    def __init__(self, num_samples: int = 5):
        self.num_samples = num_samples
        self._image = torch.randn(3, 8, 8)
        self._pointcloud = torch.randn(4, 8, 8)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return {
            "image": self._image.clone(),
            "pointcloud": self._pointcloud.clone(),
            "targets": {
                "H": torch.zeros(10, 8, 8),
                "B": torch.zeros(6, 8, 8),
                "V": torch.zeros(2, 8, 8),
            },
        }


def test_severity_ramp_shape():
    stream = WeatherOnsetStream(
        NuScenesMockDataset(num_samples=5), scene_length=30,
        onset_frame=10, ramp_length=5, severity_max=1.0,
    )
    assert stream.severity_at(9) == 0.0
    assert stream.severity_at(10) == 0.0
    assert stream.severity_at(12) == pytest.approx(0.4)
    assert stream.severity_at(15) == pytest.approx(1.0)
    assert stream.severity_at(29) == 1.0  # capped at severity_max


def test_corruption_is_noop_at_zero_severity():
    image = torch.randn(3, 8, 8)
    pointcloud = torch.randn(4, 8, 8)
    assert torch.equal(apply_camera_snow_corruption(image, 0.0), image)
    assert torch.equal(apply_lidar_snow_corruption(pointcloud, 0.0), pointcloud)


def test_corruption_changes_tensor_at_positive_severity():
    image = torch.randn(3, 8, 8)
    corrupted = apply_camera_snow_corruption(image, severity=0.8, seed=0)
    assert not torch.equal(corrupted, image)


def test_stream_pre_onset_frame_is_uncorrupted():
    base = _DeterministicMockDataset(num_samples=5)
    stream = WeatherOnsetStream(base, scene_length=10, onset_frame=6, ramp_length=2)

    sample = stream[0]
    assert sample["severity"] == 0.0
    assert sample["is_onset_regime"] is False
    assert torch.equal(sample["image"], base[0]["image"])
    assert torch.equal(sample["pointcloud"], base[0]["pointcloud"])


def test_stream_post_onset_frame_is_corrupted_and_labeled():
    base = NuScenesMockDataset(num_samples=5)
    stream = WeatherOnsetStream(base, scene_length=10, onset_frame=6, ramp_length=2)

    sample = stream[9]  # well past the ramp, severity == severity_max
    assert sample["severity"] > 0.0
    assert sample["is_onset_regime"] is True
    assert not torch.equal(sample["image"], base[9 % len(base)]["image"])


def test_stream_wraps_base_dataset_with_replacement():
    base = NuScenesMockDataset(num_samples=3)
    stream = WeatherOnsetStream(base, scene_length=7, onset_frame=6, ramp_length=1)
    assert len(stream) == 7
    # index 5 maps to base index 2, index 6 (out of base range) wraps to base index 0
    _ = stream[5]
    _ = stream[6]


def test_stream_rejects_out_of_range_index():
    stream = WeatherOnsetStream(NuScenesMockDataset(num_samples=3), scene_length=5, onset_frame=2, ramp_length=1)
    with pytest.raises(IndexError):
        stream[5]


def test_stream_rejects_invalid_onset_frame():
    with pytest.raises(ValueError):
        WeatherOnsetStream(NuScenesMockDataset(num_samples=3), scene_length=5, onset_frame=5, ramp_length=1)


def test_stream_is_reproducible_with_seed():
    base = _DeterministicMockDataset(num_samples=3)
    stream_a = WeatherOnsetStream(base, scene_length=8, onset_frame=3, ramp_length=2, seed=42)
    stream_b = WeatherOnsetStream(base, scene_length=8, onset_frame=3, ramp_length=2, seed=42)

    sample_a = stream_a[6]
    sample_b = stream_b[6]
    assert torch.equal(sample_a["image"], sample_b["image"])
    assert torch.equal(sample_a["pointcloud"], sample_b["pointcloud"])
