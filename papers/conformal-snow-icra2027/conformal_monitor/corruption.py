"""
Synthetic weather-onset corruption pipeline (plan.md "Dataset decision" /
"Next steps": "stand up the KITTI/nuScenes synthetic-corruption fallback
pipeline ... in parallel now, not as an afterthought"). Snowy Scenes access
is still pending, so this wraps any base dataset yielding craf_x-shaped
`{'image', 'pointcloud', ...}` samples (e.g.
`craf_x.datasets.nuscenes_mock.NuScenesMockDataset`, or the real
nuScenes/KITTI loaders once data is available) and simulates a scene that
transitions from clear weather into snowfall at a known frame, so that
detection-delay-vs-false-alarm evaluation has ground truth to score against.

Kept dataset-agnostic per plan.md ("Keep the method and evaluation protocol
dataset-agnostic ... so either dataset drops in without redesign") — swap in
real Snowy Scenes onset labels by replacing `WeatherOnsetStream` with a
thin adapter exposing the same `{'image', 'pointcloud', 'severity',
'is_onset_regime', 't'}` fields.
"""
from typing import Optional

import torch
from torch.utils.data import Dataset


def apply_camera_snow_corruption(image: torch.Tensor, severity: float, seed: Optional[int] = None) -> torch.Tensor:
    """
    Simulates snowfall's effect on camera images: reduced contrast/visibility
    (attenuation toward a flat gray) plus additive bright-speckle "snowflake"
    noise, both scaled by `severity` in [0, 1].
    """
    if severity <= 0.0:
        return image
    generator = torch.Generator().manual_seed(seed) if seed is not None else None

    attenuated = image * (1.0 - 0.5 * severity) + 0.5 * severity
    speckle = torch.randn(image.shape, generator=generator) * (0.3 * severity)
    bright_mask = (
        torch.rand(image.shape, generator=generator) < (0.05 * severity)
    ).float()
    corrupted = attenuated + speckle + bright_mask * severity
    return corrupted


def apply_lidar_snow_corruption(pointcloud: torch.Tensor, severity: float, seed: Optional[int] = None) -> torch.Tensor:
    """
    Simulates snowfall's effect on LiDAR returns: random point/pillar dropout
    (snow-induced beam attenuation and false near-range returns) plus
    additive range noise, both scaled by `severity` in [0, 1].
    """
    if severity <= 0.0:
        return pointcloud
    generator = torch.Generator().manual_seed(seed) if seed is not None else None

    keep_mask = (
        torch.rand(pointcloud.shape[-2:], generator=generator) >= (0.4 * severity)
    ).float()
    range_noise = torch.randn(pointcloud.shape, generator=generator) * (0.2 * severity)
    corrupted = (pointcloud + range_noise) * keep_mask.unsqueeze(0)
    return corrupted


class WeatherOnsetStream(Dataset):
    """
    Wraps a base dataset into a single ordered "scene" of `scene_length`
    frames that is clear (severity 0) until `onset_frame`, then ramps
    linearly to `severity_max` over `ramp_length` frames and holds there —
    mirroring the accumulating -> active snowfall -> highway snow
    progression described in the top-level README's Snowy Scenes summary,
    collapsed to a single scalar severity for a tractable synthetic proxy.

    Frames are drawn from the base dataset with replacement (via modular
    indexing) so `scene_length` can exceed `len(base_dataset)`.
    """

    def __init__(
        self,
        base_dataset,
        scene_length: int = 200,
        onset_frame: int = 100,
        ramp_length: int = 30,
        severity_max: float = 1.0,
        seed: Optional[int] = None,
    ):
        if not (0 <= onset_frame < scene_length):
            raise ValueError("onset_frame must be within [0, scene_length)")
        self.base_dataset = base_dataset
        self.scene_length = scene_length
        self.onset_frame = onset_frame
        self.ramp_length = max(ramp_length, 1)
        self.severity_max = severity_max
        self.seed = seed

    def __len__(self) -> int:
        return self.scene_length

    def severity_at(self, t: int) -> float:
        if t < self.onset_frame:
            return 0.0
        progress = min(1.0, (t - self.onset_frame) / self.ramp_length)
        return self.severity_max * progress

    def __getitem__(self, t: int) -> dict:
        if t < 0 or t >= self.scene_length:
            raise IndexError(t)
        base_idx = t % len(self.base_dataset)
        sample = dict(self.base_dataset[base_idx])

        severity = self.severity_at(t)
        frame_seed = None if self.seed is None else self.seed + t
        sample["image"] = apply_camera_snow_corruption(sample["image"], severity, frame_seed)
        sample["pointcloud"] = apply_lidar_snow_corruption(sample["pointcloud"], severity, frame_seed)
        sample["t"] = t
        sample["severity"] = severity
        sample["is_onset_regime"] = t >= self.onset_frame
        return sample
