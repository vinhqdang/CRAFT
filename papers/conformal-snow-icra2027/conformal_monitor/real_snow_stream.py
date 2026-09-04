"""
Real-data onset stream over Snowy Scenes itself (plan.md: "swap in real
Snowy Scenes results if/when access arrives" -- access has arrived; see
papers/conformal-snow-icra2027/README.md for how it was obtained).

Snowy Scenes turned out not to contain a clear-weather baseline split --
its three splits (`accumulated`, `falling`, `highway`) are all already
snowy driving. Measuring the real per-point "snow" semantic-segmentation
class fraction (ground truth, from `labels/*.bin`) across ~40 sampled
frames per category gave a clear, physically sensible severity ordering:

    accumulated: mean 0.0001   (settled snow on surfaces, not airborne)
    highway:     mean 0.0041
    falling:     mean 0.0119   (active/airborne snowfall -> spurious near-range LiDAR returns)

So rather than the originally planned KITTI/nuScenes synthetic-corruption
fallback (still in `corruption.py`) or a cross-dataset splice against real
KITTI data (which this environment doesn't actually have downloaded),
`RealSnowOnsetStream` constructs the onset transition entirely within
Snowy Scenes: a prefix of held-out `accumulated` frames (the nominal/H0
regime) followed by `falling` frames (the measured-most-severe regime) at
a known onset index. Every frame on both sides of the transition is real
sensor data; only the splice point itself is constructed, since Snowy
Scenes' own sequences don't contain a within-sequence onset transition
(see plan.md's "Snowy Scenes scoping notes" section).
"""
from typing import List

from torch.utils.data import Dataset, Subset

from craf_x.datasets.snowy_scenes_dataset import CRAFXSnowyScenesDataset

# Ordered mild -> severe by measured mean per-point "snow" class fraction (see module docstring).
CATEGORY_SEVERITY_ORDER = ["accumulated", "highway", "falling"]


def category_indices(dataset: CRAFXSnowyScenesDataset, category: str) -> List[int]:
    """Indices into `dataset` whose frame id belongs to the given weather category."""
    prefix = f"{category}_"
    return [i for i, frame_id in enumerate(dataset.sample_indices) if frame_id.startswith(prefix)]


def category_subset(dataset: CRAFXSnowyScenesDataset, category: str) -> Subset:
    """A `torch.utils.data.Subset` of `dataset` restricted to one weather category."""
    return Subset(dataset, category_indices(dataset, category))


class RealSnowOnsetStream(Dataset):
    """
    Splices a nominal-regime dataset (frames before `onset_frame`) with a
    degraded-regime dataset (frames from `onset_frame` on) into a single
    ordered stream, satisfying the same `len()` / `__getitem__` /
    `.onset_frame` interface `conformal_monitor.evaluate` expects from
    `corruption.WeatherOnsetStream`. Both `nominal_dataset` and
    `degraded_dataset` are sampled with wraparound (modular indexing) if
    `scene_length` exceeds their length.

    Typical construction: two disjoint `category_subset(...)` calls on the
    same category for calibration vs. nominal-stream frames (so the
    calibration set and the "clean" portion of the monitored stream don't
    overlap), and a `category_subset(..., "falling")` for the degraded
    portion.
    """

    def __init__(self, nominal_dataset: Dataset, degraded_dataset: Dataset, onset_frame: int, scene_length: int):
        if not (0 <= onset_frame < scene_length):
            raise ValueError("onset_frame must be within [0, scene_length)")
        if len(nominal_dataset) == 0 or len(degraded_dataset) == 0:
            raise ValueError("nominal_dataset and degraded_dataset must both be non-empty")
        self.nominal_dataset = nominal_dataset
        self.degraded_dataset = degraded_dataset
        self.onset_frame = onset_frame
        self.scene_length = scene_length

    def __len__(self) -> int:
        return self.scene_length

    def __getitem__(self, t: int):
        if t < 0 or t >= self.scene_length:
            raise IndexError(t)
        if t < self.onset_frame:
            return self.nominal_dataset[t % len(self.nominal_dataset)]
        return self.degraded_dataset[(t - self.onset_frame) % len(self.degraded_dataset)]
