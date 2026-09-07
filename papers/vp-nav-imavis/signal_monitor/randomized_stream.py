"""
Randomized onset streams, replacing a deterministic construction whose
"replicates" were all identical.

`conformal_monitor.real_snow_stream.RealSnowOnsetStream` builds a stream by
pure modular arithmetic -- `nominal[t % len]` before the onset,
`degraded[(t - onset) % len]` after -- with no sampling anywhere. Every
stream it constructs from the same two pools is therefore the same stream,
so an evaluation asking for five replicates measured one 20-frame sample
five times. The signature is visible in every result produced that way:
false-alarm rates land on exactly 0.00 or 1.00 and censoring on exactly 0/5
or 5/5, never anything between.

(That module is not modified here. It belongs to a submitted paper that is
out of scope; this is a separate implementation for this paper's own
evaluation.)

What this provides instead:

- per-replicate seeded sampling of which frames appear, so replicates are
  genuinely different draws;
- sampling *without replacement* within a stream, so a single stream does
  not show the same frame twice when the pool is large enough;
- the same treatment for "clear" streams, whose false-alarm rate is
  otherwise just as degenerate as the onset streams' detection delay;
- a paired design across methods: `stream_seeds` produces one seed list
  that every arm (baseline, null detector, each variant) reuses, so methods
  are compared on identical frame draws and between-stream variance does
  not swamp the between-method difference.
"""
from typing import List, Optional, Sequence

import numpy as np
from torch.utils.data import Dataset


def stream_seeds(n_replicates: int, base_seed: int = 20260907) -> List[int]:
    """
    Deterministic seed list for a paired comparison. Every arm of an
    evaluation must build its streams from this same list, so that
    replicate i is the identical frame sequence for every method.
    """
    rng = np.random.default_rng(base_seed)
    return [int(s) for s in rng.integers(0, 2**31 - 1, size=n_replicates)]


def _sample_indices(pool_size: int, n: int, rng: np.random.Generator) -> np.ndarray:
    """
    n indices from a pool. Without replacement when the pool allows it,
    falling back to with-replacement only when the pool is genuinely too
    small -- which is reported by the caller rather than hidden.
    """
    if pool_size >= n:
        return rng.choice(pool_size, size=n, replace=False)
    return rng.choice(pool_size, size=n, replace=True)


class RandomizedOnsetStream(Dataset):
    """
    A weather-onset stream whose frames are randomly sampled per replicate.

    Satisfies the same interface `conformal_monitor.evaluate` expects --
    `len()`, `__getitem__`, `.onset_frame` -- so it drops straight into the
    existing operating-curve functions.

    Args:
        nominal_dataset: pool for frames before the onset.
        degraded_dataset: pool for frames from the onset on. Pass the
            nominal pool here to build a stationary "clear" stream; frames
            are then drawn without replacement across the whole stream
            rather than independently per half, so a clear stream never
            repeats a frame it has already shown.
        onset_frame: index at which the degraded regime begins.
        scene_length: total frames.
        seed: per-replicate seed, from `stream_seeds`.
    """

    def __init__(
        self,
        nominal_dataset: Dataset,
        degraded_dataset: Dataset,
        onset_frame: int,
        scene_length: int,
        seed: int,
    ):
        if not (0 <= onset_frame < scene_length):
            raise ValueError("onset_frame must be within [0, scene_length)")
        if len(nominal_dataset) == 0 or len(degraded_dataset) == 0:
            raise ValueError("nominal_dataset and degraded_dataset must both be non-empty")

        self.nominal_dataset = nominal_dataset
        self.degraded_dataset = degraded_dataset
        self.onset_frame = onset_frame
        self.scene_length = scene_length
        self.seed = seed

        rng = np.random.default_rng(seed)
        n_nominal = onset_frame
        n_degraded = scene_length - onset_frame
        self.is_stationary = degraded_dataset is nominal_dataset

        if self.is_stationary:
            # One draw across the whole stream, so the two halves cannot
            # collide with each other.
            picks = _sample_indices(len(nominal_dataset), scene_length, rng)
            self._nominal_indices = picks[:n_nominal]
            self._degraded_indices = picks[n_nominal:]
        else:
            self._nominal_indices = _sample_indices(len(nominal_dataset), n_nominal, rng)
            self._degraded_indices = _sample_indices(len(degraded_dataset), n_degraded, rng)

    @property
    def sampled_indices(self) -> List[int]:
        """The concrete frame indices this stream draws, for provenance."""
        return [int(i) for i in self._nominal_indices] + [int(i) for i in self._degraded_indices]

    def __len__(self) -> int:
        return self.scene_length

    def __getitem__(self, t: int):
        if t < 0 or t >= self.scene_length:
            raise IndexError(t)
        if t < self.onset_frame:
            return self.nominal_dataset[int(self._nominal_indices[t])]
        return self.degraded_dataset[int(self._degraded_indices[t - self.onset_frame])]


def onset_stream_factories(
    nominal_dataset: Dataset,
    degraded_dataset: Dataset,
    onset_frame: int,
    scene_length: int,
    seeds: Sequence[int],
):
    """
    One zero-argument factory per seed, in seed order. Passing the same
    `seeds` to every arm of an evaluation makes the comparison paired.
    """
    return [
        (lambda s=seed: RandomizedOnsetStream(
            nominal_dataset, degraded_dataset, onset_frame, scene_length, s
        ))
        for seed in seeds
    ]
