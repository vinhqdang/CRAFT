import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "conformal-snow-icra2027")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest

from signal_monitor.randomized_stream import (
    RandomizedOnsetStream,
    onset_stream_factories,
    stream_seeds,
)


class _TaggedPool:
    """Pool whose items identify themselves, so a stream's frame sequence
    can be compared directly."""

    def __init__(self, tag, n):
        self.tag, self.n = tag, n

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        return f"{self.tag}{i}"


def _sequence(stream):
    return [stream[t] for t in range(len(stream))]


def test_replicates_actually_differ():
    # THE regression guard. The previous deterministic stream produced five
    # identical "replicates", so every reported statistic was one 20-frame
    # sample measured five times, with false-alarm rates landing on exactly
    # 0.00 or 1.00 and censoring on exactly 0/5 or 5/5.
    nominal, degraded = _TaggedPool("nom", 100), _TaggedPool("deg", 100)
    seeds = stream_seeds(5, base_seed=1)
    sequences = [
        _sequence(RandomizedOnsetStream(nominal, degraded, 8, 20, s)) for s in seeds
    ]

    unique = {tuple(s) for s in sequences}
    assert len(unique) == len(sequences), "replicates must be distinct draws"


def test_clear_streams_also_differ_across_replicates():
    # The false-alarm rate is just as meaningless as the detection delay if
    # the stationary streams are all the same stream.
    nominal = _TaggedPool("nom", 100)
    seeds = stream_seeds(5, base_seed=2)
    sequences = [_sequence(RandomizedOnsetStream(nominal, nominal, 8, 20, s)) for s in seeds]

    assert len({tuple(s) for s in sequences}) == len(sequences)


def test_same_seed_reproduces_the_same_stream():
    nominal, degraded = _TaggedPool("nom", 50), _TaggedPool("deg", 50)
    a = _sequence(RandomizedOnsetStream(nominal, degraded, 8, 20, seed=1234))
    b = _sequence(RandomizedOnsetStream(nominal, degraded, 8, 20, seed=1234))
    assert a == b


def test_frames_are_not_repeated_within_a_stream_when_the_pool_allows():
    nominal, degraded = _TaggedPool("nom", 100), _TaggedPool("deg", 100)
    seq = _sequence(RandomizedOnsetStream(nominal, degraded, 8, 20, seed=7))
    assert len(set(seq)) == len(seq)


def test_stationary_stream_does_not_repeat_across_its_two_halves():
    # A clear stream draws both halves from one pool; if the halves were
    # sampled independently they could collide, showing the same frame twice.
    nominal = _TaggedPool("nom", 100)
    seq = _sequence(RandomizedOnsetStream(nominal, nominal, 8, 20, seed=11))
    assert len(set(seq)) == len(seq)


def test_stream_respects_the_onset_boundary():
    nominal, degraded = _TaggedPool("nom", 100), _TaggedPool("deg", 100)
    stream = RandomizedOnsetStream(nominal, degraded, onset_frame=8, scene_length=20, seed=3)
    seq = _sequence(stream)

    assert all(f.startswith("nom") for f in seq[:8])
    assert all(f.startswith("deg") for f in seq[8:])
    assert stream.onset_frame == 8
    assert len(stream) == 20


def test_small_pool_falls_back_to_replacement_without_crashing():
    # A pool smaller than the stream cannot supply distinct frames; the
    # stream must still build rather than raise.
    nominal, degraded = _TaggedPool("nom", 3), _TaggedPool("deg", 2)
    seq = _sequence(RandomizedOnsetStream(nominal, degraded, 8, 20, seed=5))
    assert len(seq) == 20


def test_stream_seeds_are_deterministic_and_distinct():
    assert stream_seeds(10, base_seed=42) == stream_seeds(10, base_seed=42)
    assert len(set(stream_seeds(30, base_seed=42))) == 30
    assert stream_seeds(5, base_seed=1) != stream_seeds(5, base_seed=2)


def test_factories_give_a_paired_design_across_arms():
    # Two different "methods" building streams from the same seed list must
    # see identical frame sequences, replicate by replicate -- that is what
    # makes the comparison paired.
    nominal, degraded = _TaggedPool("nom", 100), _TaggedPool("deg", 100)
    seeds = stream_seeds(6, base_seed=99)

    arm_a = onset_stream_factories(nominal, degraded, 8, 20, seeds)
    arm_b = onset_stream_factories(nominal, degraded, 8, 20, seeds)
    for fa, fb in zip(arm_a, arm_b):
        assert _sequence(fa()) == _sequence(fb())

    # ...and different replicates within one arm must still differ.
    assert len({tuple(_sequence(f())) for f in arm_a}) == len(arm_a)


def test_invalid_arguments_rejected():
    nominal, degraded = _TaggedPool("nom", 10), _TaggedPool("deg", 10)
    with pytest.raises(ValueError):
        RandomizedOnsetStream(nominal, degraded, onset_frame=20, scene_length=20, seed=1)
    with pytest.raises(ValueError):
        RandomizedOnsetStream(_TaggedPool("nom", 0), degraded, 8, 20, seed=1)
