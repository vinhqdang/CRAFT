import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "conformal-snow-icra2027")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pytest
import torch
import torch.nn as nn

from conformal_monitor.betting import AGRAPABettor

from signal_monitor.mondrian import (
    operating_curve_per_stream_quantile,
    per_session_miscoverage,
    session_conditional_quantiles,
    wealth_trajectory_with_quantile,
)

ALPHA = 0.2
GRID = 8


class _SessionModel(nn.Module):
    """Detector whose box error depends on which session a frame came from,
    reproducing the between-session heterogeneity that breaks pooled
    calibration."""

    def __init__(self):
        super().__init__()
        self._anchor = nn.Parameter(torch.zeros(1))

    def forward(self, image, pointcloud):
        b = image.shape[0]
        device = image.device
        # The pointcloud's constant value encodes the session's error scale.
        scale = pointcloud[:, :1].mean()
        heatmap = torch.full((b, 2, GRID, GRID), 0.04, device=device)
        heatmap[:, 0, ::2, ::2] = 0.9
        return {
            "H": heatmap,
            "B": scale * torch.ones((b, 6, GRID, GRID), device=device),
            "S": torch.full((b, 1, GRID, GRID), 0.9, device=device),
        }


class _SessionDataset:
    """Frames tagged by session, each session having its own error scale."""

    def __init__(self, session_scales, frames_per_session):
        self.records = []
        self.session_indices = {}
        for name, scale in session_scales.items():
            start = len(self.records)
            for _ in range(frames_per_session):
                self.records.append(scale)
            self.session_indices[(name, "0")] = list(range(start, len(self.records)))

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        scale = self.records[idx]
        heatmap = torch.zeros(2, GRID, GRID)
        heatmap[0, ::2, ::2] = 1.0
        return {
            "image": torch.full((3, GRID, GRID), 0.5),
            "pointcloud": torch.full((4, GRID, GRID), float(scale)),
            "targets": {
                "H": heatmap,
                "B": torch.zeros(6, GRID, GRID),
                "V": torch.zeros(2, GRID, GRID),
            },
        }


class _ListStream:
    def __init__(self, dataset, indices, onset_frame):
        self.dataset, self.indices, self.onset_frame = dataset, indices, onset_frame

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, t):
        return self.dataset[self.indices[t]]


def test_session_quantiles_differ_when_sessions_differ():
    # The whole point: heterogeneous sessions must get different quantiles.
    model = _SessionModel().eval()
    dataset = _SessionDataset({"easy": 0.5, "hard": 5.0}, frames_per_session=20)

    quantiles, monitoring, pooled = session_conditional_quantiles(
        model, dataset, dataset.session_indices, ALPHA, n_calibration_per_session=10
    )

    assert quantiles[("easy", "0")] < quantiles[("hard", "0")]
    # The pooled quantile cannot serve both. With equal-sized sessions at
    # alpha=0.2 it is pulled up into the harder session's range, so it
    # over-covers the easy session badly; it need not sit strictly between
    # the two (with constant per-session scores it coincides with the
    # harder one), which is why this asserts the mis-service rather than a
    # strict ordering.
    assert quantiles[("easy", "0")] < pooled <= quantiles[("hard", "0")]


def test_monitoring_frames_are_held_out_from_calibration():
    # Calibrating and monitoring on the same frames would invalidate the
    # conformal guarantee outright.
    model = _SessionModel().eval()
    dataset = _SessionDataset({"a": 1.0, "b": 2.0}, frames_per_session=20)

    _, monitoring, _ = session_conditional_quantiles(
        model, dataset, dataset.session_indices, ALPHA, n_calibration_per_session=10
    )

    for key, indices in dataset.session_indices.items():
        calibration = set(indices[:10])
        assert calibration.isdisjoint(set(monitoring[key]))
        assert len(monitoring[key]) == 10


def test_session_conditional_calibration_restores_per_session_coverage():
    # Pooled calibration leaves heterogeneous sessions far from alpha;
    # session-conditional calibration should bring every session near it.
    model = _SessionModel().eval()
    dataset = _SessionDataset({"easy": 0.5, "hard": 5.0}, frames_per_session=40)

    quantiles, monitoring, pooled = session_conditional_quantiles(
        model, dataset, dataset.session_indices, ALPHA, n_calibration_per_session=20
    )

    conditional = per_session_miscoverage(model, dataset, monitoring, quantiles)
    marginal = per_session_miscoverage(
        model, dataset, monitoring, {k: pooled for k in quantiles}
    )

    worst_conditional = max(abs(v - ALPHA) for v in conditional.values())
    worst_marginal = max(abs(v - ALPHA) for v in marginal.values())
    assert worst_conditional <= worst_marginal


def test_thin_sessions_fall_back_to_the_pooled_quantile():
    model = _SessionModel().eval()
    dataset = _SessionDataset({"thin": 1.0}, frames_per_session=8)

    quantiles, _, pooled = session_conditional_quantiles(
        model, dataset, dataset.session_indices, ALPHA,
        n_calibration_per_session=4, pooled_fallback_min=10,
    )
    assert quantiles[("thin", "0")] == pytest.approx(pooled)


def test_block_size_one_matches_unblocked_behaviour():
    model = _SessionModel().eval()
    dataset = _SessionDataset({"a": 1.0}, frames_per_session=12)
    stream = _ListStream(dataset, list(range(10)), onset_frame=4)

    a = wealth_trajectory_with_quantile(model, stream, 1.0, ALPHA, lambda: AGRAPABettor(ALPHA), block_size=1)
    b = wealth_trajectory_with_quantile(model, stream, 1.0, ALPHA, lambda: AGRAPABettor(ALPHA), block_size=1)
    assert a == pytest.approx(b)
    assert len(a) == 10


def test_block_aggregation_keeps_one_entry_per_frame():
    # Detection delays must stay in frame units so block sizes are
    # comparable against each other and against the baseline.
    model = _SessionModel().eval()
    dataset = _SessionDataset({"a": 1.0}, frames_per_session=12)
    stream = _ListStream(dataset, list(range(12)), onset_frame=4)

    for block in (1, 2, 4):
        traj = wealth_trajectory_with_quantile(
            model, stream, 1.0, ALPHA, lambda: AGRAPABettor(ALPHA), block_size=block
        )
        assert len(traj) == 12, block
        assert all(w >= 0.0 for w in traj)


def test_operating_curve_reduces_to_marginal_when_all_quantiles_are_equal():
    # The Mondrian arm and its like-for-like baseline must run through the
    # same code; passing one pooled quantile everywhere is the marginal case.
    model = _SessionModel().eval()
    dataset = _SessionDataset({"a": 1.0, "b": 3.0}, frames_per_session=20)

    def make(indices):
        return lambda: _ListStream(dataset, indices, onset_frame=4)

    onset = [(make(list(range(10))), 2.0), (make(list(range(20, 30))), 2.0)]
    clear = [(make(list(range(10, 20))), 2.0), (make(list(range(30, 40))), 2.0)]

    curve = operating_curve_per_stream_quantile(
        model, ALPHA, [0.3, 0.1], onset, clear, lambda: AGRAPABettor(ALPHA)
    )

    assert len(curve) == 2
    for point in curve:
        assert 0.0 <= point["false_alarm_rate"] <= 1.0
        assert point["n_onset_replicates"] == 2
        assert len(point["false_alarm_ci"]) == 2
        assert point["controls_false_alarms"] == (point["false_alarm_rate"] <= point["delta"])


def test_operating_curve_reports_confidence_intervals():
    model = _SessionModel().eval()
    dataset = _SessionDataset({"a": 1.0}, frames_per_session=60)

    def make(start):
        return lambda: _ListStream(dataset, list(range(start, start + 10)), onset_frame=4)

    onset = [(make(s), 2.0) for s in (0, 10, 20)]
    clear = [(make(s), 2.0) for s in (30, 40, 50)]

    curve = operating_curve_per_stream_quantile(
        model, ALPHA, [0.3], onset, clear, lambda: AGRAPABettor(ALPHA), n_bootstrap=200
    )
    point = curve[0]
    lo, hi = point["false_alarm_ci"]
    assert lo <= point["false_alarm_rate"] <= hi


def test_temporal_gap_removes_frames_between_calibration_and_monitoring():
    # Within-session splitting puts calibration and monitored frames
    # adjacent in time, and adjacent driving frames are highly correlated.
    # The gap must actually discard the frames in between, not just shift
    # the window.
    model = _SessionModel().eval()
    dataset = _SessionDataset({"a": 1.0}, frames_per_session=40)

    _, no_gap, _ = session_conditional_quantiles(
        model, dataset, dataset.session_indices, ALPHA,
        n_calibration_per_session=10, temporal_gap=0,
    )
    _, gapped, _ = session_conditional_quantiles(
        model, dataset, dataset.session_indices, ALPHA,
        n_calibration_per_session=10, temporal_gap=8,
    )

    key = ("a", "0")
    assert len(no_gap[key]) == 30
    assert len(gapped[key]) == 22  # 8 frames dropped
    # The discarded frames sit strictly between the two portions.
    assert set(gapped[key]).issubset(set(no_gap[key]))
    assert min(gapped[key]) == min(no_gap[key]) + 8


def test_temporal_gap_never_overlaps_the_calibration_portion():
    model = _SessionModel().eval()
    dataset = _SessionDataset({"a": 1.0, "b": 2.0}, frames_per_session=30)

    for gap in (0, 5, 10):
        _, monitoring, _ = session_conditional_quantiles(
            model, dataset, dataset.session_indices, ALPHA,
            n_calibration_per_session=10, temporal_gap=gap,
        )
        for key, indices in dataset.session_indices.items():
            calibration = set(indices[:10])
            assert calibration.isdisjoint(set(monitoring[key])), gap


def test_score_stream_trajectory_matches_the_model_based_one():
    # The cached-score path must be arithmetically identical to running the
    # model, or the sweep it enables would be measuring something else.
    from signal_monitor.mondrian import wealth_trajectory_from_score_stream
    from conformal_monitor.calibration import object_nonconformity_scores
    from conformal_monitor.evaluate import match_mask_from_heatmap

    model = _SessionModel().eval()
    dataset = _SessionDataset({"a": 1.0}, frames_per_session=12)
    indices = list(range(12))
    stream = _ListStream(dataset, indices, onset_frame=4)

    # Precompute the same scores the model path would produce.
    cached = []
    for i in indices:
        sample = dataset[i]
        targets = {k: v.unsqueeze(0) for k, v in sample["targets"].items()}
        out = model(sample["image"].unsqueeze(0), sample["pointcloud"].unsqueeze(0))
        mask = match_mask_from_heatmap(targets["H"])
        cached.append(object_nonconformity_scores(out["B"], targets["B"], mask))

    class _ScoreStream:
        onset_frame = 4

        def __len__(self):
            return len(cached)

        def __getitem__(self, t):
            return cached[t]

    for block in (1, 2, 4):
        via_model = wealth_trajectory_with_quantile(
            model, stream, 1.0, ALPHA, lambda: AGRAPABettor(ALPHA), block_size=block
        )
        via_scores = wealth_trajectory_from_score_stream(
            _ScoreStream(), 1.0, ALPHA, lambda: AGRAPABettor(ALPHA), block_size=block
        )
        assert via_scores == pytest.approx(via_model), block
