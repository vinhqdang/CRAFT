import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "conformal-snow-icra2027")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pytest
import torch

from craf_x.config import CRAFXConfig
from craf_x.datasets.nuscenes_mock import NuScenesMockDataset
from craf_x.models.crafx_net import CRAFX_Net

from conformal_monitor.betting import AGRAPABettor
from conformal_monitor.calibration import frame_miscoverage_rate, object_nonconformity_scores
from conformal_monitor.evaluate import calibrate_on_clear_weather, compute_global_wealth_trajectory
from conformal_monitor.corruption import WeatherOnsetStream

from signal_monitor.phantom_score import (
    calibrate_phantom_aware,
    compute_phantom_aware_wealth_trajectory,
    measure_miscoverage_by_regime,
    phantom_aware_frame_miscoverage,
    phantom_nonconformity_scores,
)

CONFORMAL_ALPHA = 0.2


def _build_model():
    config = CRAFXConfig(bev_h=32, bev_w=32)
    return CRAFX_Net(config)


def test_phantom_scores_come_from_unmatched_cells_only():
    # Arrange: a heatmap that is hot exactly where the match mask is 1 and
    # cold everywhere else. The phantom score looks at the COMPLEMENT of the
    # mask, so it must see only the cold values.
    heatmap = torch.zeros(1, 2, 4, 4)
    match_mask = torch.zeros(1, 1, 4, 4)
    match_mask[0, 0, 0, 0] = 1.0
    match_mask[0, 0, 1, 1] = 1.0
    heatmap[0, 0][match_mask[0, 0].bool()] = 0.9
    heatmap[0, 0][~match_mask[0, 0].bool()] = 0.1

    scores = phantom_nonconformity_scores(heatmap, match_mask)

    assert scores.shape[0] == 16 - 2  # every cell except the two matched ones
    assert np.allclose(scores, 0.1)


def test_phantom_score_rises_with_spurious_activation():
    # Two frames with identical (empty) ground truth; the second has phantom
    # activation on unmatched cells. The score must increase.
    match_mask = torch.zeros(1, 1, 4, 4)
    quiet = torch.full((1, 1, 4, 4), 0.02)
    phantom = torch.full((1, 1, 4, 4), 0.7)

    assert phantom_nonconformity_scores(phantom, match_mask).mean() > phantom_nonconformity_scores(
        quiet, match_mask
    ).mean()


def test_phantom_weight_zero_reduces_to_existing_miscoverage_exactly():
    # The key regression guard: w=0 must reproduce the existing
    # matched-cell-only m(t) bit for bit, so the new score is strictly an
    # extension of the old one rather than a silent replacement.
    torch.manual_seed(0)
    box_preds = torch.randn(1, 8, 8, 8)
    box_targets = torch.randn(1, 8, 8, 8)
    heatmap = torch.rand(1, 2, 8, 8)
    match_mask = (torch.rand(1, 1, 8, 8) > 0.5).float()
    q_loc, q_phantom = 3.0, 0.5

    expected = frame_miscoverage_rate(
        object_nonconformity_scores(box_preds, box_targets, match_mask), q_loc
    )
    actual = phantom_aware_frame_miscoverage(
        box_preds, box_targets, heatmap, match_mask, q_loc, q_phantom, phantom_weight=0.0
    )

    assert actual == expected


def test_phantom_aware_miscoverage_is_bounded_and_convex():
    torch.manual_seed(1)
    box_preds = torch.randn(1, 8, 8, 8)
    box_targets = torch.randn(1, 8, 8, 8)
    heatmap = torch.rand(1, 2, 8, 8)
    match_mask = (torch.rand(1, 1, 8, 8) > 0.5).float()
    q_loc, q_phantom = 3.0, 0.5

    m_loc = phantom_aware_frame_miscoverage(
        box_preds, box_targets, heatmap, match_mask, q_loc, q_phantom, 0.0
    )
    m_phantom = phantom_aware_frame_miscoverage(
        box_preds, box_targets, heatmap, match_mask, q_loc, q_phantom, 1.0
    )
    m_mid = phantom_aware_frame_miscoverage(
        box_preds, box_targets, heatmap, match_mask, q_loc, q_phantom, 0.5
    )

    # m(t) must stay a valid rate, and the convex combination must lie
    # between its two endpoints -- the property the H0 bound relies on.
    for m in (m_loc, m_phantom, m_mid):
        assert 0.0 <= m <= 1.0
    assert min(m_loc, m_phantom) <= m_mid <= max(m_loc, m_phantom)
    assert m_mid == pytest.approx(0.5 * m_loc + 0.5 * m_phantom)


def test_phantom_weight_out_of_range_rejected():
    box_preds = torch.zeros(1, 2, 4, 4)
    box_targets = torch.zeros(1, 2, 4, 4)
    heatmap = torch.zeros(1, 1, 4, 4)
    match_mask = torch.ones(1, 1, 4, 4)
    for bad_weight in (-0.1, 1.1):
        with pytest.raises(ValueError):
            phantom_aware_frame_miscoverage(
                box_preds, box_targets, heatmap, match_mask, 1.0, 1.0, bad_weight
            )


def test_calibrate_phantom_aware_matches_existing_q_hat_on_localization():
    # q_loc uses the identical score and identical quantile estimator as the
    # existing calibration, so the two must agree exactly.
    model = _build_model()
    dataset = NuScenesMockDataset(num_samples=4)

    # NuScenesMockDataset draws fresh random tensors on every __getitem__,
    # so the two passes must be seeded identically to see the same data.
    torch.manual_seed(7)
    q_loc, q_phantom = calibrate_phantom_aware(model, dataset, CONFORMAL_ALPHA, batch_size=2)
    torch.manual_seed(7)
    q_existing = calibrate_on_clear_weather(model, dataset, CONFORMAL_ALPHA, batch_size=2)

    assert q_loc == pytest.approx(q_existing)
    assert 0.0 <= q_phantom <= 1.0  # a sigmoid heatmap activation


def test_measure_miscoverage_by_regime_reports_both_regimes():
    model = _build_model()
    dataset = NuScenesMockDataset(num_samples=4)
    q_loc, q_phantom = calibrate_phantom_aware(model, dataset, CONFORMAL_ALPHA, batch_size=2)
    stream = WeatherOnsetStream(NuScenesMockDataset(num_samples=4), scene_length=6, onset_frame=3, ramp_length=1)

    summary = measure_miscoverage_by_regime(model, stream, q_loc, q_phantom, [0.0, 0.5, 1.0])

    assert set(summary.keys()) == {0.0, 0.5, 1.0}
    for stats in summary.values():
        assert 0.0 <= stats["nominal_mean"] <= 1.0
        assert 0.0 <= stats["degraded_mean"] <= 1.0
        assert stats["jump"] == pytest.approx(stats["degraded_mean"] - stats["nominal_mean"])


def test_phantom_aware_trajectory_at_zero_weight_matches_existing_monitor():
    # End-to-end guard: with w=0 the whole new trajectory must coincide with
    # the existing global monitor's, confirming the only behavioral change
    # comes from the phantom component itself.
    model = _build_model()
    dataset = NuScenesMockDataset(num_samples=4)
    q_loc, q_phantom = calibrate_phantom_aware(model, dataset, CONFORMAL_ALPHA, batch_size=2)

    def make_stream():
        return WeatherOnsetStream(
            NuScenesMockDataset(num_samples=4), scene_length=5, onset_frame=2, ramp_length=1
        )

    # Same seeding requirement as above: the mock stream's frames are drawn
    # randomly per access, so both trajectories must start from one seed.
    torch.manual_seed(11)
    new = compute_phantom_aware_wealth_trajectory(
        model, make_stream(), q_loc, q_phantom, CONFORMAL_ALPHA,
        lambda: AGRAPABettor(CONFORMAL_ALPHA), phantom_weight=0.0,
    )
    torch.manual_seed(11)
    existing = compute_global_wealth_trajectory(
        model, make_stream(), q_loc, CONFORMAL_ALPHA, lambda: AGRAPABettor(CONFORMAL_ALPHA)
    )

    assert new == pytest.approx(existing)
