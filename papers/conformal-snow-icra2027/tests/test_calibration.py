import numpy as np
import torch

from conformal_monitor.calibration import (
    calibrate_quantile,
    coverage_indicators,
    frame_miscoverage_rate,
    object_nonconformity_scores,
)


def test_object_nonconformity_scores_extracts_matched_cells_only():
    box_preds = torch.zeros(1, 2, 2, 2)
    box_targets = torch.zeros(1, 2, 2, 2)
    box_targets[0, :, 0, 0] = 3.0  # residual = |0-3| + |0-3| = 6 at (0,0)
    box_targets[0, :, 1, 1] = 1.0  # residual = 2 at (1,1), but unmatched

    match_mask = torch.zeros(1, 1, 2, 2)
    match_mask[0, 0, 0, 0] = 1.0

    scores = object_nonconformity_scores(box_preds, box_targets, match_mask)
    assert scores.shape == (1,)
    assert scores[0] == 6.0


def test_calibrate_quantile_covers_at_least_target_level():
    rng = np.random.default_rng(0)
    scores = rng.uniform(0, 1, size=1000)
    alpha = 0.1

    q_hat = calibrate_quantile(scores, alpha)
    empirical_coverage = np.mean(scores <= q_hat)
    # Finite-sample guarantee is for a *fresh* test point, not the
    # calibration set itself, but on 1000 iid uniform draws the empirical
    # coverage on the calibration set should still land close to 1 - alpha.
    assert empirical_coverage >= 1 - alpha - 0.02


def test_calibrate_quantile_rejects_invalid_alpha():
    scores = np.array([0.1, 0.2, 0.3])
    try:
        calibrate_quantile(scores, 1.5)
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_calibrate_quantile_rejects_empty_scores():
    try:
        calibrate_quantile(np.zeros(0), 0.1)
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_coverage_indicators():
    scores = np.array([0.1, 0.5, 0.9])
    cov = coverage_indicators(scores, q_hat=0.5)
    assert list(cov) == [1.0, 1.0, 0.0]


def test_frame_miscoverage_rate_empty_frame_is_zero():
    assert frame_miscoverage_rate(np.zeros(0), q_hat=0.5) == 0.0


def test_frame_miscoverage_rate_half_covered():
    scores = np.array([0.1, 0.9])  # one covered, one not, at q_hat=0.5
    assert frame_miscoverage_rate(scores, q_hat=0.5) == 0.5
