"""
Split conformal calibration of craf_x detections.

Standard, non-novel part of the method (see plan.md, "Setup" section) — the
novelty lives in `betting.py` and `spatial.py`, which consume the outputs of
this module. Kept dataset/model-agnostic: everything here operates on plain
NumPy arrays / torch tensors of scores, not on craf_x types directly, so it
drops in for the KITTI/nuScenes synthetic-corruption fallback or, later,
Snowy Scenes without changes.
"""
import numpy as np
import torch


def object_nonconformity_scores(
    box_preds: torch.Tensor,
    box_targets: torch.Tensor,
    match_mask: torch.Tensor,
) -> np.ndarray:
    """
    Per-object nonconformity score s_i = ||box_pred_i - box_target_i||_1,
    restricted to grid cells flagged as matched (ground-truth object present).

    Args:
        box_preds: (B, C, H, W) predicted box regression (craf_x head's 'B' output)
        box_targets: (B, C, H, W) ground-truth box regression
        match_mask: (B, 1, H, W) binary mask, 1 where a ground-truth object
            occupies that BEV cell (e.g. thresholded heatmap peaks)

    Returns:
        1D array of per-object scores, one entry per matched cell across the batch.
    """
    residual = torch.abs(box_preds - box_targets).sum(dim=1, keepdim=True)  # (B, 1, H, W)
    mask = match_mask.bool()
    scores = residual[mask].detach().cpu().numpy()
    return scores


def calibrate_quantile(calibration_scores: np.ndarray, alpha: float) -> float:
    """
    Finite-sample-valid split conformal quantile: the
    ceil((n+1)(1-alpha))/n empirical quantile of the calibration scores.

    Args:
        calibration_scores: 1D array of nonconformity scores from a held-out,
            clear-weather calibration set.
        alpha: target miscoverage level in (0, 1).

    Returns:
        q_hat, the calibrated threshold. A new object is "covered" iff its
        nonconformity score is <= q_hat.
    """
    n = calibration_scores.shape[0]
    if n == 0:
        raise ValueError("calibration_scores must be non-empty")
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")

    level = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)
    q_hat = float(np.quantile(calibration_scores, level, method="higher"))
    return q_hat


def coverage_indicators(scores: np.ndarray, q_hat: float) -> np.ndarray:
    """1 where the object is covered by the conformal region (score <= q_hat)."""
    return (scores <= q_hat).astype(np.float64)


def frame_miscoverage_rate(scores: np.ndarray, q_hat: float) -> float:
    """
    m(t) = fraction of objects in the frame NOT covered by the conformal region.

    A frame with no detected/matched objects carries no evidence of
    miscoverage and is defined to have m(t) = 0.0.
    """
    if scores.shape[0] == 0:
        return 0.0
    coverage = coverage_indicators(scores, q_hat)
    return float(np.mean(1.0 - coverage))
