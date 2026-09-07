"""
Phantom-detection-aware nonconformity score.

The existing score (`conformal_monitor.calibration.object_nonconformity_scores`)
is the l1 box-regression residual, restricted by `match_mask` to BEV cells a
ground-truth object occupies. This paper's own dataset analysis
(`conformal_monitor.real_snow_stream`, module docstring) establishes that
snow's dominant physical effect on the LiDAR modality is *spurious
near-range returns* -- that is, phantom detections in cells with no
ground-truth object at all. The matched-cell-only score cannot see those
by construction: every cell a phantom appears in is masked out before the
residual is taken.

This module adds the missing component. Two nonconformity score families
are calibrated separately by the same split-conformal procedure, each at
level alpha:

- localization, on matched cells: s = ||box_pred - box_target||_1
  (identical to the existing score, reused directly, not reimplemented)
- phantom, on unmatched cells:    s = max_c heatmap_pred[c]

and the frame-level miscoverage rate is their convex combination

    m(t) = (1 - w) * m_loc(t) + w * m_phantom(t).

Validity is preserved without new theory. The split-conformal guarantee is
agnostic to the choice of score function, so each component's calibrated
quantile gives E[m_component(t)] <= alpha under the nominal (exchangeable)
distribution; a convex combination of two quantities each with expectation
at most alpha itself has expectation at most alpha, which is exactly the H0
condition `conformal_monitor.betting`'s wealth process requires. Setting
w = 0 recovers the existing score and the existing m(t) exactly, which is
asserted as a regression test rather than assumed.
"""
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from craf_x.models.crafx_net import CRAFX_Net

from conformal_monitor.calibration import (
    calibrate_quantile,
    frame_miscoverage_rate,
    object_nonconformity_scores,
)
from conformal_monitor.evaluate import _to_device, match_mask_from_heatmap


def phantom_nonconformity_scores(
    heatmap_preds: torch.Tensor,
    match_mask: torch.Tensor,
) -> np.ndarray:
    """
    Per-cell phantom score on cells with NO ground-truth object: the
    predicted heatmap's peak class activation in that cell.

    A well-behaved detector under nominal conditions puts near-zero
    activation on empty cells, so this score is small; spurious returns
    that produce phantom object evidence drive it up. `heatmap_preds` is
    the model's "H" output, already sigmoid-squashed to [0, 1] by
    `CRAFXCenterHead`.

    Args:
        heatmap_preds: (B, C, H, W) predicted class heatmap.
        match_mask: (B, 1, H, W) binary mask, 1 where a ground-truth object
            occupies that BEV cell (the same mask the localization score is
            restricted *to*; this score uses its complement).

    Returns:
        1D array of per-cell scores, one entry per unmatched cell.
    """
    activation = heatmap_preds.amax(dim=1, keepdim=True)  # (B, 1, H, W)
    unmatched = (1.0 - match_mask).bool()
    return activation[unmatched].detach().cpu().numpy()


def phantom_aware_frame_miscoverage(
    box_preds: torch.Tensor,
    box_targets: torch.Tensor,
    heatmap_preds: torch.Tensor,
    match_mask: torch.Tensor,
    q_loc: float,
    q_phantom: float,
    phantom_weight: float,
) -> float:
    """
    m(t) = (1 - w) * m_loc(t) + w * m_phantom(t), each component calibrated
    against its own conformal quantile.

    At `phantom_weight=0.0` this is exactly the existing frame-level
    miscoverage rate on the existing score, bit for bit.
    """
    if not (0.0 <= phantom_weight <= 1.0):
        raise ValueError(f"phantom_weight must be in [0, 1], got {phantom_weight}")

    m_loc = frame_miscoverage_rate(
        object_nonconformity_scores(box_preds, box_targets, match_mask), q_loc
    )
    if phantom_weight == 0.0:
        return m_loc

    m_phantom = frame_miscoverage_rate(
        phantom_nonconformity_scores(heatmap_preds, match_mask), q_phantom
    )
    return (1.0 - phantom_weight) * m_loc + phantom_weight * m_phantom


@torch.no_grad()
def calibrate_phantom_aware(
    model: CRAFX_Net,
    calibration_dataset,
    alpha: float,
    batch_size: int = 8,
    num_workers: int = 0,
) -> Tuple[float, float]:
    """
    Split-conformal calibration of BOTH score families on the same held-out
    nominal calibration set, each at level alpha.

    Returns:
        (q_loc, q_phantom). `q_loc` is identical to what
        `conformal_monitor.evaluate.calibrate_on_clear_weather` returns for
        the same model/dataset/alpha (same score, same estimator).
    """
    model.eval()
    device = next(model.parameters()).device
    loader = DataLoader(calibration_dataset, batch_size=batch_size, num_workers=num_workers)

    loc_scores: List[np.ndarray] = []
    phantom_scores: List[np.ndarray] = []

    for batch in loader:
        image, pointcloud, targets = _to_device(
            batch["image"], batch["pointcloud"], batch["targets"], device
        )
        out = model(image, pointcloud)
        match_mask = match_mask_from_heatmap(targets["H"])
        loc_scores.append(object_nonconformity_scores(out["B"], targets["B"], match_mask))
        phantom_scores.append(phantom_nonconformity_scores(out["H"], match_mask))

    q_loc = calibrate_quantile(
        np.concatenate(loc_scores) if loc_scores else np.zeros(0), alpha
    )
    q_phantom = calibrate_quantile(
        np.concatenate(phantom_scores) if phantom_scores else np.zeros(0), alpha
    )
    return q_loc, q_phantom


@torch.no_grad()
def measure_miscoverage_by_regime(
    model: CRAFX_Net,
    stream,
    q_loc: float,
    q_phantom: float,
    phantom_weights: List[float],
) -> Dict[float, Dict[str, float]]:
    """
    Diagnostic used BEFORE running any operating curve: measure the mean
    m(t) on the stream's nominal frames (t < onset_frame) versus its
    degraded frames (t >= onset_frame), for each candidate phantom weight.

    The e-process's detection delay is driven by how far m(t) rises above
    alpha after onset, so a score change is only worth evaluating end-to-end
    if it genuinely widens that gap. Reports the real per-regime means and
    their difference so that can be checked directly rather than assumed.

    Returns:
        {phantom_weight: {"nominal_mean": ..., "degraded_mean": ...,
                          "jump": degraded_mean - nominal_mean}}
    """
    model.eval()
    device = next(model.parameters()).device
    per_weight_nominal: Dict[float, List[float]] = {w: [] for w in phantom_weights}
    per_weight_degraded: Dict[float, List[float]] = {w: [] for w in phantom_weights}

    for t in range(len(stream)):
        sample = stream[t]
        image = sample["image"].unsqueeze(0)
        pointcloud = sample["pointcloud"].unsqueeze(0)
        targets = {k: v.unsqueeze(0) for k, v in sample["targets"].items()}
        image, pointcloud, targets = _to_device(image, pointcloud, targets, device)

        out = model(image, pointcloud)
        match_mask = match_mask_from_heatmap(targets["H"])
        for w in phantom_weights:
            m_t = phantom_aware_frame_miscoverage(
                out["B"], targets["B"], out["H"], match_mask, q_loc, q_phantom, w
            )
            bucket = per_weight_nominal if t < stream.onset_frame else per_weight_degraded
            bucket[w].append(m_t)

    summary: Dict[float, Dict[str, float]] = {}
    for w in phantom_weights:
        nominal = float(np.mean(per_weight_nominal[w])) if per_weight_nominal[w] else float("nan")
        degraded = float(np.mean(per_weight_degraded[w])) if per_weight_degraded[w] else float("nan")
        summary[w] = {
            "nominal_mean": nominal,
            "degraded_mean": degraded,
            "jump": degraded - nominal,
        }
    return summary


@torch.no_grad()
def compute_phantom_aware_wealth_trajectory(
    model: CRAFX_Net,
    stream,
    q_loc: float,
    q_phantom: float,
    alpha: float,
    bettor_factory,
    phantom_weight: float,
) -> List[float]:
    """
    The existing global monitor's wealth trajectory
    (`conformal_monitor.evaluate.compute_global_wealth_trajectory`) with the
    phantom-aware m(t) substituted for the matched-cell-only one. Kept
    delta-independent for the same reason: one trajectory serves a whole
    delta sweep.
    """
    from conformal_monitor.betting import WealthProcess

    model.eval()
    device = next(model.parameters()).device
    bettor = bettor_factory()
    wealth_process = WealthProcess(alpha, lambda_max=bettor.lambda_max)
    trajectory: List[float] = []

    for t in range(len(stream)):
        sample = stream[t]
        image = sample["image"].unsqueeze(0)
        pointcloud = sample["pointcloud"].unsqueeze(0)
        targets = {k: v.unsqueeze(0) for k, v in sample["targets"].items()}
        image, pointcloud, targets = _to_device(image, pointcloud, targets, device)

        out = model(image, pointcloud)
        match_mask = match_mask_from_heatmap(targets["H"])
        m_t = phantom_aware_frame_miscoverage(
            out["B"], targets["B"], out["H"], match_mask, q_loc, q_phantom, phantom_weight
        )
        ccp_disagreement = float((1.0 - out["S"]).mean().item())

        lam = bettor.next_lambda(ccp_disagreement=ccp_disagreement)
        trajectory.append(wealth_process.step(m_t, lam))
        bettor.update(m_t, ccp_disagreement=ccp_disagreement)

    return trajectory
