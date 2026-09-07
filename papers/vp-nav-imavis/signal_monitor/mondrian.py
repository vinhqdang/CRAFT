"""
Session-conditional (Mondrian) conformal calibration.

Marginal split conformal calibration pools nonconformity scores across all
calibration data and produces one quantile. That is valid *on average* over
the calibration distribution, and on CADC it fails per session: measured
per-drive mean miscoverage under a single pooled q_hat ranges from 0.083 to
0.623 against a target of 0.20 -- a 7.5x spread across drives that are all
nominal, clear-road, same collection date. Four of nine nominal drives sit
above alpha. The monitor then alarms on clear streams from the unlucky
drives, giving a false-alarm rate of 1.00.

The between-drive standard deviation of per-frame m (0.061) is as large as
the weather effect the monitor is meant to detect (+0.065, CI [+0.036,
+0.094]), so under pooled calibration the monitored quantity is dominated
by which session you are on rather than by the weather.

Mondrian conformal prediction (Vovk, Lindsay, Nouretdinov & Gammerman,
"Mondrian Confidence Machine", Working Paper 4, On-line Compression
Modelling project, Royal Holloway, 2003; see also Vovk, Gammerman & Shafer,
"Algorithmic Learning in a Random World", 2005) conditions calibration on a
taxonomy, giving validity within each category rather than marginally. Here
the category is the driving session.

## The design question this forces

Mondrian needs calibration data in every category it will be asked about.
The existing CADC protocol splits *whole drives* into calibration drives
and nominal drives, so a monitored drive has no calibration frames of its
own and a per-drive quantile simply does not exist for it. Session-
conditional calibration is therefore impossible under that split, not
merely awkward.

This module changes the split to a within-session one: each drive
contributes its own first `n_calibration_per_session` frames to
calibration, and is monitored on its remaining frames. That is not a
workaround -- it is the deployment-faithful arrangement. A vehicle
calibrates on the early, known-nominal portion of the session it is
currently driving, then monitors the rest of that same session. It never
has to assume a quantile estimated on last week's drives transfers to
today's.

Degraded frames are scored against the quantile of the nominal session the
stream started in, which is likewise what deployment does: you calibrate on
your current session's clear portion and raise an alarm when incoming
frames stop matching it. The degraded regime is the anomaly being detected,
so it has no calibration data by construction and must not have any.
"""
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Subset

from craf_x.models.crafx_net import CRAFX_Net

from conformal_monitor.betting import WealthProcess
from conformal_monitor.calibration import (
    calibrate_quantile,
    frame_miscoverage_rate,
    object_nonconformity_scores,
)
from conformal_monitor.evaluate import (
    alarm_time_from_trajectory,
    _to_device,
    match_mask_from_heatmap,
)


@torch.no_grad()
def _frame_scores(model, dataset, index, device):
    sample = dataset[int(index)]
    image = sample["image"].unsqueeze(0)
    pointcloud = sample["pointcloud"].unsqueeze(0)
    targets = {k: v.unsqueeze(0) for k, v in sample["targets"].items()}
    image, pointcloud, targets = _to_device(image, pointcloud, targets, device)
    out = model(image, pointcloud)
    mask = match_mask_from_heatmap(targets["H"])
    return object_nonconformity_scores(out["B"], targets["B"], mask)


@torch.no_grad()
def session_conditional_quantiles(
    model: CRAFX_Net,
    dataset,
    session_indices: Dict[Tuple[str, str], List[int]],
    alpha: float,
    n_calibration_per_session: int,
    temporal_gap: int = 0,
    pooled_fallback_min: int = 10,
) -> Tuple[Dict[Tuple[str, str], float], Dict[Tuple[str, str], List[int]], float]:
    """
    One calibrated quantile per session, from that session's own frames.

    Args:
        session_indices: session key -> that session's frame indices into
            `dataset`, in temporal order.
        n_calibration_per_session: how many of each session's leading frames
            are reserved for calibration and withheld from monitoring.
        temporal_gap: frames discarded between the calibration portion and
            the monitored portion of each session. Within-session splitting
            places calibration and monitored frames adjacent in time, and
            adjacent driving frames are highly correlated -- the quantile
            would be fitted on data nearly identical to what it scores,
            making nominal-side miscoverage optimistically low and
            *widening* the apparent nominal-to-degraded gap. That would
            manufacture part of the very effect being measured. A gap breaks
            the adjacency while keeping the deployment story intact: a
            vehicle can calibrate on the first stretch of a drive, wait,
            then begin monitoring. Sweep it; an effect that survives a
            growing gap is real, one that shrinks was partly adjacency.
        pooled_fallback_min: a session with fewer than this many calibration
            frames gets the pooled quantile instead of its own, since a
            quantile from a handful of frames is worse than a marginal one.
            Which sessions fell back is returned by the caller's inspection
            of the returned dict against `session_indices`.

    Returns:
        (per-session quantiles, per-session held-out monitoring indices,
         pooled quantile). The pooled quantile is returned both as the
         thin-session fallback and so a like-for-like marginal baseline can
         be run on the identical split.
    """
    model.eval()
    device = next(model.parameters()).device

    per_session_scores: Dict[Tuple[str, str], List[np.ndarray]] = {}
    monitoring_indices: Dict[Tuple[str, str], List[int]] = {}
    all_scores: List[np.ndarray] = []

    for key, indices in session_indices.items():
        calibration_part = indices[:n_calibration_per_session]
        monitoring_indices[key] = list(indices[n_calibration_per_session + temporal_gap:])
        scores = [_frame_scores(model, dataset, i, device) for i in calibration_part]
        per_session_scores[key] = scores
        all_scores.extend(scores)

    pooled = calibrate_quantile(
        np.concatenate(all_scores) if all_scores else np.zeros(0), alpha
    )

    quantiles: Dict[Tuple[str, str], float] = {}
    for key, scores in per_session_scores.items():
        if len(scores) < pooled_fallback_min:
            quantiles[key] = pooled
            continue
        stacked = np.concatenate(scores) if scores else np.zeros(0)
        quantiles[key] = calibrate_quantile(stacked, alpha) if stacked.size else pooled

    return quantiles, monitoring_indices, pooled


@torch.no_grad()
def per_session_miscoverage(
    model: CRAFX_Net,
    dataset,
    monitoring_indices: Dict[Tuple[str, str], List[int]],
    quantiles: Dict[Tuple[str, str], float],
    max_frames: Optional[int] = None,
) -> Dict[Tuple[str, str], float]:
    """
    Each session's mean held-out frame miscoverage under its own quantile.
    Under working session-conditional calibration these should sit near
    alpha for every session, which is precisely what the pooled quantile
    failed to achieve.
    """
    model.eval()
    device = next(model.parameters()).device
    result = {}
    for key, indices in monitoring_indices.items():
        chosen = indices[:max_frames] if max_frames else indices
        values = [
            frame_miscoverage_rate(_frame_scores(model, dataset, i, device), quantiles[key])
            for i in chosen
        ]
        result[key] = float(np.mean(values)) if values else float("nan")
    return result


@torch.no_grad()
def wealth_trajectory_with_quantile(
    model: CRAFX_Net,
    stream,
    q_hat: float,
    alpha: float,
    bettor_factory: Callable[[], object],
    block_size: int = 1,
) -> List[float]:
    """
    Global wealth trajectory for one stream under one quantile.

    `block_size` averages m over consecutive frames before the betting step.
    At 1 this is exactly the existing per-frame behaviour; above 1 it trades
    temporal resolution for a ~sqrt(block_size) reduction in the noise on m,
    which matters when the effect is well under one standard deviation. The
    returned trajectory has one entry per frame regardless, so detection
    delays stay in frame units and remain comparable across block sizes --
    wealth simply only updates when a block completes.
    """
    model.eval()
    device = next(model.parameters()).device
    bettor = bettor_factory()
    wealth_process = WealthProcess(alpha, lambda_max=bettor.lambda_max)

    trajectory: List[float] = []
    pending_m: List[float] = []
    pending_ccp: List[float] = []
    wealth = 1.0

    for t in range(len(stream)):
        sample = stream[t]
        image = sample["image"].unsqueeze(0)
        pointcloud = sample["pointcloud"].unsqueeze(0)
        targets = {k: v.unsqueeze(0) for k, v in sample["targets"].items()}
        image, pointcloud, targets = _to_device(image, pointcloud, targets, device)

        out = model(image, pointcloud)
        mask = match_mask_from_heatmap(targets["H"])
        scores = object_nonconformity_scores(out["B"], targets["B"], mask)
        pending_m.append(frame_miscoverage_rate(scores, q_hat))
        pending_ccp.append(float((1.0 - out["S"]).mean().item()))

        if len(pending_m) >= block_size:
            m_block = float(np.mean(pending_m))
            ccp_block = float(np.mean(pending_ccp))
            lam = bettor.next_lambda(ccp_disagreement=ccp_block)
            wealth = wealth_process.step(m_block, lam)
            bettor.update(m_block, ccp_disagreement=ccp_block)
            pending_m, pending_ccp = [], []

        trajectory.append(wealth)

    return trajectory


def operating_curve_per_stream_quantile(
    model: CRAFX_Net,
    alpha: float,
    deltas: Sequence[float],
    onset_specs: Sequence[Tuple[Callable[[], object], float]],
    clear_specs: Sequence[Tuple[Callable[[], object], float]],
    bettor_factory: Callable[[], object],
    block_size: int = 1,
    n_bootstrap: int = 2000,
    rng_seed: int = 20260907,
) -> List[dict]:
    """
    Operating curve where each replicate carries its own quantile.

    This generalizes the marginal case rather than replacing it: passing the
    same pooled quantile in every spec reproduces standard split-conformal
    calibration exactly, so the Mondrian arm and its like-for-like baseline
    run through identical code on identical streams and differ only in which
    quantile each replicate is scored against.

    Reports bootstrap CIs on both detection delay and false-alarm rate,
    since point estimates over a handful of replicates were what let an
    n=1 evaluation masquerade as n=5.
    """
    rng = np.random.default_rng(rng_seed)

    onset_runs = [
        (wealth_trajectory_with_quantile(model, factory(), q, alpha, bettor_factory, block_size),
         factory().onset_frame)
        for factory, q in onset_specs
    ]
    clear_trajectories = [
        wealth_trajectory_with_quantile(model, factory(), q, alpha, bettor_factory, block_size)
        for factory, q in clear_specs
    ]

    curve = []
    for delta in deltas:
        delays, n_censored = [], 0
        for trajectory, onset_frame in onset_runs:
            alarm_time = alarm_time_from_trajectory(trajectory, delta)
            if alarm_time is None:
                n_censored += 1
            else:
                delays.append(alarm_time - onset_frame)

        alarmed = [
            1.0 if alarm_time_from_trajectory(t, delta) is not None else 0.0
            for t in clear_trajectories
        ]
        fa_rate = float(np.mean(alarmed)) if alarmed else 0.0

        fa_ci = _bootstrap_ci(np.asarray(alarmed), rng, n_bootstrap) if alarmed else (0.0, 0.0)
        delay_ci = (
            _bootstrap_ci(np.asarray(delays, dtype=float), rng, n_bootstrap)
            if delays else (None, None)
        )

        curve.append(
            {
                "delta": delta,
                "false_alarm_rate": fa_rate,
                "false_alarm_ci": list(fa_ci),
                "mean_detection_delay": float(np.mean(delays)) if delays else None,
                "detection_delay_ci": list(delay_ci),
                "n_censored": n_censored,
                "n_onset_replicates": len(onset_runs),
                "n_clear_replicates": len(clear_trajectories),
                "controls_false_alarms": bool(fa_rate <= delta),
            }
        )
    return curve


def _bootstrap_ci(values: np.ndarray, rng, n_bootstrap: int, level: float = 95.0):
    if values.size == 0:
        return (None, None)
    draws = np.array([rng.choice(values, size=values.size, replace=True).mean()
                      for _ in range(n_bootstrap)])
    tail = (100.0 - level) / 2.0
    return (float(np.percentile(draws, tail)), float(np.percentile(draws, 100.0 - tail)))
