"""
Two-level bootstrap for the nominal-vs-degraded effect.

The previous estimator resampled only the nominal arm. The degraded side
was a single fixed pool of frames, drawn once and reused unchanged for
every session, so all variation across the degraded drives was implicitly
treated as zero. Two consequences:

- the reported intervals were intervals for the nominal mean shifted by a
  constant, not intervals for a two-sample contrast, so "the CI excludes
  zero" did not license "a weather effect exists";
- under marginal (pooled-quantile) calibration the degraded term was
  literally identical in every per-session difference, which is visible in
  the old artifacts as a `degraded_mean` that does not change with the gap.

It also made "session-paired" a misnomer: nothing was paired: one fixed
degraded pool was differenced against each nominal session.

This module resamples **both** levels -- nominal sessions and degraded
drives -- which is the correct unit structure for the design. Degraded
drives, not degraded frames: frames within a drive are ~10 Hz samples of
one scene and are nowhere near independent, and the measured between-drive
standard deviation on CADC (0.057) is the variance component that actually
matters.

Efficiency note: the estimator precomputes a (session x degraded-drive)
matrix of mean miscoverage, so a bootstrap replicate is a pair of index
draws and two means rather than a rescoring pass. Scores are cached
upstream, so the whole sweep is arithmetic.
"""
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from conformal_monitor.calibration import calibrate_quantile, frame_miscoverage_rate

N_BOOTSTRAP_DEFAULT = 3000


def _mean_miscoverage(score_arrays: Sequence[np.ndarray], q: float) -> float:
    if not len(score_arrays):
        return float("nan")
    return float(np.mean([frame_miscoverage_rate(s, q) for s in score_arrays]))


def two_level_effect(
    session_scores: Dict[str, Tuple[List[np.ndarray], List[np.ndarray]]],
    degraded_by_drive: Dict[str, List[np.ndarray]],
    alpha: float,
    n_calibration: int,
    gap: int,
    rng: np.random.Generator,
    calibration_mode: str = "mondrian",
    n_bootstrap: int = N_BOOTSTRAP_DEFAULT,
    min_calibration: int = 10,
    min_monitored: int = 5,
) -> Optional[dict]:
    """
    Effect and CI, resampling nominal sessions and degraded drives.

    Args:
        session_scores: session key -> (calibration_prefix_scores,
            monitored_scores). The caller applies the temporal gap when it
            builds these, or passes full sequences and lets `gap` apply
            here; `gap` is applied here to the monitored side.
        degraded_by_drive: degraded drive key -> that drive's frame scores.
            Keyed by drive precisely so the bootstrap can resample drives.
        calibration_mode: "mondrian" (each session's own quantile) or
            "marginal" (one pooled quantile). Both run over identical
            inputs so the comparison isolates the calibration scheme.

    Returns:
        dict with the point estimate, the two-level percentile CI, the
        nominal-only CI for comparison, and the per-level variance
        contributions -- or None when no session has enough frames.
    """
    if calibration_mode not in ("mondrian", "marginal"):
        raise ValueError(f"unknown calibration_mode: {calibration_mode}")

    pooled_q = None
    if calibration_mode == "marginal":
        pooled = [s for cal, _ in session_scores.values() for s in cal[:n_calibration]]
        if not pooled:
            return None
        pooled_q = calibrate_quantile(np.concatenate(pooled), alpha)

    session_keys: List[str] = []
    nominal_means: List[float] = []
    quantiles: List[float] = []
    for key, (calibration, monitored) in session_scores.items():
        calibration_part = calibration[:n_calibration]
        monitored_part = monitored[gap:]
        if len(calibration_part) < min_calibration or len(monitored_part) < min_monitored:
            continue
        q = pooled_q if pooled_q is not None else calibrate_quantile(
            np.concatenate(calibration_part), alpha
        )
        session_keys.append(key)
        quantiles.append(q)
        nominal_means.append(_mean_miscoverage(monitored_part, q))

    drive_keys = sorted(degraded_by_drive)
    if not session_keys or not drive_keys:
        return None

    # (session x degraded drive) mean miscoverage, each degraded drive
    # scored under each session's own quantile.
    grid = np.empty((len(session_keys), len(drive_keys)))
    for i, q in enumerate(quantiles):
        for j, drive in enumerate(drive_keys):
            grid[i, j] = _mean_miscoverage(degraded_by_drive[drive], q)

    nominal_means = np.asarray(nominal_means)
    point_effect = float(grid.mean(axis=1).mean() - nominal_means.mean())

    n_sessions, n_drives = grid.shape
    two_level = np.empty(n_bootstrap)
    nominal_only = np.empty(n_bootstrap)
    for b in range(n_bootstrap):
        si = rng.integers(0, n_sessions, n_sessions)
        di = rng.integers(0, n_drives, n_drives)
        two_level[b] = grid[np.ix_(si, di)].mean(axis=1).mean() - nominal_means[si].mean()
        # Same draw on the nominal side only, so the two intervals differ
        # solely by whether the degraded arm was resampled.
        nominal_only[b] = grid[si, :].mean(axis=1).mean() - nominal_means[si].mean()

    lo, hi = np.percentile(two_level, [2.5, 97.5])
    lo_n, hi_n = np.percentile(nominal_only, [2.5, 97.5])
    between_session_sd = float(nominal_means.std(ddof=1)) if n_sessions > 1 else float("nan")
    between_drive_sd = float(grid.mean(axis=0).std(ddof=1)) if n_drives > 1 else float("nan")

    return {
        "calibration_mode": calibration_mode,
        "gap": gap,
        "n_sessions": int(n_sessions),
        "n_degraded_drives": int(n_drives),
        "nominal_mean": float(nominal_means.mean()),
        "degraded_mean": float(grid.mean()),
        "effect": point_effect,
        "ci_low": float(lo),
        "ci_high": float(hi),
        "excludes_zero": bool(lo > 0.0 or hi < 0.0),
        "ci_low_nominal_only": float(lo_n),
        "ci_high_nominal_only": float(hi_n),
        "excludes_zero_nominal_only": bool(lo_n > 0.0 or hi_n < 0.0),
        "ci_width_inflation": float((hi - lo) / (hi_n - lo_n)) if hi_n > lo_n else float("nan"),
        "between_session_sd": between_session_sd,
        "between_drive_sd": between_drive_sd,
        "mean_session_quantile": float(np.mean(quantiles)),
    }
