"""
Detection-delay-vs-false-alarm evaluation harness (plan.md "Next steps":
"design the detection-delay-vs-false-alarm operating-curve evaluation").

Wires together craf_x's `CRAFX_Net` (for detections and the CCP consistency
score), `calibration.py` (split conformal calibration), `betting.py` /
`spatial.py` (the sequential monitor, global or per-cell), and
`corruption.py` (the synthetic weather-onset stream) into one pipeline:
calibrate on clear weather, run the monitor over an onset scene, and report
whether/when it alarmed relative to the true onset frame.
"""
from dataclasses import dataclass
from typing import Callable, List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

from craf_x.models.crafx_net import CRAFX_Net

from .betting import WealthProcess
from .calibration import calibrate_quantile, object_nonconformity_scores, frame_miscoverage_rate
from .corruption import WeatherOnsetStream
from .spatial import SpatialEProcessGrid, pool_to_cell_grid


def match_mask_from_heatmap(heatmap: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    """(B, C, H, W) heatmap -> (B, 1, H, W) binary mask of object-occupied cells."""
    return (heatmap.amax(dim=1, keepdim=True) > threshold).float()


@torch.no_grad()
def calibrate_on_clear_weather(model: CRAFX_Net, calibration_dataset, alpha: float, batch_size: int = 8) -> float:
    """
    Runs the model over a held-out, clear-weather calibration set and
    returns the calibrated nonconformity quantile q_hat.
    """
    model.eval()
    loader = DataLoader(calibration_dataset, batch_size=batch_size)
    all_scores: List[np.ndarray] = []

    for batch in loader:
        out = model(batch["image"], batch["pointcloud"])
        match_mask = match_mask_from_heatmap(batch["targets"]["H"])
        scores = object_nonconformity_scores(out["B"], batch["targets"]["B"], match_mask)
        all_scores.append(scores)

    calibration_scores = np.concatenate(all_scores) if all_scores else np.zeros(0)
    return calibrate_quantile(calibration_scores, alpha)


@dataclass
class MonitorRun:
    alarm_time: Optional[int]
    onset_frame: int
    detection_delay: Optional[int]  # None if never alarmed, negative if alarmed before onset
    wealth_trajectory: List[float]


@torch.no_grad()
def compute_global_wealth_trajectory(
    model: CRAFX_Net,
    stream: WeatherOnsetStream,
    q_hat: float,
    alpha: float,
    bettor_factory: Callable[[], object],
) -> List[float]:
    """
    Runs the model over the full stream and drives a single global
    (whole-frame) wealth process to completion. Deliberately
    delta-independent (delta only gates the post-hoc alarm threshold 1/delta
    applied to this trajectory, via `alarm_time_from_trajectory`), so a
    sweep over multiple deltas can reuse one trajectory instead of rerunning
    the model once per delta.
    """
    model.eval()
    bettor = bettor_factory()
    wealth_process = WealthProcess(alpha, lambda_max=bettor.lambda_max)
    trajectory: List[float] = []

    for t in range(len(stream)):
        sample = stream[t]
        image = sample["image"].unsqueeze(0)
        pointcloud = sample["pointcloud"].unsqueeze(0)
        targets = {k: v.unsqueeze(0) for k, v in sample["targets"].items()}

        out = model(image, pointcloud)
        match_mask = match_mask_from_heatmap(targets["H"])
        scores = object_nonconformity_scores(out["B"], targets["B"], match_mask)
        m_t = frame_miscoverage_rate(scores, q_hat)
        ccp_disagreement = float((1.0 - out["S"]).mean().item())

        lam = bettor.next_lambda(ccp_disagreement=ccp_disagreement)
        wealth = wealth_process.step(m_t, lam)
        bettor.update(m_t, ccp_disagreement=ccp_disagreement)
        trajectory.append(wealth)

    return trajectory


def alarm_time_from_trajectory(trajectory: List[float], delta: float) -> Optional[int]:
    """First t with trajectory[t] >= 1/delta, or None if the process never alarms."""
    threshold = 1.0 / delta
    for t, wealth in enumerate(trajectory):
        if wealth >= threshold:
            return t
    return None


def run_global_monitor(
    model: CRAFX_Net,
    stream: WeatherOnsetStream,
    q_hat: float,
    alpha: float,
    delta: float,
    bettor_factory: Callable[[], object],
) -> MonitorRun:
    """
    Runs a single global (whole-frame) sequential monitor over an onset
    stream and reports the alarm time, if any, relative to the stream's
    true onset frame. For a sweep over multiple deltas on the same stream,
    call `compute_global_wealth_trajectory` once and reuse it with
    `alarm_time_from_trajectory` instead of calling this per delta.
    """
    full_trajectory = compute_global_wealth_trajectory(model, stream, q_hat, alpha, bettor_factory)
    alarm_time = alarm_time_from_trajectory(full_trajectory, delta)
    trajectory = full_trajectory[: alarm_time + 1] if alarm_time is not None else full_trajectory

    delay = None if alarm_time is None else alarm_time - stream.onset_frame
    return MonitorRun(alarm_time, stream.onset_frame, delay, trajectory)


@torch.no_grad()
def run_spatial_monitor(
    model: CRAFX_Net,
    stream: WeatherOnsetStream,
    q_hat: float,
    alpha: float,
    delta: float,
    bettor_factory: Callable[[], object],
    n_cells_h: int = 4,
    n_cells_w: int = 4,
    correction: str = "ebh",
) -> List[np.ndarray]:
    """
    Runs the per-BEV-cell spatial monitor (plan.md claim 2) over an onset
    stream. Returns the sequence of per-frame "untrustworthy cell" boolean
    maps (n_cells_h, n_cells_w).
    """
    model.eval()
    grid = SpatialEProcessGrid(alpha, delta, n_cells_h, n_cells_w, bettor_factory, correction)
    flagged_maps: List[np.ndarray] = []

    for t in range(len(stream)):
        sample = stream[t]
        image = sample["image"].unsqueeze(0)
        pointcloud = sample["pointcloud"].unsqueeze(0)
        targets = {k: v.unsqueeze(0) for k, v in sample["targets"].items()}

        out = model(image, pointcloud)
        match_mask = match_mask_from_heatmap(targets["H"])  # (1, 1, H, W)

        residual = torch.abs(out["B"] - targets["B"]).sum(dim=1, keepdim=True)  # (1, 1, H, W)
        covered = (residual <= q_hat).float()
        miscovered = (1.0 - covered) * match_mask  # only meaningful at matched cells
        h, w = miscovered.shape[-2:]

        cell_miscoverage = pool_to_cell_grid(miscovered[0, 0].cpu().numpy(), n_cells_h, n_cells_w)
        cell_ccp_disagreement = pool_to_cell_grid((1.0 - out["S"])[0, 0].cpu().numpy(), n_cells_h, n_cells_w)

        flagged_maps.append(grid.step(cell_miscoverage, cell_ccp_disagreement))

    return flagged_maps


def false_alarm_rate(
    model: CRAFX_Net,
    clear_stream_factory: Callable[[], WeatherOnsetStream],
    q_hat: float,
    alpha: float,
    delta: float,
    bettor_factory: Callable[[], object],
    n_replicates: int = 5,
) -> float:
    """
    Empirical false-alarm rate under stationary clear weather: fraction of
    independent clear-weather-only replicate scenes on which the global
    monitor alarms at all. `clear_stream_factory` should build a
    `WeatherOnsetStream` with `severity_max=0.0` (severity stays 0
    throughout regardless of `onset_frame`, which must still be a valid
    index into the stream).
    """
    n_alarmed = 0
    for _ in range(n_replicates):
        stream = clear_stream_factory()
        run = run_global_monitor(model, stream, q_hat, alpha, delta, bettor_factory)
        if run.alarm_time is not None:
            n_alarmed += 1
    return n_alarmed / n_replicates


def operating_curve(
    model: CRAFX_Net,
    q_hat: float,
    alpha: float,
    deltas: List[float],
    onset_stream_factory: Callable[[], WeatherOnsetStream],
    clear_stream_factory: Callable[[], WeatherOnsetStream],
    bettor_factory: Callable[[], object],
    n_onset_replicates: int = 5,
    n_clear_replicates: int = 5,
) -> List[dict]:
    """
    Sweeps the false-alarm budget `delta` and reports (false_alarm_rate,
    mean_detection_delay) pairs — the operating curve compared across
    bettors (covariate-blind vs. CCP-informed) in plan.md's evaluation plan.
    Runs that never alarm are excluded from the mean delay and reported
    separately as `n_censored`.

    Each replicate's wealth trajectory is delta-independent (see
    `compute_global_wealth_trajectory`), so it is computed once per
    replicate here and reused across the whole `deltas` sweep — this also
    means every delta's point is evaluated on the exact same replicate
    streams, a paired comparison rather than independently resampled ones.
    """
    onset_runs = []
    for _ in range(n_onset_replicates):
        stream = onset_stream_factory()
        trajectory = compute_global_wealth_trajectory(model, stream, q_hat, alpha, bettor_factory)
        onset_runs.append((trajectory, stream.onset_frame))

    clear_trajectories = [
        compute_global_wealth_trajectory(model, clear_stream_factory(), q_hat, alpha, bettor_factory)
        for _ in range(n_clear_replicates)
    ]

    curve = []
    for delta in deltas:
        delays = []
        n_censored = 0
        for trajectory, onset_frame in onset_runs:
            alarm_time = alarm_time_from_trajectory(trajectory, delta)
            if alarm_time is None:
                n_censored += 1
            else:
                delays.append(alarm_time - onset_frame)

        n_alarmed = sum(
            1 for trajectory in clear_trajectories if alarm_time_from_trajectory(trajectory, delta) is not None
        )
        fa_rate = n_alarmed / n_clear_replicates if n_clear_replicates else 0.0

        curve.append(
            {
                "delta": delta,
                "false_alarm_rate": fa_rate,
                "mean_detection_delay": float(np.mean(delays)) if delays else None,
                "n_censored": n_censored,
            }
        )
    return curve
