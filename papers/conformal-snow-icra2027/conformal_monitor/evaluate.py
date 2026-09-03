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

from .betting import SequentialTester
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
    true onset frame.
    """
    model.eval()
    tester = SequentialTester(alpha, delta, bettor_factory())
    wealth_trajectory: List[float] = []

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

        wealth = tester.step(m_t, t, ccp_disagreement=ccp_disagreement)
        wealth_trajectory.append(wealth)
        if tester.alarm_time is not None:
            break

    delay = None if tester.alarm_time is None else tester.alarm_time - stream.onset_frame
    return MonitorRun(tester.alarm_time, stream.onset_frame, delay, wealth_trajectory)


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
    """
    curve = []
    for delta in deltas:
        delays = []
        n_censored = 0
        for _ in range(n_onset_replicates):
            stream = onset_stream_factory()
            run = run_global_monitor(model, stream, q_hat, alpha, delta, bettor_factory)
            if run.detection_delay is None:
                n_censored += 1
            else:
                delays.append(run.detection_delay)

        fa_rate = false_alarm_rate(
            model, clear_stream_factory, q_hat, alpha, delta, bettor_factory, n_clear_replicates
        )
        curve.append(
            {
                "delta": delta,
                "false_alarm_rate": fa_rate,
                "mean_detection_delay": float(np.mean(delays)) if delays else None,
                "n_censored": n_censored,
            }
        )
    return curve
