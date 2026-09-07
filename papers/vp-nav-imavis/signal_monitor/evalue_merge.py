"""
Global monitoring by merging per-cell e-values, instead of averaging
miscoverage across the grid before betting.

The existing global monitor
(`conformal_monitor.evaluate.compute_global_wealth_trajectory`) computes one
frame-level miscoverage rate m(t) over the whole BEV grid and drives a
single wealth process with it. Snow degrades some regions hard -- near-range
cells, where airborne snowfall produces spurious returns -- while most of
the grid stays clean, so a whole-frame average dilutes the very signal the
monitor is trying to detect: the average barely moves even when a handful
of cells move a lot.

This module keeps the spatial decomposition the paper already builds
(`conformal_monitor.spatial`) but changes *where* the averaging happens.
Each cell runs its own wealth process on its own local miscoverage, so each
cell amplifies its own local evidence multiplicatively over time; only then
are the per-cell wealth values combined, by a weighted average. Averaging
evidence at the e-value level after amplification is a different operation
from averaging the raw signal away beforehand, and it is exactly as valid:
each cell's wealth K_t^(c) is an e-value at any fixed t, and a convex
combination of e-values is an e-value (Vovk & Wang, 2021), so Ville's
inequality applies to the merged process with the same 1/delta threshold.

Two differences from `SpatialEProcessGrid` are deliberate:

- it produces a single global alarm statistic (a merged e-process) rather
  than a per-cell flagged map, so it is directly comparable against the
  existing global monitor's detection delay;
- per-cell miscoverage is computed as the fraction of *matched* cells in
  the block that are miscovered, rather than the mean of a masked indicator
  over all cells in the block. The latter (what `run_spatial_monitor`'s
  `pool_to_cell_grid(miscovered)` computes) is diluted by however many
  empty cells a block happens to contain, which reintroduces the averaging
  problem at block level. `conformal_monitor` is not modified; the correct
  pooling is computed here.
"""
from typing import Callable, Dict, List, Optional

import numpy as np
import torch

from craf_x.models.crafx_net import CRAFX_Net

from conformal_monitor.betting import WealthProcess
from conformal_monitor.evaluate import _to_device, alarm_time_from_trajectory, match_mask_from_heatmap


def pool_cell_miscoverage(
    miscovered: np.ndarray, matched: np.ndarray, n_cells_h: int, n_cells_w: int
) -> np.ndarray:
    """
    Per-cell miscoverage rate: within each block, the fraction of matched
    cells that are miscovered. Blocks containing no matched cell carry no
    evidence and are defined to have rate 0.0, matching
    `conformal_monitor.calibration.frame_miscoverage_rate`'s own convention
    for an object-free frame.

    Args:
        miscovered: (H, W) indicator, 1 where a matched cell is miscovered.
        matched: (H, W) indicator, 1 where a ground-truth object is present.
        n_cells_h, n_cells_w: block-grid shape.

    Returns:
        (n_cells_h, n_cells_w) array of per-cell miscoverage rates.
    """
    h, w = miscovered.shape
    if h % n_cells_h != 0 or w % n_cells_w != 0:
        raise ValueError(
            f"grid shape ({h}, {w}) must be evenly divisible by "
            f"cluster shape ({n_cells_h}, {n_cells_w})"
        )
    block_h, block_w = h // n_cells_h, w // n_cells_w

    miscovered_sum = miscovered.reshape(n_cells_h, block_h, n_cells_w, block_w).sum(axis=(1, 3))
    matched_sum = matched.reshape(n_cells_h, block_h, n_cells_w, block_w).sum(axis=(1, 3))
    return np.divide(
        miscovered_sum, matched_sum, out=np.zeros_like(miscovered_sum, dtype=np.float64),
        where=matched_sum > 0,
    )


def merge_e_values(e_values: np.ndarray, weights: Optional[np.ndarray] = None) -> float:
    """
    Merge per-cell e-values into one, by weighted average (Vovk & Wang,
    2021). With `weights=None` this is the uniform average.

    The merged value is an e-value whenever the inputs are, under arbitrary
    dependence between them -- which matters here, since neighbouring BEV
    cells are anything but independent.
    """
    if e_values.size == 0:
        return 1.0
    if weights is None:
        return float(np.mean(e_values))

    if weights.shape != e_values.shape:
        raise ValueError(f"weights shape {weights.shape} != e_values shape {e_values.shape}")
    if np.any(weights < 0):
        raise ValueError("weights must be non-negative")
    if not np.isclose(weights.sum(), 1.0):
        raise ValueError(f"weights must sum to 1, got {weights.sum()}")
    return float(np.sum(weights * e_values))


@torch.no_grad()
def compute_merged_evalue_trajectory(
    model: CRAFX_Net,
    stream,
    q_hat: float,
    alpha: float,
    bettor_factory: Callable[[], object],
    n_cells_h: int = 4,
    n_cells_w: int = 4,
) -> Dict[str, List[float]]:
    """
    Run per-cell wealth processes over the stream and return both the merged
    global e-process trajectory and, for reference, the whole-frame
    trajectory the existing monitor would have produced from the same
    forward passes.

    Returning both from one pass makes the comparison exact: the two
    trajectories see identical model outputs, so any difference in
    detection delay is attributable to where the averaging happens and
    nothing else.
    """
    model.eval()
    device = next(model.parameters()).device

    n_cells = n_cells_h * n_cells_w
    cell_bettors = [bettor_factory() for _ in range(n_cells)]
    cell_processes = [
        WealthProcess(alpha, lambda_max=b.lambda_max) for b in cell_bettors
    ]
    global_bettor = bettor_factory()
    global_process = WealthProcess(alpha, lambda_max=global_bettor.lambda_max)

    merged_trajectory: List[float] = []
    global_trajectory: List[float] = []

    for t in range(len(stream)):
        sample = stream[t]
        image = sample["image"].unsqueeze(0)
        pointcloud = sample["pointcloud"].unsqueeze(0)
        targets = {k: v.unsqueeze(0) for k, v in sample["targets"].items()}
        image, pointcloud, targets = _to_device(image, pointcloud, targets, device)

        out = model(image, pointcloud)
        match_mask = match_mask_from_heatmap(targets["H"])
        residual = torch.abs(out["B"] - targets["B"]).sum(dim=1, keepdim=True)
        miscovered = ((residual > q_hat).float() * match_mask)[0, 0].cpu().numpy()
        matched = match_mask[0, 0].cpu().numpy()

        cell_m = pool_cell_miscoverage(miscovered, matched, n_cells_h, n_cells_w).reshape(-1)
        cell_ccp = (
            (1.0 - out["S"])[0, 0]
            .reshape(n_cells_h, matched.shape[0] // n_cells_h, n_cells_w, matched.shape[1] // n_cells_w)
            .mean(dim=(1, 3))
            .cpu()
            .numpy()
            .reshape(-1)
        )

        wealths = np.empty(n_cells)
        for i in range(n_cells):
            covariates = {"ccp_disagreement": float(cell_ccp[i])}
            lam = cell_bettors[i].next_lambda(**covariates)
            wealths[i] = cell_processes[i].step(float(cell_m[i]), lam)
            cell_bettors[i].update(float(cell_m[i]), **covariates)
        merged_trajectory.append(merge_e_values(wealths))

        # Whole-frame reference, from the same forward pass.
        n_matched = matched.sum()
        m_t = float(miscovered.sum() / n_matched) if n_matched > 0 else 0.0
        ccp_frame = float((1.0 - out["S"]).mean().item())
        lam = global_bettor.next_lambda(ccp_disagreement=ccp_frame)
        global_trajectory.append(global_process.step(m_t, lam))
        global_bettor.update(m_t, ccp_disagreement=ccp_frame)

    return {"merged": merged_trajectory, "whole_frame": global_trajectory}


def merged_evalue_operating_curve(
    model: CRAFX_Net,
    q_hat: float,
    alpha: float,
    deltas: List[float],
    onset_stream_factory: Callable[[], object],
    clear_stream_factory: Callable[[], object],
    bettor_factory: Callable[[], object],
    n_cells_h: int = 4,
    n_cells_w: int = 4,
    n_onset_replicates: int = 5,
    n_clear_replicates: int = 5,
) -> Dict[str, List[dict]]:
    """
    Operating curve for the merged-e-value monitor and, side by side from
    the same forward passes, the whole-frame monitor -- mirroring
    `conformal_monitor.evaluate.operating_curve`'s protocol and output
    format so the numbers drop straight into the same comparison.
    """
    onset_runs = []
    for _ in range(n_onset_replicates):
        stream = onset_stream_factory()
        trajectories = compute_merged_evalue_trajectory(
            model, stream, q_hat, alpha, bettor_factory, n_cells_h, n_cells_w
        )
        onset_runs.append((trajectories, stream.onset_frame))

    clear_runs = [
        compute_merged_evalue_trajectory(
            model, clear_stream_factory(), q_hat, alpha, bettor_factory, n_cells_h, n_cells_w
        )
        for _ in range(n_clear_replicates)
    ]

    curves: Dict[str, List[dict]] = {}
    for name in ("merged", "whole_frame"):
        curve = []
        for delta in deltas:
            delays, n_censored = [], 0
            for trajectories, onset_frame in onset_runs:
                alarm_time = alarm_time_from_trajectory(trajectories[name], delta)
                if alarm_time is None:
                    n_censored += 1
                else:
                    delays.append(alarm_time - onset_frame)

            n_alarmed = sum(
                1 for r in clear_runs if alarm_time_from_trajectory(r[name], delta) is not None
            )
            curve.append(
                {
                    "delta": delta,
                    "false_alarm_rate": n_alarmed / n_clear_replicates if n_clear_replicates else 0.0,
                    "mean_detection_delay": float(np.mean(delays)) if delays else None,
                    "n_censored": n_censored,
                }
            )
        curves[name] = curve
    return curves
