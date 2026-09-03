"""
Per-BEV-cell e-processes with online multiplicity control (plan.md claim 2):
a grid of independent wealth processes, one per BEV cell (or cell-cluster),
producing a live, spatially localized "where is perception currently
untrustworthy" map instead of one global per-frame alarm.

Two multiplicity-control modes are provided, matching plan.md's "Baselines
to beat":

- `"bonferroni"`: each cell gets a fixed delta/n_cells alarm budget — valid
  (family-wise error controlled at delta) but conservative, and once a cell
  alarms it stays alarmed (a permanent per-cell detection time, used for the
  detection-delay evaluation).
- `"ebh"`: at every timestep, treat each cell's current wealth as an e-value
  (E_H0[K_t] <= 1 for a nonnegative supermartingale evaluated at any fixed
  time t) and apply the e-BH procedure (Wang & Ramdas, "True and false
  discoveries with e-values", JRSS-B 2022) across cells at target FDR level
  delta, giving a live, non-sticky flagged-cell map with online FDR control.
"""
from typing import Callable, List, Optional

import numpy as np

from .betting import SequentialTester


def e_bh_reject(e_values: np.ndarray, q: float) -> np.ndarray:
    """
    Batch e-BH procedure (Wang & Ramdas, 2022): given e-values for n
    hypotheses (here: n BEV cells at a fixed timestep), reject the set

        R = {i : e_i >= n / (k* q)}

    where k* is the largest k such that the k-th largest e-value satisfies
    e_(k) >= n / (k q). Controls FDR at level q under arbitrary dependence
    across the e-values.

    Args:
        e_values: 1D array of nonnegative e-values, one per cell.
        q: target FDR level in (0, 1).

    Returns:
        Boolean mask, True where the cell is rejected (flagged untrustworthy).
    """
    n = e_values.shape[0]
    if n == 0:
        return np.zeros(0, dtype=bool)
    order = np.argsort(e_values)[::-1]
    sorted_e = e_values[order]

    k_star = 0
    for k in range(1, n + 1):
        if sorted_e[k - 1] >= n / (k * q):
            k_star = k

    rejected = np.zeros(n, dtype=bool)
    if k_star > 0:
        rejected[order[:k_star]] = True
    return rejected


class SpatialEProcessGrid:
    """
    Grid of per-cell wealth processes over an (n_cells_h, n_cells_w) BEV
    cell-cluster layout.

    Args:
        alpha: target per-object miscoverage level (shared across cells).
        delta: false-alarm budget — the Bonferroni family-wise budget, or the
            e-BH FDR level, depending on `correction`.
        n_cells_h, n_cells_w: grid shape.
        bettor_factory: zero-arg callable returning a fresh bettor instance
            (e.g. `lambda: AGRAPABettor(alpha)` or a `CCPInformedBettor`) —
            called once per cell so each cell tracks its own betting state.
        correction: "bonferroni" or "ebh".
    """

    def __init__(
        self,
        alpha: float,
        delta: float,
        n_cells_h: int,
        n_cells_w: int,
        bettor_factory: Callable[[], object],
        correction: str = "ebh",
    ):
        if correction not in ("bonferroni", "ebh"):
            raise ValueError(f"unknown correction mode: {correction}")
        self.alpha = alpha
        self.delta = delta
        self.n_cells_h = n_cells_h
        self.n_cells_w = n_cells_w
        self.n_cells = n_cells_h * n_cells_w
        self.correction = correction

        cell_delta = delta / self.n_cells if correction == "bonferroni" else delta
        self.testers: List[SequentialTester] = [
            SequentialTester(alpha, cell_delta, bettor_factory()) for _ in range(self.n_cells)
        ]
        self._alarmed_at: List[Optional[int]] = [None] * self.n_cells
        self._t = 0

    def step(
        self,
        cell_miscoverage: np.ndarray,
        cell_ccp_disagreement: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Advance every cell's wealth process by one frame.

        Args:
            cell_miscoverage: (n_cells_h, n_cells_w) array of per-cell m(t).
            cell_ccp_disagreement: optional (n_cells_h, n_cells_w) array of
                per-cell CCP disagreement (mean(1 - S)) for covariate-informed
                bettors; ignored by covariate-blind ones.

        Returns:
            (n_cells_h, n_cells_w) boolean "currently untrustworthy" map.
        """
        flat_m = cell_miscoverage.reshape(-1)
        flat_ccp = (
            cell_ccp_disagreement.reshape(-1)
            if cell_ccp_disagreement is not None
            else np.zeros_like(flat_m)
        )

        wealths = np.empty(self.n_cells)
        for i, tester in enumerate(self.testers):
            cov = {"ccp_disagreement": float(flat_ccp[i])}
            wealths[i] = tester.step(float(flat_m[i]), self._t, **cov)
            if self._alarmed_at[i] is None and tester.alarm_time is not None:
                self._alarmed_at[i] = tester.alarm_time

        if self.correction == "bonferroni":
            flagged = np.array([a is not None for a in self._alarmed_at])
        else:
            flagged = e_bh_reject(wealths, self.delta)

        self._t += 1
        return flagged.reshape(self.n_cells_h, self.n_cells_w)

    def first_alarm_times(self) -> np.ndarray:
        """
        Per-cell first alarm time (Bonferroni sense: first t the cell's own
        wealth crossed its budget), NaN where the cell never alarmed. Used
        for the detection-delay evaluation regardless of `correction` mode.
        """
        return np.array(
            [np.nan if a is None else a for a in self._alarmed_at]
        ).reshape(self.n_cells_h, self.n_cells_w)


def pool_to_cell_grid(values: np.ndarray, n_cells_h: int, n_cells_w: int) -> np.ndarray:
    """
    Average-pool a (H, W) per-BEV-cell array (e.g. per-cell miscoverage or
    1 - CCP consistency score) down to an (n_cells_h, n_cells_w) cluster
    grid, per plan.md's "cell-cluster" granularity knob.
    """
    h, w = values.shape
    if h % n_cells_h != 0 or w % n_cells_w != 0:
        raise ValueError(
            f"grid shape ({h}, {w}) must be evenly divisible by "
            f"cluster shape ({n_cells_h}, {n_cells_w})"
        )
    block_h, block_w = h // n_cells_h, w // n_cells_w
    reshaped = values.reshape(n_cells_h, block_h, n_cells_w, block_w)
    return reshaped.mean(axis=(1, 3))
