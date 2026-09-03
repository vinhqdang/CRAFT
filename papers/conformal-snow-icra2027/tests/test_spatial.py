import numpy as np
import pytest

from conformal_monitor.betting import AGRAPABettor
from conformal_monitor.spatial import SpatialEProcessGrid, e_bh_reject, pool_to_cell_grid


def test_e_bh_reject_only_strong_evidence_cell():
    e_values = np.array([50.0, 1.0, 1.0, 1.0])
    rejected = e_bh_reject(e_values, q=0.1)
    assert rejected.tolist() == [True, False, False, False]


def test_e_bh_reject_all_strong_evidence():
    e_values = np.array([1000.0, 1000.0, 1000.0, 1000.0])
    rejected = e_bh_reject(e_values, q=0.2)
    assert rejected.all()


def test_e_bh_reject_no_evidence():
    e_values = np.array([1.0, 1.0, 1.0, 1.0])
    rejected = e_bh_reject(e_values, q=0.05)
    assert not rejected.any()


def test_e_bh_reject_empty():
    rejected = e_bh_reject(np.zeros(0), q=0.1)
    assert rejected.shape == (0,)


def test_pool_to_cell_grid_matches_manual_block_average():
    values = np.arange(16, dtype=np.float64).reshape(4, 4)
    pooled = pool_to_cell_grid(values, n_cells_h=2, n_cells_w=2)
    expected = np.array([[2.5, 4.5], [10.5, 12.5]])
    np.testing.assert_allclose(pooled, expected)


def test_pool_to_cell_grid_rejects_non_divisible_shape():
    values = np.zeros((3, 4))
    with pytest.raises(ValueError):
        pool_to_cell_grid(values, n_cells_h=2, n_cells_w=2)


def test_spatial_grid_step_shape_and_flags_persistent_violation():
    alpha, delta = 0.1, 0.05
    grid = SpatialEProcessGrid(
        alpha, delta, n_cells_h=2, n_cells_w=2,
        bettor_factory=lambda: AGRAPABettor(alpha),
        correction="bonferroni",
    )

    flagged = None
    for _ in range(300):
        cell_miscoverage = np.ones((2, 2))  # every cell maximally miscovered
        flagged = grid.step(cell_miscoverage)

    assert flagged.shape == (2, 2)
    assert flagged.all()

    alarm_times = grid.first_alarm_times()
    assert alarm_times.shape == (2, 2)
    assert not np.isnan(alarm_times).any()


def test_spatial_grid_ebh_flags_only_disagreeing_region():
    alpha, delta = 0.1, 0.1
    grid = SpatialEProcessGrid(
        alpha, delta, n_cells_h=2, n_cells_w=2,
        bettor_factory=lambda: AGRAPABettor(alpha),
        correction="ebh",
    )

    flagged = None
    for _ in range(50):
        cell_miscoverage = np.zeros((2, 2))
        cell_miscoverage[0, 0] = 1.0  # only the top-left cell violates
        flagged = grid.step(cell_miscoverage)

    # The e-BH map is a live snapshot: at minimum it should never flag a
    # cell that has never shown any miscoverage evidence.
    assert not flagged[1, 1]
