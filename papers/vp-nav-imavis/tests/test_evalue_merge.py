import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "conformal-snow-icra2027")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pytest
import torch

from craf_x.config import CRAFXConfig
from craf_x.datasets.nuscenes_mock import NuScenesMockDataset
from craf_x.models.crafx_net import CRAFX_Net

from conformal_monitor.betting import AGRAPABettor
from conformal_monitor.corruption import WeatherOnsetStream
from conformal_monitor.evaluate import calibrate_on_clear_weather

from signal_monitor.evalue_merge import (
    compute_merged_evalue_trajectory,
    merge_e_values,
    merged_evalue_operating_curve,
    pool_cell_miscoverage,
)

CONFORMAL_ALPHA = 0.2


def _build_model():
    return CRAFX_Net(CRAFXConfig(bev_h=32, bev_w=32))


def test_pool_cell_miscoverage_normalizes_by_matched_cells_not_block_size():
    # A block with one matched cell that is miscovered has rate 1.0, not
    # 1/block_size. This is the dilution the module exists to avoid.
    miscovered = np.zeros((4, 4))
    matched = np.zeros((4, 4))
    matched[0, 0] = 1.0
    miscovered[0, 0] = 1.0

    rates = pool_cell_miscoverage(miscovered, matched, 2, 2)
    assert rates[0, 0] == pytest.approx(1.0)
    assert rates[0, 1] == pytest.approx(0.0)


def test_pool_cell_miscoverage_is_zero_for_blocks_with_no_objects():
    miscovered = np.zeros((4, 4))
    matched = np.zeros((4, 4))

    rates = pool_cell_miscoverage(miscovered, matched, 2, 2)
    assert np.all(rates == 0.0)
    assert not np.any(np.isnan(rates))  # no 0/0 leaking through


def test_pool_cell_miscoverage_computes_a_real_fraction():
    matched = np.ones((4, 4))
    miscovered = np.zeros((4, 4))
    miscovered[0, 0] = 1.0
    miscovered[0, 1] = 1.0  # 2 of the 4 cells in the top-left block

    rates = pool_cell_miscoverage(miscovered, matched, 2, 2)
    assert rates[0, 0] == pytest.approx(0.5)


def test_pool_cell_miscoverage_rejects_indivisible_grid():
    with pytest.raises(ValueError):
        pool_cell_miscoverage(np.zeros((5, 5)), np.ones((5, 5)), 2, 2)


def test_merge_e_values_is_the_uniform_average_by_default():
    e_values = np.array([0.5, 1.5, 4.0])
    assert merge_e_values(e_values) == pytest.approx(2.0)


def test_merge_e_values_lies_between_its_inputs():
    # The bound that makes the merged process a valid e-process.
    rng = np.random.default_rng(0)
    for _ in range(20):
        e_values = rng.uniform(0.0, 50.0, size=8)
        merged = merge_e_values(e_values)
        assert e_values.min() <= merged <= e_values.max()


def test_merge_e_values_honours_weights_and_validates_them():
    e_values = np.array([0.0, 10.0])
    assert merge_e_values(e_values, np.array([1.0, 0.0])) == pytest.approx(0.0)
    assert merge_e_values(e_values, np.array([0.0, 1.0])) == pytest.approx(10.0)

    with pytest.raises(ValueError):
        merge_e_values(e_values, np.array([0.5, 0.9]))  # does not sum to 1
    with pytest.raises(ValueError):
        merge_e_values(e_values, np.array([1.5, -0.5]))  # negative weight


def test_merged_trajectory_starts_at_one_and_stays_non_negative():
    model = _build_model()
    calibration = NuScenesMockDataset(num_samples=4)
    q_hat = calibrate_on_clear_weather(model, calibration, CONFORMAL_ALPHA, batch_size=2)
    stream = WeatherOnsetStream(NuScenesMockDataset(num_samples=4), scene_length=5, onset_frame=2, ramp_length=1)

    trajectories = compute_merged_evalue_trajectory(
        model, stream, q_hat, CONFORMAL_ALPHA, lambda: AGRAPABettor(CONFORMAL_ALPHA),
        n_cells_h=4, n_cells_w=4,
    )

    assert set(trajectories.keys()) == {"merged", "whole_frame"}
    for name, trajectory in trajectories.items():
        assert len(trajectory) == 5
        assert all(w >= 0.0 for w in trajectory), name


def test_single_cell_grid_reduces_to_the_whole_frame_monitor():
    # Degenerate case: with a 1x1 "grid" the merged process bets on exactly
    # the same frame-level miscoverage as the whole-frame monitor, so the
    # two trajectories must coincide. This pins down that the difference
    # between them comes from the spatial decomposition and nothing else.
    model = _build_model()
    calibration = NuScenesMockDataset(num_samples=4)
    q_hat = calibrate_on_clear_weather(model, calibration, CONFORMAL_ALPHA, batch_size=2)
    stream = WeatherOnsetStream(NuScenesMockDataset(num_samples=4), scene_length=5, onset_frame=2, ramp_length=1)

    trajectories = compute_merged_evalue_trajectory(
        model, stream, q_hat, CONFORMAL_ALPHA, lambda: AGRAPABettor(CONFORMAL_ALPHA),
        n_cells_h=1, n_cells_w=1,
    )

    assert trajectories["merged"] == pytest.approx(trajectories["whole_frame"])


def test_merged_operating_curve_reports_both_monitors():
    model = _build_model()
    calibration = NuScenesMockDataset(num_samples=4)
    q_hat = calibrate_on_clear_weather(model, calibration, CONFORMAL_ALPHA, batch_size=2)

    def make_onset_stream():
        return WeatherOnsetStream(NuScenesMockDataset(num_samples=4), scene_length=5, onset_frame=2, ramp_length=1)

    def make_clear_stream():
        return WeatherOnsetStream(
            NuScenesMockDataset(num_samples=4), scene_length=5, onset_frame=2, ramp_length=1, severity_max=0.0
        )

    curves = merged_evalue_operating_curve(
        model, q_hat, CONFORMAL_ALPHA, deltas=[0.3, 0.1],
        onset_stream_factory=make_onset_stream, clear_stream_factory=make_clear_stream,
        bettor_factory=lambda: AGRAPABettor(CONFORMAL_ALPHA),
        n_cells_h=4, n_cells_w=4, n_onset_replicates=2, n_clear_replicates=2,
    )

    assert set(curves.keys()) == {"merged", "whole_frame"}
    for curve in curves.values():
        assert len(curve) == 2
        for point in curve:
            assert 0.0 <= point["false_alarm_rate"] <= 1.0
            assert 0 <= point["n_censored"] <= 2
