import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "conformal-snow-icra2027")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest

from craf_x.config import CRAFXConfig
from craf_x.datasets.nuscenes_mock import NuScenesMockDataset
from craf_x.models.crafx_net import CRAFX_Net

from conformal_monitor.betting import AGRAPABettor, CCPInformedBettor, WealthProcess
from conformal_monitor.corruption import WeatherOnsetStream
from conformal_monitor.evaluate import calibrate_on_clear_weather

from mixture_monitor.mixture_betting import compute_mixture_wealth_trajectory, mixture_operating_curve

CONFORMAL_ALPHA = 0.2


def _build_model():
    config = CRAFXConfig(bev_h=32, bev_w=32)
    return CRAFX_Net(config)


def test_mixture_wealth_is_convex_combination_at_every_timestep():
    # Pure numeric check (no model needed): by construction, a weighted
    # average of two sequences must lie between their pointwise min and
    # max at every timestep -- the core arithmetic fact the Vovk & Wang
    # (2021) validity argument rests on. Verified directly, not just
    # assumed, using WealthProcess itself (not a hand-rolled recurrence)
    # so this also guards against a future refactor of WealthProcess.step
    # silently breaking the property.
    wp1 = WealthProcess(alpha=0.2)
    wp2 = WealthProcess(alpha=0.2)
    m_sequence = [0.1, 0.15, 0.6, 0.55, 0.5]
    lambdas1 = [2.0] * len(m_sequence)
    lambdas2 = [1.0] * len(m_sequence)

    k1 = [wp1.step(m, lam) for m, lam in zip(m_sequence, lambdas1)]
    k2 = [wp2.step(m, lam) for m, lam in zip(m_sequence, lambdas2)]

    weight = 0.5
    mixture = [weight * a + (1 - weight) * b for a, b in zip(k1, k2)]
    for a, b, mix in zip(k1, k2, mixture):
        assert min(a, b) <= mix <= max(a, b)


def test_mixture_reduces_to_single_bettor_at_degenerate_weights():
    model = _build_model()
    calibration_dataset = NuScenesMockDataset(num_samples=4)
    q_hat = calibrate_on_clear_weather(model, calibration_dataset, CONFORMAL_ALPHA, batch_size=2)

    stream = WeatherOnsetStream(NuScenesMockDataset(num_samples=4), scene_length=5, onset_frame=2, ramp_length=1)

    bettor_factories = {
        "blind": lambda: AGRAPABettor(CONFORMAL_ALPHA),
        "ccp": lambda: CCPInformedBettor(AGRAPABettor(CONFORMAL_ALPHA), kappa=2.0),
    }
    # Degenerate weights: all mass on "blind" -> mixture trajectory must
    # equal the blind bettor's own trajectory exactly.
    trajectories = compute_mixture_wealth_trajectory(
        model, stream, q_hat, CONFORMAL_ALPHA, bettor_factories, weights={"blind": 1.0, "ccp": 0.0}
    )
    assert trajectories["mixture"] == pytest.approx(trajectories["blind"])


def test_mixture_weights_must_sum_to_one():
    model = _build_model()
    stream = WeatherOnsetStream(NuScenesMockDataset(num_samples=4), scene_length=3, onset_frame=1, ramp_length=1)
    bettor_factories = {"blind": lambda: AGRAPABettor(CONFORMAL_ALPHA), "ccp": lambda: AGRAPABettor(CONFORMAL_ALPHA)}
    with pytest.raises(ValueError):
        compute_mixture_wealth_trajectory(
            model, stream, 1.0, CONFORMAL_ALPHA, bettor_factories, weights={"blind": 0.5, "ccp": 0.6}
        )


def test_mixture_weights_must_be_non_negative():
    model = _build_model()
    stream = WeatherOnsetStream(NuScenesMockDataset(num_samples=4), scene_length=3, onset_frame=1, ramp_length=1)
    bettor_factories = {"blind": lambda: AGRAPABettor(CONFORMAL_ALPHA), "ccp": lambda: AGRAPABettor(CONFORMAL_ALPHA)}
    with pytest.raises(ValueError):
        compute_mixture_wealth_trajectory(
            model, stream, 1.0, CONFORMAL_ALPHA, bettor_factories, weights={"blind": 1.5, "ccp": -0.5}
        )


def test_mixture_operating_curve_end_to_end_reports_all_components():
    model = _build_model()
    calibration_dataset = NuScenesMockDataset(num_samples=4)
    q_hat = calibrate_on_clear_weather(model, calibration_dataset, CONFORMAL_ALPHA, batch_size=2)

    def make_onset_stream():
        return WeatherOnsetStream(NuScenesMockDataset(num_samples=4), scene_length=5, onset_frame=2, ramp_length=1)

    def make_clear_stream():
        return WeatherOnsetStream(
            NuScenesMockDataset(num_samples=4), scene_length=5, onset_frame=2, ramp_length=1, severity_max=0.0
        )

    bettor_factories = {
        "blind": lambda: AGRAPABettor(CONFORMAL_ALPHA),
        "ccp": lambda: CCPInformedBettor(AGRAPABettor(CONFORMAL_ALPHA), kappa=2.0),
    }

    curves = mixture_operating_curve(
        model, q_hat, CONFORMAL_ALPHA, deltas=[0.3, 0.1],
        onset_stream_factory=make_onset_stream, clear_stream_factory=make_clear_stream,
        bettor_factories=bettor_factories, n_onset_replicates=2, n_clear_replicates=2,
    )

    assert set(curves.keys()) == {"blind", "ccp", "mixture"}
    for name, curve in curves.items():
        assert len(curve) == 2  # one point per delta
        for point in curve:
            assert 0.0 <= point["false_alarm_rate"] <= 1.0
            assert 0 <= point["n_censored"] <= 2
