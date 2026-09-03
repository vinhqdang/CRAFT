from craf_x.config import CRAFXConfig
from craf_x.datasets.nuscenes_mock import NuScenesMockDataset
from craf_x.models.crafx_net import CRAFX_Net

from conformal_monitor.betting import AGRAPABettor, CCPInformedBettor
from conformal_monitor.corruption import WeatherOnsetStream
from conformal_monitor.evaluate import (
    calibrate_on_clear_weather,
    operating_curve,
    run_global_monitor,
    run_spatial_monitor,
)

CONFORMAL_ALPHA = 0.2  # target miscoverage level (unrelated to CRAFXConfig.alpha, the CCP temperature)


def _build_model():
    # bev_h/bev_w must match NuScenesMockDataset's fixed 32x32 target shapes.
    config = CRAFXConfig(bev_h=32, bev_w=32)
    return CRAFX_Net(config)


def test_calibrate_on_clear_weather_returns_finite_quantile():
    model = _build_model()
    calibration_dataset = NuScenesMockDataset(num_samples=4)
    q_hat = calibrate_on_clear_weather(model, calibration_dataset, CONFORMAL_ALPHA, batch_size=2)
    assert q_hat >= 0.0


def test_run_global_monitor_end_to_end():
    model = _build_model()
    calibration_dataset = NuScenesMockDataset(num_samples=4)
    q_hat = calibrate_on_clear_weather(model, calibration_dataset, CONFORMAL_ALPHA, batch_size=2)

    stream = WeatherOnsetStream(
        NuScenesMockDataset(num_samples=4), scene_length=6, onset_frame=3, ramp_length=1,
    )
    run = run_global_monitor(
        model, stream, q_hat, CONFORMAL_ALPHA, delta=0.2,
        bettor_factory=lambda: AGRAPABettor(CONFORMAL_ALPHA),
    )

    assert run.onset_frame == 3
    assert 1 <= len(run.wealth_trajectory) <= len(stream)
    if run.alarm_time is not None:
        assert run.detection_delay == run.alarm_time - run.onset_frame


def test_run_spatial_monitor_end_to_end():
    model = _build_model()
    calibration_dataset = NuScenesMockDataset(num_samples=4)
    q_hat = calibrate_on_clear_weather(model, calibration_dataset, CONFORMAL_ALPHA, batch_size=2)

    stream = WeatherOnsetStream(
        NuScenesMockDataset(num_samples=4), scene_length=3, onset_frame=1, ramp_length=1,
    )
    flagged_maps = run_spatial_monitor(
        model, stream, q_hat, CONFORMAL_ALPHA, delta=0.2,
        bettor_factory=lambda: AGRAPABettor(CONFORMAL_ALPHA),
        n_cells_h=4, n_cells_w=4, correction="ebh",
    )

    assert len(flagged_maps) == len(stream)
    for flagged in flagged_maps:
        assert flagged.shape == (4, 4)
        assert flagged.dtype == bool


def test_operating_curve_covariate_informed_vs_blind_shapes():
    model = _build_model()
    calibration_dataset = NuScenesMockDataset(num_samples=4)
    q_hat = calibrate_on_clear_weather(model, calibration_dataset, CONFORMAL_ALPHA, batch_size=2)

    def onset_stream_factory():
        return WeatherOnsetStream(
            NuScenesMockDataset(num_samples=4), scene_length=4, onset_frame=2, ramp_length=1,
        )

    def clear_stream_factory():
        # onset_frame == scene_length -> severity stays 0 for the whole scene
        return WeatherOnsetStream(
            NuScenesMockDataset(num_samples=4), scene_length=4, onset_frame=3, ramp_length=1, severity_max=0.0,
        )

    for bettor_factory in (
        lambda: AGRAPABettor(CONFORMAL_ALPHA),
        lambda: CCPInformedBettor(AGRAPABettor(CONFORMAL_ALPHA), kappa=1.0),
    ):
        curve = operating_curve(
            model, q_hat, CONFORMAL_ALPHA, deltas=[0.2, 0.4],
            onset_stream_factory=onset_stream_factory,
            clear_stream_factory=clear_stream_factory,
            bettor_factory=bettor_factory,
            n_onset_replicates=1, n_clear_replicates=1,
        )
        assert len(curve) == 2
        for point in curve:
            assert 0.0 <= point["false_alarm_rate"] <= 1.0
            assert "mean_detection_delay" in point
