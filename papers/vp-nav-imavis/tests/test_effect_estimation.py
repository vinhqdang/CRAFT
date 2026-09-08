import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "conformal-snow-icra2027")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pytest

from signal_monitor.effect_estimation import two_level_effect

ALPHA = 0.2


def _scores(values):
    """One score array per frame; each frame carries a single object."""
    return [np.array([v]) for v in values]


def _make_inputs(n_sessions=6, n_drives=5, nominal_level=1.0, degraded_level=2.0,
                 drive_spread=0.0, seed=0):
    rng = np.random.default_rng(seed)
    sessions = {}
    for s in range(n_sessions):
        calibration = _scores(rng.normal(nominal_level, 0.3, 20))
        monitored = _scores(rng.normal(nominal_level, 0.3, 20))
        sessions[f"s{s}"] = (calibration, monitored)
    degraded = {}
    for d in range(n_drives):
        offset = drive_spread * (d - (n_drives - 1) / 2)
        degraded[f"d{d}"] = _scores(rng.normal(degraded_level + offset, 0.3, 20))
    return sessions, degraded


def test_two_level_ci_is_wider_than_nominal_only_when_drives_vary():
    # The whole point: ignoring degraded-arm variance understates the
    # interval. With heterogeneous degraded drives the two-level CI must be
    # strictly wider than the nominal-only one.
    sessions, degraded = _make_inputs(drive_spread=0.8, seed=1)
    r = two_level_effect(sessions, degraded, ALPHA, n_calibration=20, gap=0,
                         rng=np.random.default_rng(0), n_bootstrap=800)

    assert r["ci_width_inflation"] > 1.0
    assert (r["ci_high"] - r["ci_low"]) > (r["ci_high_nominal_only"] - r["ci_low_nominal_only"])


def test_the_two_intervals_agree_when_degraded_drives_are_identical():
    # Degenerate check: if every degraded drive is the same, resampling
    # drives adds nothing and the two intervals should nearly coincide.
    sessions, degraded = _make_inputs(n_drives=5, drive_spread=0.0, seed=2)
    identical = {k: degraded["d0"] for k in degraded}
    r = two_level_effect(sessions, identical, ALPHA, n_calibration=20, gap=0,
                         rng=np.random.default_rng(0), n_bootstrap=800)

    assert r["ci_width_inflation"] == pytest.approx(1.0, abs=0.05)
    assert r["between_drive_sd"] == pytest.approx(0.0, abs=1e-12)


def test_recovers_a_real_effect_and_rejects_a_null_one():
    sessions, degraded = _make_inputs(nominal_level=1.0, degraded_level=6.0, seed=3)
    real = two_level_effect(sessions, degraded, ALPHA, n_calibration=20, gap=0,
                            rng=np.random.default_rng(0), n_bootstrap=800)
    assert real["effect"] > 0.0
    assert real["excludes_zero"]

    sessions, degraded = _make_inputs(nominal_level=1.0, degraded_level=1.0, seed=4)
    null = two_level_effect(sessions, degraded, ALPHA, n_calibration=20, gap=0,
                            rng=np.random.default_rng(0), n_bootstrap=800)
    assert not null["excludes_zero"]


def test_marginal_and_mondrian_run_over_identical_inputs():
    sessions, degraded = _make_inputs(seed=5)
    common = dict(alpha=ALPHA, n_calibration=20, gap=0, n_bootstrap=400)
    m = two_level_effect(sessions, degraded, rng=np.random.default_rng(0),
                         calibration_mode="marginal", **common)
    d = two_level_effect(sessions, degraded, rng=np.random.default_rng(0),
                         calibration_mode="mondrian", **common)

    assert m["n_sessions"] == d["n_sessions"]
    assert m["n_degraded_drives"] == d["n_degraded_drives"]
    # Marginal uses one pooled quantile for every session; Mondrian does not.
    assert m["calibration_mode"] == "marginal"
    assert d["calibration_mode"] == "mondrian"


def test_gap_removes_leading_monitored_frames():
    sessions, degraded = _make_inputs(seed=6)
    no_gap = two_level_effect(sessions, degraded, ALPHA, 20, gap=0,
                              rng=np.random.default_rng(0), n_bootstrap=200)
    gapped = two_level_effect(sessions, degraded, ALPHA, 20, gap=15,
                              rng=np.random.default_rng(0), n_bootstrap=200)
    assert no_gap is not None and gapped is not None
    # 20 monitored frames minus a 15-frame gap leaves 5, which is the floor;
    # a larger gap must drop the sessions entirely rather than silently
    # reporting on an empty monitored set.
    assert two_level_effect(sessions, degraded, ALPHA, 20, gap=18,
                            rng=np.random.default_rng(0), n_bootstrap=100) is None


def test_reproducible_under_the_same_seed():
    sessions, degraded = _make_inputs(seed=7)
    a = two_level_effect(sessions, degraded, ALPHA, 20, 0, np.random.default_rng(42), n_bootstrap=300)
    b = two_level_effect(sessions, degraded, ALPHA, 20, 0, np.random.default_rng(42), n_bootstrap=300)
    assert a["ci_low"] == pytest.approx(b["ci_low"])
    assert a["ci_high"] == pytest.approx(b["ci_high"])


def test_rejects_unknown_calibration_mode():
    sessions, degraded = _make_inputs(seed=8)
    with pytest.raises(ValueError):
        two_level_effect(sessions, degraded, ALPHA, 20, 0, np.random.default_rng(0),
                         calibration_mode="pooled")
