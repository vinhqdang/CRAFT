"""
Runs the conformal_monitor pipeline at a larger scale than the unit tests
(plan.md next step: "Run the implemented pipeline at scale ... to actually
produce the head-to-head operating curve"), comparing the covariate-blind
AGRAPA baseline against the CCP-informed bettor on the KITTI/nuScenes
synthetic weather-onset fallback stream.

IMPORTANT — this is a mechanics smoke-run, not an experimental result: it
uses `CRAFX_Net` with randomly initialized (untrained) weights and the mock
dataset's random tensors, so the detections themselves are meaningless. What
it does validate is that calibration, both bettors, and the operating-curve
harness run correctly end to end at scene lengths and replicate counts
closer to a real experiment than the tiny unit tests use. Swap in a trained
checkpoint and a real (or corruption-augmented real) dataset before treating
any numbers this prints as evidence for the paper's claim.
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import json
import time

from craf_x.config import CRAFXConfig
from craf_x.datasets.nuscenes_mock import NuScenesMockDataset
from craf_x.models.crafx_net import CRAFX_Net

from conformal_monitor.betting import AGRAPABettor, CCPInformedBettor
from conformal_monitor.corruption import WeatherOnsetStream
from conformal_monitor.evaluate import calibrate_on_clear_weather, operating_curve

CONFORMAL_ALPHA = 0.2  # target miscoverage level (unrelated to CRAFXConfig.alpha, the CCP temperature)
SCENE_LENGTH = 20
ONSET_FRAME = 8
RAMP_LENGTH = 5
DELTAS = [0.3, 0.1, 0.05]
N_ONSET_REPLICATES = 3
N_CLEAR_REPLICATES = 3
KAPPA = 2.0


def make_onset_stream():
    return WeatherOnsetStream(
        NuScenesMockDataset(num_samples=8),
        scene_length=SCENE_LENGTH,
        onset_frame=ONSET_FRAME,
        ramp_length=RAMP_LENGTH,
    )


def make_clear_stream():
    # severity_max=0.0 -> severity stays 0 for the whole scene regardless of onset_frame
    return WeatherOnsetStream(
        NuScenesMockDataset(num_samples=8),
        scene_length=SCENE_LENGTH,
        onset_frame=ONSET_FRAME,
        ramp_length=RAMP_LENGTH,
        severity_max=0.0,
    )


def main():
    config = CRAFXConfig(bev_h=32, bev_w=32)  # must match NuScenesMockDataset's fixed target shapes
    model = CRAFX_Net(config)

    print("Calibrating on clear-weather set...")
    q_hat = calibrate_on_clear_weather(model, NuScenesMockDataset(num_samples=8), CONFORMAL_ALPHA, batch_size=4)
    print(f"q_hat = {q_hat:.4f}")

    bettor_factories = {
        "covariate_blind_agrapa": lambda: AGRAPABettor(CONFORMAL_ALPHA),
        "ccp_informed": lambda: CCPInformedBettor(AGRAPABettor(CONFORMAL_ALPHA), kappa=KAPPA),
    }

    results = {}
    for name, factory in bettor_factories.items():
        print(f"\nRunning operating curve for bettor: {name}")
        start = time.time()
        curve = operating_curve(
            model, q_hat, CONFORMAL_ALPHA, DELTAS,
            onset_stream_factory=make_onset_stream,
            clear_stream_factory=make_clear_stream,
            bettor_factory=factory,
            n_onset_replicates=N_ONSET_REPLICATES,
            n_clear_replicates=N_CLEAR_REPLICATES,
        )
        elapsed = time.time() - start
        results[name] = curve
        print(f"  ({elapsed:.1f}s)")
        for point in curve:
            print(
                f"  delta={point['delta']:.2f}  "
                f"false_alarm_rate={point['false_alarm_rate']:.2f}  "
                f"mean_detection_delay={point['mean_detection_delay']}  "
                f"n_censored={point['n_censored']}"
            )

    out_path = os.path.join(os.path.dirname(__file__), "..", "operating_curve_smoke_run.json")
    with open(out_path, "w") as f:
        json.dump(
            {
                "note": "Smoke run on an untrained model and synthetic mock data — mechanics check, not a result.",
                "config": {
                    "conformal_alpha": CONFORMAL_ALPHA,
                    "scene_length": SCENE_LENGTH,
                    "onset_frame": ONSET_FRAME,
                    "ramp_length": RAMP_LENGTH,
                    "deltas": DELTAS,
                    "n_onset_replicates": N_ONSET_REPLICATES,
                    "n_clear_replicates": N_CLEAR_REPLICATES,
                    "kappa": KAPPA,
                },
                "results": results,
            },
            f,
            indent=2,
        )
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
