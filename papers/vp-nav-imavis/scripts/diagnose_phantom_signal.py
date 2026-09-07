"""
Diagnostic (run BEFORE any operating curve): does the phantom-aware
nonconformity score actually make m(t) jump harder at onset on real data?

Detection delay is driven by how far m(t) rises above alpha after onset.
There is no point running a full operating-curve sweep on a score that does
not widen that gap, so this script measures the real per-regime mean m(t)
-- nominal frames versus degraded frames -- across a sweep of phantom
weights, on whichever real dataset is selected.

Usage (Snowy Scenes):
    python diagnose_phantom_signal.py --dataset snowy \
        --zip-path ../../../data/ROADVIEW5k.zip \
        --checkpoint ../../../checkpoints/snowy_scenes_fixed/checkpoint_final.pth

Usage (CADC):
    python diagnose_phantom_signal.py --dataset cadc \
        --data-root ../../../data/cadcd \
        --checkpoint ../../../checkpoints/cadc/checkpoint_final.pth
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "conformal-snow-icra2027")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch

from craf_x.config import CRAFXConfig
from craf_x.models.crafx_net import CRAFX_Net

from conformal_monitor.real_snow_stream import RealSnowOnsetStream

from signal_monitor.phantom_score import calibrate_phantom_aware, measure_miscoverage_by_regime
from experiment_common import DATASET_DEFAULTS, build_real_setting

PHANTOM_WEIGHTS = [0.0, 0.25, 0.5, 0.75, 1.0]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["snowy", "cadc"], required=True)
    parser.add_argument("--zip-path", help="Snowy Scenes ROADVIEW5k.zip (dataset=snowy)")
    parser.add_argument("--data-root", help="CADC data root (dataset=cadc)")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--bev-size", type=int, default=128)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    setting = build_real_setting(args)
    model = setting.model
    defaults = DATASET_DEFAULTS[args.dataset]

    print(f"Calibrating both score families on {len(setting.calibration_set)} nominal frames...")
    q_loc, q_phantom = calibrate_phantom_aware(
        model, setting.calibration_set, defaults["alpha"], batch_size=4, num_workers=4
    )
    print(f"q_loc = {q_loc:.6f}   q_phantom = {q_phantom:.6f}")

    stream = RealSnowOnsetStream(
        setting.nominal_set,
        setting.degraded_set,
        onset_frame=defaults["onset_frame"],
        scene_length=defaults["scene_length"],
    )
    print(f"\nMeasuring m(t) by regime over a {len(stream)}-frame real stream "
          f"(onset at t={stream.onset_frame})...")
    summary = measure_miscoverage_by_regime(model, stream, q_loc, q_phantom, PHANTOM_WEIGHTS)

    print(f"\n{'w':>6}  {'nominal m':>10}  {'degraded m':>11}  {'jump':>9}")
    for w in PHANTOM_WEIGHTS:
        s = summary[w]
        print(f"{w:>6.2f}  {s['nominal_mean']:>10.4f}  {s['degraded_mean']:>11.4f}  {s['jump']:>+9.4f}")

    baseline_jump = summary[0.0]["jump"]
    best_w = max(PHANTOM_WEIGHTS, key=lambda w: summary[w]["jump"])
    print(f"\nBaseline (w=0, existing score) jump: {baseline_jump:+.4f}")
    print(f"Best weight: w={best_w} with jump {summary[best_w]['jump']:+.4f}")
    if summary[best_w]["jump"] > baseline_jump:
        print("=> Phantom component DOES widen the onset gap. Worth an operating curve.")
    else:
        print("=> Phantom component does NOT widen the onset gap. Report as such.")

    out_path = args.out or f"../manuscript/phantom_signal_diagnostic_{args.dataset}.json"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(
            {
                "note": f"Real m(t)-by-regime diagnostic for the phantom-aware score ({args.dataset}).",
                "dataset": args.dataset,
                "checkpoint": args.checkpoint,
                "q_loc": q_loc,
                "q_phantom": q_phantom,
                "alpha": defaults["alpha"],
                "onset_frame": stream.onset_frame,
                "scene_length": len(stream),
                "summary": {str(w): summary[w] for w in PHANTOM_WEIGHTS},
            },
            f, indent=2,
        )
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
