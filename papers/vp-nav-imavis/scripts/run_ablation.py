"""
The ablation: does any signal-strength variant beat the covariate-blind
baseline on detection delay, on a real dataset with a retrained detector?

Each variant is measured independently against the same baseline, on the
same replicate streams and the same calibration protocol, so the
contribution of each change is separable rather than bundled:

  baseline   covariate-blind aGRAPA on the existing matched-cell score
  phantom    same bettor, phantom-aware score (swept over phantom weight)
  merged     same bettor and score, per-cell e-processes merged globally

Interpretation rule, fixed before any of these numbers existed (see
plan.md): the detectors are undertrained, which confounds the two outcomes
asymmetrically. A variant that BEATS the baseline is a valid, conservative
positive -- undertraining worked against it. A variant that FAILS to beat
the baseline is confounded, because "the method does not help" and "the
detector is too weak for the method to have anything to work with" are
indistinguishable from this measurement, and must be escalated rather than
written up as a finding about the method.

Usage:
    python run_ablation.py --dataset snowy --zip-path ../../../data/ROADVIEW5k.zip \
        --checkpoint ../../../checkpoints/snowy_scenes_focal/checkpoint_final.pth
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "conformal-snow-icra2027")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch

from conformal_monitor.betting import AGRAPABettor
from conformal_monitor.evaluate import calibrate_on_clear_weather, operating_curve
from conformal_monitor.real_snow_stream import RealSnowOnsetStream

from signal_monitor.evalue_merge import merged_evalue_operating_curve
from signal_monitor.phantom_score import calibrate_phantom_aware, phantom_aware_operating_curve
from experiment_common import DATASET_DEFAULTS, build_real_setting

PHANTOM_WEIGHTS = [0.25, 0.5]
CELL_GRIDS = [(4, 4), (8, 8)]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["snowy", "cadc"], required=True)
    parser.add_argument("--zip-path")
    parser.add_argument("--data-root")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--bev-size", type=int, default=128)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--variants", nargs="+", default=["baseline", "phantom", "merged"])
    parser.add_argument("--out", default=None)
    return parser.parse_args()


def _print_curve(name, curve):
    print(f"\n  {name}")
    for point in curve:
        delay = point["mean_detection_delay"]
        delay_str = f"{delay:.1f}" if delay is not None else "-- (censored)"
        print(f"    delta={point['delta']:.2f}  FA={point['false_alarm_rate']:.2f}  "
              f"delay={delay_str}  censored={point['n_censored']}/5")


def main():
    args = parse_args()
    setting = build_real_setting(args)
    d = DATASET_DEFAULTS[args.dataset]
    alpha, deltas = d["alpha"], d["deltas"]
    model = setting.model

    def make_onset_stream():
        return RealSnowOnsetStream(setting.nominal_set, setting.degraded_set,
                                   onset_frame=d["onset_frame"], scene_length=d["scene_length"])

    def make_clear_stream():
        return RealSnowOnsetStream(setting.nominal_set, setting.nominal_set,
                                   onset_frame=d["onset_frame"], scene_length=d["scene_length"])

    def bettor_factory():
        return AGRAPABettor(alpha)

    results = {}

    print(f"\nCalibrating on {len(setting.calibration_set)} frames...")
    q_hat = calibrate_on_clear_weather(model, setting.calibration_set, alpha, batch_size=4, num_workers=4)
    print(f"q_hat = {q_hat:.6f}")

    if "baseline" in args.variants:
        print("\n=== BASELINE (covariate-blind aGRAPA, existing score) ===")
        curve = operating_curve(
            model, q_hat, alpha, deltas, make_onset_stream, make_clear_stream, bettor_factory,
            n_onset_replicates=d["n_onset_replicates"], n_clear_replicates=d["n_clear_replicates"],
        )
        results["baseline"] = curve
        _print_curve("baseline", curve)

    if "phantom" in args.variants:
        print("\n=== VARIANT 1: phantom-aware score ===")
        q_loc, q_phantom = calibrate_phantom_aware(
            model, setting.calibration_set, alpha, batch_size=4, num_workers=4
        )
        print(f"  q_loc={q_loc:.6f}  q_phantom={q_phantom:.6f}")
        results["phantom_quantiles"] = {"q_loc": q_loc, "q_phantom": q_phantom}
        for w in PHANTOM_WEIGHTS:
            curve = phantom_aware_operating_curve(
                model, q_loc, q_phantom, alpha, deltas, make_onset_stream, make_clear_stream,
                bettor_factory, phantom_weight=w,
                n_onset_replicates=d["n_onset_replicates"], n_clear_replicates=d["n_clear_replicates"],
            )
            results[f"phantom_w{w}"] = curve
            _print_curve(f"phantom w={w}", curve)

    if "merged" in args.variants:
        print("\n=== VARIANT 2: merged per-cell e-values ===")
        for n_h, n_w in CELL_GRIDS:
            curves = merged_evalue_operating_curve(
                model, q_hat, alpha, deltas, make_onset_stream, make_clear_stream, bettor_factory,
                n_cells_h=n_h, n_cells_w=n_w,
                n_onset_replicates=d["n_onset_replicates"], n_clear_replicates=d["n_clear_replicates"],
            )
            results[f"merged_{n_h}x{n_w}"] = curves["merged"]
            results[f"whole_frame_ref_{n_h}x{n_w}"] = curves["whole_frame"]
            _print_curve(f"merged {n_h}x{n_w}", curves["merged"])
            _print_curve(f"  (whole-frame reference, same passes)", curves["whole_frame"])

    out_path = args.out or f"../manuscript/ablation_{args.dataset}.json"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(
            {
                "note": f"Signal-strength ablation on real {args.dataset} data, retrained detector.",
                "dataset": args.dataset,
                "checkpoint": args.checkpoint,
                "checkpoint_epoch": setting.checkpoint_epoch,
                "q_hat": q_hat,
                "config": {k: v for k, v in d.items()},
                "provenance": setting.provenance,
                "results": results,
            },
            f, indent=2, default=str,
        )
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
