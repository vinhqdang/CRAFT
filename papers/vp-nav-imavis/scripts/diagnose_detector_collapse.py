"""
Detector-collapse diagnostic: is the conformal monitor actually measuring
the detector at all?

This script exists because the phantom-score diagnostic
(`diagnose_phantom_signal.py`) produced an impossible-looking result -- the
phantom component moved in the wrong direction at onset -- which traced
back to something more fundamental than the score design: on both real
checkpoints, the monitored detector's output heads have collapsed to
constants, so the nonconformity score
s = ||box_pred - box_target||_1 reduces to ||box_target||_1 and the monitor
is measuring the ground-truth scene content, not detector error.

Three independent checks, all reported as real measured numbers:

1. Head statistics: the range/std of the predicted heatmap and box
   regression outputs, and whether the heatmap can distinguish a
   ground-truth object cell from an empty one.
2. Phantom-detection census: the fraction of empty cells whose predicted
   activation exceeds each of several thresholds spanning the noise floor
   up to real detection confidence.
3. Zero-prediction ablation (the decisive one): recompute the calibrated
   quantile and the per-regime miscoverage rates with the entire trained
   box head's output replaced by literal zeros. If the monitor depends on
   the detector at all, these must differ from the real-model numbers.

Usage:
    python diagnose_detector_collapse.py --dataset snowy \
        --zip-path ../../../data/ROADVIEW5k.zip \
        --checkpoint ../../../checkpoints/snowy_scenes_fixed/checkpoint_final.pth

    python diagnose_detector_collapse.py --dataset cadc \
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

import numpy as np
import torch

from conformal_monitor.calibration import (
    calibrate_quantile,
    frame_miscoverage_rate,
    object_nonconformity_scores,
)
from conformal_monitor.evaluate import _to_device, match_mask_from_heatmap

from experiment_common import DATASET_DEFAULTS, build_real_setting

ACTIVATION_THRESHOLDS = [0.05, 0.1, 0.25, 0.5, 0.75]
# A detection head that has genuinely learned to localize should put
# noticeably more heatmap mass on ground-truth object cells than on empty
# ones. This is the margin below which we call the head collapsed.
SEPARATION_EPSILON = 1e-3


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["snowy", "cadc"], required=True)
    parser.add_argument("--zip-path")
    parser.add_argument("--data-root")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--bev-size", type=int, default=128)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--n-frames", type=int, default=40)
    parser.add_argument("--out", default=None)
    return parser.parse_args()


@torch.no_grad()
def head_statistics(model, subset, n_frames):
    """Check 1 + 2: are the heads producing anything input-dependent?"""
    device = next(model.parameters()).device
    n = min(n_frames, len(subset))
    heatmap_std, box_std, box_absmax = [], [], []
    peak_matched, peak_empty = [], []
    above = {thr: [] for thr in ACTIVATION_THRESHOLDS}

    for i in range(n):
        sample = subset[i]
        image = sample["image"].unsqueeze(0)
        pointcloud = sample["pointcloud"].unsqueeze(0)
        targets = {k: v.unsqueeze(0) for k, v in sample["targets"].items()}
        image, pointcloud, targets = _to_device(image, pointcloud, targets, device)

        out = model(image, pointcloud)
        match_mask = match_mask_from_heatmap(targets["H"]).bool()
        peak = out["H"].amax(dim=1, keepdim=True)

        heatmap_std.append(float(out["H"].std().item()))
        box_std.append(float(out["B"].std().item()))
        box_absmax.append(float(out["B"].abs().max().item()))
        if match_mask.any():
            peak_matched.append(float(peak[match_mask].mean().item()))
        peak_empty.append(float(peak[~match_mask].mean().item()))
        for thr in ACTIVATION_THRESHOLDS:
            above[thr].append(float((peak[~match_mask] > thr).float().mean().item()))

    matched_mean = float(np.mean(peak_matched)) if peak_matched else float("nan")
    empty_mean = float(np.mean(peak_empty))
    return {
        "n_frames": n,
        "heatmap_std": float(np.mean(heatmap_std)),
        "box_pred_std": float(np.mean(box_std)),
        "box_pred_abs_max": float(np.mean(box_absmax)),
        "heatmap_peak_at_gt_object_cells": matched_mean,
        "heatmap_peak_at_empty_cells": empty_mean,
        "object_vs_empty_separation": matched_mean - empty_mean,
        "frac_empty_cells_above": {
            str(thr): float(np.mean(above[thr])) for thr in ACTIVATION_THRESHOLDS
        },
    }


@torch.no_grad()
def _regime_scores(model, subset, n_frames, zero_prediction):
    device = next(model.parameters()).device
    collected = []
    for i in range(min(n_frames, len(subset))):
        sample = subset[i]
        image = sample["image"].unsqueeze(0)
        pointcloud = sample["pointcloud"].unsqueeze(0)
        targets = {k: v.unsqueeze(0) for k, v in sample["targets"].items()}
        image, pointcloud, targets = _to_device(image, pointcloud, targets, device)

        out = model(image, pointcloud)
        box_pred = torch.zeros_like(out["B"]) if zero_prediction else out["B"]
        match_mask = match_mask_from_heatmap(targets["H"])
        collected.append(object_nonconformity_scores(box_pred, targets["B"], match_mask))
    return collected


def zero_prediction_ablation(model, setting, alpha, n_frames):
    """Check 3: replace the trained box head's output with literal zeros."""
    results = {}
    for zero_prediction in (False, True):
        key = "zero_prediction" if zero_prediction else "real_model"
        calibration = np.concatenate(
            _regime_scores(model, setting.calibration_set, n_frames, zero_prediction)
        )
        q_hat = calibrate_quantile(calibration, alpha)
        nominal = [
            frame_miscoverage_rate(s, q_hat)
            for s in _regime_scores(model, setting.nominal_set, n_frames, zero_prediction)
        ]
        degraded = [
            frame_miscoverage_rate(s, q_hat)
            for s in _regime_scores(model, setting.degraded_set, n_frames, zero_prediction)
        ]
        results[key] = {
            "q_hat": q_hat,
            "nominal_m": float(np.mean(nominal)),
            "degraded_m": float(np.mean(degraded)),
            "jump": float(np.mean(degraded) - np.mean(nominal)),
        }
    return results


def main():
    args = parse_args()
    setting = build_real_setting(args)
    alpha = DATASET_DEFAULTS[args.dataset]["alpha"]

    print("\n[1/3] Head statistics on nominal frames")
    stats = head_statistics(setting.model, setting.nominal_set, args.n_frames)
    print(f"  predicted heatmap std           : {stats['heatmap_std']:.8f}")
    print(f"  predicted box std               : {stats['box_pred_std']:.8f}")
    print(f"  predicted box |max|             : {stats['box_pred_abs_max']:.8f}")
    print(f"  heatmap peak @ GT object cells  : {stats['heatmap_peak_at_gt_object_cells']:.6f}")
    print(f"  heatmap peak @ empty cells      : {stats['heatmap_peak_at_empty_cells']:.6f}")
    print(f"  object-vs-empty separation      : {stats['object_vs_empty_separation']:+.6f}")

    print("\n[2/3] Phantom-detection census (fraction of empty cells above threshold)")
    for thr in ACTIVATION_THRESHOLDS:
        print(f"  > {thr:<5}: {stats['frac_empty_cells_above'][str(thr)]:.6f}")

    print("\n[3/3] Zero-prediction ablation (decisive)")
    ablation = zero_prediction_ablation(setting.model, setting, alpha, args.n_frames)
    for key in ("real_model", "zero_prediction"):
        r = ablation[key]
        print(f"  {key:<16} q_hat={r['q_hat']:.6f}  nominal_m={r['nominal_m']:.5f}  "
              f"degraded_m={r['degraded_m']:.5f}  jump={r['jump']:+.5f}")

    m_delta = abs(ablation["real_model"]["nominal_m"] - ablation["zero_prediction"]["nominal_m"])
    collapsed = (
        stats["object_vs_empty_separation"] < SEPARATION_EPSILON and m_delta < 1e-4
    )
    verdict = (
        "COLLAPSED: the monitor's evidence does not depend on the detector's output. "
        "The nonconformity score reduces to the ground-truth box magnitude, so the "
        "monitor is measuring scene content, not perception degradation."
        if collapsed
        else "Detector output does affect the monitor's evidence."
    )
    print(f"\nVERDICT: {verdict}")

    out_path = args.out or f"../manuscript/detector_collapse_{args.dataset}.json"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(
            {
                "note": f"Detector-collapse diagnostic ({args.dataset}).",
                "dataset": args.dataset,
                "checkpoint": args.checkpoint,
                "checkpoint_epoch": setting.checkpoint_epoch,
                "alpha": alpha,
                "head_statistics": stats,
                "zero_prediction_ablation": ablation,
                "collapsed": bool(collapsed),
                "verdict": verdict,
            },
            f, indent=2,
        )
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
