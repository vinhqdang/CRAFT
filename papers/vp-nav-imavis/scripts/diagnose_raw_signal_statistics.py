"""
Raw per-regime signal statistics, with no conformal thresholding in the way.

The phantom-aware score diagnostic (`diagnose_phantom_signal.py`) reported
that the phantom component moves in the WRONG direction at onset on real
data. That could mean either (a) the underlying hypothesis is wrong -- snow
does not in fact produce more phantom heatmap evidence for this detector --
or (b) the hypothesis is right but the score parameterization is wrong:
the conformal threshold lands at the noise floor rather than in the
detection regime, so the statistic is measuring overall heatmap scale
rather than genuine phantom detections.

This script distinguishes the two by reporting the raw distribution
statistics per regime, un-thresholded: mean/max predicted activation on
unmatched cells, counts above several fixed activation thresholds spanning
noise floor to real-detection confidence, and, for reference, the
localization residual the existing score is built on.

Usage mirrors diagnose_phantom_signal.py.
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

from conformal_monitor.evaluate import _to_device, match_mask_from_heatmap
from conformal_monitor.real_snow_stream import RealSnowOnsetStream

from experiment_common import DATASET_DEFAULTS, build_real_setting

ACTIVATION_THRESHOLDS = [0.05, 0.1, 0.25, 0.5, 0.75]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["snowy", "cadc"], required=True)
    parser.add_argument("--zip-path")
    parser.add_argument("--data-root")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--bev-size", type=int, default=128)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--n-frames", type=int, default=60,
                        help="frames sampled per regime (nominal / degraded)")
    parser.add_argument("--out", default=None)
    return parser.parse_args()


@torch.no_grad()
def regime_statistics(model, subset, n_frames):
    """Raw statistics over up to `n_frames` frames of one regime."""
    device = next(model.parameters()).device
    n = min(n_frames, len(subset))

    mean_activation, max_activation = [], []
    above = {thr: [] for thr in ACTIVATION_THRESHOLDS}
    mean_residual, n_matched = [], []

    for i in range(n):
        sample = subset[i]
        image = sample["image"].unsqueeze(0)
        pointcloud = sample["pointcloud"].unsqueeze(0)
        targets = {k: v.unsqueeze(0) for k, v in sample["targets"].items()}
        image, pointcloud, targets = _to_device(image, pointcloud, targets, device)

        out = model(image, pointcloud)
        match_mask = match_mask_from_heatmap(targets["H"])
        unmatched = (1.0 - match_mask).bool()

        activation = out["H"].amax(dim=1, keepdim=True)[unmatched]
        mean_activation.append(float(activation.mean().item()))
        max_activation.append(float(activation.max().item()))
        for thr in ACTIVATION_THRESHOLDS:
            # Fraction of empty cells carrying activation above this level.
            above[thr].append(float((activation > thr).float().mean().item()))

        residual = torch.abs(out["B"] - targets["B"]).sum(dim=1, keepdim=True)
        matched = match_mask.bool()
        if matched.any():
            mean_residual.append(float(residual[matched].mean().item()))
        n_matched.append(int(matched.sum().item()))

    return {
        "n_frames": n,
        "mean_unmatched_activation": float(np.mean(mean_activation)),
        "max_unmatched_activation": float(np.mean(max_activation)),
        "frac_above": {str(thr): float(np.mean(above[thr])) for thr in ACTIVATION_THRESHOLDS},
        "mean_matched_residual": float(np.mean(mean_residual)) if mean_residual else float("nan"),
        "mean_n_matched_cells": float(np.mean(n_matched)),
    }


def main():
    args = parse_args()
    setting = build_real_setting(args)

    print(f"\nComputing raw statistics over up to {args.n_frames} frames per regime...")
    nominal = regime_statistics(setting.model, setting.nominal_set, args.n_frames)
    degraded = regime_statistics(setting.model, setting.degraded_set, args.n_frames)

    def row(label, key_fn):
        nv, dv = key_fn(nominal), key_fn(degraded)
        arrow = "UP  " if dv > nv else ("DOWN" if dv < nv else "same")
        print(f"  {label:<38} {nv:>10.5f}  {dv:>10.5f}  {arrow}  {dv - nv:>+10.5f}")

    print(f"\n  {'statistic':<38} {'nominal':>10}  {'degraded':>10}  {'dir':<4}  {'delta':>10}")
    row("mean activation (empty cells)", lambda s: s["mean_unmatched_activation"])
    row("max activation (empty cells)", lambda s: s["max_unmatched_activation"])
    for thr in ACTIVATION_THRESHOLDS:
        row(f"frac empty cells > {thr}", lambda s, t=str(thr): s["frac_above"][t])
    row("mean residual (matched cells)", lambda s: s["mean_matched_residual"])
    row("mean # matched cells", lambda s: s["mean_n_matched_cells"])

    out_path = args.out or f"../manuscript/raw_signal_statistics_{args.dataset}.json"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(
            {
                "note": f"Raw un-thresholded per-regime signal statistics ({args.dataset}).",
                "dataset": args.dataset,
                "checkpoint": args.checkpoint,
                "nominal": nominal,
                "degraded": degraded,
            },
            f, indent=2,
        )
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
