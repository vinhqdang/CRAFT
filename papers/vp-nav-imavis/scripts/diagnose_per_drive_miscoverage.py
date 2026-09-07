"""
Per-drive nominal miscoverage: is split conformal's exchangeability
assumption holding across driving sessions?

The CADC monitor posts a false-alarm rate of 1.00 under the retrained
detector, having previously been fixed by date-stratified calibration under
the collapsed one. The mechanism cannot be the same in both cases: with a
collapsed detector the nonconformity score was ||B_target||_1, so
drive-to-drive heterogeneity was *content* heterogeneity, which
date-stratification balances. With a working detector the score is genuine
detector error, so any heterogeneity is now per-drive variation in how well
the detector performs -- a different quantity that date-stratification was
never balancing.

This script measures it directly. Calibrate q_hat on the calibration drives
exactly as the evaluation does, then report each drive's own mean frame
miscoverage under that single pooled quantile. Under exchangeability across
drives every drive should sit near alpha. If some drives run far above
alpha while others sit well below, marginal validity is holding only on
average while failing per session -- which is precisely a monitor that
alarms constantly on the unlucky drives.

Reports the nominal (bare) drives, since those are what the "clear"
replicate streams are drawn from and therefore what drives the false-alarm
rate. Degraded (covered) drives are reported alongside for context.

Usage:
    python diagnose_per_drive_miscoverage.py --dataset cadc \
        --data-root ../../../data/cadcd \
        --checkpoint ../../../checkpoints/cadc_focal/checkpoint_final.pth
"""
import argparse
import json
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "conformal-snow-icra2027")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import torch
from torch.utils.data import Subset

from conformal_monitor.calibration import frame_miscoverage_rate, object_nonconformity_scores
from conformal_monitor.evaluate import calibrate_on_clear_weather, _to_device, match_mask_from_heatmap

from experiment_common import DATASET_DEFAULTS, build_real_setting, _drives_for_category


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["cadc"], default="cadc")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--bev-size", type=int, default=128)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-frames-per-drive", type=int, default=60)
    parser.add_argument("--out", default="../manuscript/per_drive_miscoverage_cadc.json")
    return parser.parse_args()


@torch.no_grad()
def drive_miscoverage(model, dataset, indices, q_hat, max_frames):
    device = next(model.parameters()).device
    per_frame = []
    for i in indices[:max_frames]:
        sample = dataset[i]
        image = sample["image"].unsqueeze(0)
        pointcloud = sample["pointcloud"].unsqueeze(0)
        targets = {k: v.unsqueeze(0) for k, v in sample["targets"].items()}
        image, pointcloud, targets = _to_device(image, pointcloud, targets, device)

        out = model(image, pointcloud)
        mask = match_mask_from_heatmap(targets["H"])
        scores = object_nonconformity_scores(out["B"], targets["B"], mask)
        per_frame.append(frame_miscoverage_rate(scores, q_hat))
    return per_frame


def main():
    args = parse_args()
    args.zip_path = None
    setting = build_real_setting(args)
    d = DATASET_DEFAULTS[args.dataset]
    alpha = d["alpha"]
    model = setting.model

    q_hat = calibrate_on_clear_weather(model, setting.calibration_set, alpha,
                                       batch_size=4, num_workers=4)
    print(f"\nPooled q_hat calibrated on {len(setting.calibration_set)} frames: {q_hat:.6f}")
    print(f"Target miscoverage alpha = {alpha}\n")

    calibration_drives = set(tuple(x) for x in setting.provenance["calibration_drives"])
    results = {}

    for category in ("bare", "covered"):
        drives = _drives_for_category(setting.dataset, category)
        print(f"=== {category} drives ===")
        print(f"  {'drive':<22} {'role':<12} {'n':>4} {'mean m':>8} {'above alpha?':>13}")
        for key in sorted(drives):
            role = ("calibration" if key in calibration_drives else "nominal") if category == "bare" else "degraded"
            per_frame = drive_miscoverage(model, setting.dataset, drives[key], q_hat,
                                          args.max_frames_per_drive)
            mean_m = float(np.mean(per_frame)) if per_frame else float("nan")
            flag = "YES <-- " if mean_m > alpha else ""
            name = f"{key[0]}_{key[1]}"
            print(f"  {name:<22} {role:<12} {len(per_frame):>4} {mean_m:>8.4f} {flag:>13}")
            results[f"{category}/{name}"] = {
                "role": role, "n_frames": len(per_frame), "mean_miscoverage": mean_m,
                "above_alpha": bool(mean_m > alpha),
            }
        print()

    bare = {k: v for k, v in results.items() if k.startswith("bare/")}
    nominal_only = {k: v for k, v in bare.items() if v["role"] == "nominal"}
    means = [v["mean_miscoverage"] for v in bare.values()]
    nominal_means = [v["mean_miscoverage"] for v in nominal_only.values()]
    n_above = sum(1 for v in nominal_only.values() if v["above_alpha"])

    print("SUMMARY (bare / nominal-regime drives -- what the clear streams sample)")
    print(f"  drives: {len(nominal_only)}   above alpha: {n_above}")
    print(f"  spread across ALL bare drives: min={min(means):.4f} max={max(means):.4f} "
          f"ratio={max(means) / max(min(means), 1e-9):.1f}x")
    if nominal_means:
        print(f"  nominal-only: min={min(nominal_means):.4f} max={max(nominal_means):.4f} "
              f"mean={np.mean(nominal_means):.4f}")
    verdict = (
        "HETEROGENEOUS: some nominal drives sit far above alpha under a single pooled "
        "quantile, so marginal validity holds on average while failing per session."
        if n_above > 0 else
        "Nominal drives all sit at or below alpha; heterogeneity is not the explanation."
    )
    print(f"\n  {verdict}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(
            {
                "note": "Per-drive mean frame miscoverage under a single pooled q_hat.",
                "checkpoint": args.checkpoint, "q_hat": q_hat, "alpha": alpha,
                "max_frames_per_drive": args.max_frames_per_drive,
                "drives": results, "n_nominal_drives_above_alpha": n_above,
                "verdict": verdict,
            },
            f, indent=2,
        )
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
