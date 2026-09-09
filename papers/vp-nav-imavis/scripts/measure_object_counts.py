"""
Direct measurement of matched ground-truth object counts, nominal vs.
degraded, on CADC's drive-disjoint evaluation split.

The manuscript claims (5discussion.tex) that degraded frames contain fewer
matched objects than nominal ones, by a specific ratio, as the basis for
ruling out object-count as the driver of the null-detector's content signal
in favour of box-dimension/position statistics. That ratio was previously
asserted with no artifact behind it. This computes it directly from the same
ground-truth heatmap used everywhere else in the pipeline to define a
"matched object" (`match_mask_from_heatmap`, threshold 0.5), on exactly the
nominal/degraded frame sets `experiment_common.build_real_setting` uses for
every other CADC result -- so the object-count comparison is over the same
frames the effect estimates are computed on, not a separate sample.

No model inference is needed: matched-object count comes from the
ground-truth target heatmap alone, so this only reads the dataset.

Usage:
    python measure_object_counts.py --data-root ../../../data/cadcd \
        --checkpoint ../../../checkpoints/cadc_split/checkpoint_final.pth \
        --out ../manuscript/object_counts_cadc.json
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

from conformal_monitor.evaluate import match_mask_from_heatmap
from experiment_common import build_real_setting


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", choices=["cadc"], default="cadc")
    p.add_argument("--data-root", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--bev-size", type=int, default=128)
    p.add_argument("--device", default="cpu")  # no inference needed
    p.add_argument("--out", default="../manuscript/object_counts_cadc.json")
    return p.parse_args()


def counts_for_subset(subset) -> np.ndarray:
    counts = []
    for i in range(len(subset)):
        item = subset[i]
        heatmap = item["targets"]["H"].unsqueeze(0)  # (1, C, H, W)
        mask = match_mask_from_heatmap(heatmap)
        counts.append(float(mask.sum().item()))
    return np.array(counts)


def main():
    args = parse_args()
    args.zip_path = None
    setting = build_real_setting(args)

    nominal_counts = counts_for_subset(setting.nominal_set)
    degraded_counts = counts_for_subset(setting.degraded_set)

    nominal_mean = float(nominal_counts.mean())
    degraded_mean = float(degraded_counts.mean())
    ratio = degraded_mean / nominal_mean if nominal_mean else float("nan")

    print(f"nominal frames: n={len(nominal_counts)}, mean matched objects/frame = {nominal_mean:.4f}")
    print(f"degraded frames: n={len(degraded_counts)}, mean matched objects/frame = {degraded_mean:.4f}")
    print(f"degraded / nominal ratio = {ratio:.4f}")

    out = {
        "note": "Matched-object count per frame, from the ground-truth heatmap "
                "(match_mask_from_heatmap, threshold 0.5), no model inference. "
                "Same nominal/degraded frame sets as every other CADC result "
                "(experiment_common.build_real_setting).",
        "checkpoint": args.checkpoint,
        "nominal": {"n_frames": len(nominal_counts), "mean_objects_per_frame": nominal_mean,
                    "sd": float(nominal_counts.std(ddof=1))},
        "degraded": {"n_frames": len(degraded_counts), "mean_objects_per_frame": degraded_mean,
                     "sd": float(degraded_counts.std(ddof=1))},
        "degraded_over_nominal_ratio": ratio,
    }
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
