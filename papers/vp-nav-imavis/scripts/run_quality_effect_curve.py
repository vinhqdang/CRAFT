"""
Is monitorable weather signal non-monotone in detector quality?

Three checkpoints already suggested it -- collapsed (content artifact
only), 5-epoch (real, attributable signal), converged (none) -- but three
points with a sign change is thin support for a shape claim. This measures
every available checkpoint on one consistent protocol so the curve stands
on more than its endpoints.

The x-axis is **heatmap object-vs-empty separation**, not epoch count or
training loss. Those decouple: between epochs 12 and 20 the loss kept
falling (5.52 -> 4.87) while separation plateaued, so an epoch axis would
imply the detector kept improving in the dimension the monitor consumes
when it had stopped. Separation is the monitoring-relevant quality measure,
and using anything else risks a flat curve that merely reflects a
stationary x-axis.

Both nonconformity scores are measured at every checkpoint -- localization
(w=0) and pure phantom (w=1) -- because they are structurally different
(one blind to phantom detections, one dominated by them) and agreement
between them is much stronger evidence about the underlying signal than
either alone.

Protocol matches the effect-existence gate: session-conditional quantiles,
session-paired scoring, session-level bootstrap over 18 sessions, temporal
gap of 10 frames.

Usage:
    python run_quality_effect_curve.py --data-root ../../../data/cadcd
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

from craf_x.config import CRAFXConfig
from craf_x.models.crafx_net import CRAFX_Net
from craf_x.datasets.cadc_dataset import CADC_NUM_CLASSES

from conformal_monitor.evaluate import _to_device, match_mask_from_heatmap

from experiment_common import DATASET_DEFAULTS, build_real_setting, _drives_for_category
from run_phantom_effect_test import cache_both_scores, measure

GAP = 10
RNG_SEED = 20260908

CHECKPOINTS = [
    ("collapsed", "../../../checkpoints/cadc/checkpoint_final.pth"),
    ("focal_ep5", "../../../checkpoints/cadc_focal/checkpoint_final.pth"),
    ("focal_ep9", "../../../checkpoints/cadc_converged/checkpoint_epoch9.pth"),
    ("focal_ep13", "../../../checkpoints/cadc_converged/checkpoint_epoch13.pth"),
    ("focal_ep16", "../../../checkpoints/cadc_converged/checkpoint_epoch16.pth"),
    ("focal_ep20", "../../../checkpoints/cadc_converged/checkpoint_final.pth"),
]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["cadc"], default="cadc")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--checkpoint", default="../../../checkpoints/cadc_focal/checkpoint_final.pth",
                        help="only used to build the dataset splits")
    parser.add_argument("--split", default="val")
    parser.add_argument("--bev-size", type=int, default=128)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-frames-per-session", type=int, default=60)
    parser.add_argument("--max-degraded-frames", type=int, default=300)
    parser.add_argument("--quality-frames", type=int, default=40)
    parser.add_argument("--out", default="../manuscript/quality_effect_curve_cadc.json")
    return parser.parse_args()


@torch.no_grad()
def heatmap_separation(model, dataset, indices):
    """Monitoring-relevant quality: how much more heatmap mass lands on
    ground-truth object cells than on empty ones."""
    device = next(model.parameters()).device
    values = []
    for i in indices:
        sample = dataset[int(i)]
        image = sample["image"].unsqueeze(0)
        pointcloud = sample["pointcloud"].unsqueeze(0)
        targets = {k: v.unsqueeze(0) for k, v in sample["targets"].items()}
        image, pointcloud, targets = _to_device(image, pointcloud, targets, device)
        out = model(image, pointcloud)
        mask = match_mask_from_heatmap(targets["H"]).bool()
        peak = out["H"].amax(dim=1, keepdim=True)
        if mask.any():
            values.append(float(peak[mask].mean().item() - peak[~mask].mean().item()))
    return float(np.mean(values)) if values else float("nan")


def main():
    args = parse_args()
    args.zip_path = None
    setting = build_real_setting(args)
    alpha = DATASET_DEFAULTS[args.dataset]["alpha"]
    device = torch.device(args.device)

    sessions = _drives_for_category(setting.dataset, "bare")
    degraded_sessions = _drives_for_category(setting.dataset, "covered")
    degraded_indices = [i for k in sorted(degraded_sessions) for i in degraded_sessions[k]]
    pick = np.random.default_rng(RNG_SEED)
    if len(degraded_indices) > args.max_degraded_frames:
        degraded_indices = sorted(pick.choice(degraded_indices, args.max_degraded_frames, replace=False))

    quality_indices = [i for k in sorted(sessions) for i in sessions[k][:5]][:args.quality_frames]
    config = CRAFXConfig(bev_h=args.bev_size, bev_w=args.bev_size, num_classes=CADC_NUM_CLASSES)

    print(f"\nQuality-vs-effect curve on CADC: {len(CHECKPOINTS)} checkpoints, "
          f"{len(sessions)} sessions, gap={GAP}, alpha={alpha}\n")
    print(f"  {'checkpoint':<12} {'separation':>11} {'loc effect':>12} {'loc CI':>22} "
          f"{'phantom effect':>15} {'phantom CI':>22}")

    rows = []
    for label, path in CHECKPOINTS:
        full_path = os.path.abspath(os.path.join(os.path.dirname(__file__), path))
        if not os.path.exists(full_path):
            print(f"  {label:<12} MISSING: {path}")
            continue

        model = CRAFX_Net(config).to(device)
        loaded = torch.load(full_path, map_location=device, weights_only=False)
        model.load_state_dict(loaded["model_state_dict"])
        model.eval()

        separation = heatmap_separation(model, setting.dataset, quality_indices)
        session_cache = {
            key: cache_both_scores(model, setting.dataset, idxs[:args.max_frames_per_session])
            for key, idxs in sessions.items()
        }
        degraded_cache = cache_both_scores(model, setting.dataset, degraded_indices)

        rng = np.random.default_rng(RNG_SEED)  # same resampling for every checkpoint
        loc = measure(session_cache, degraded_cache, alpha, 0.0, GAP, rng)
        rng = np.random.default_rng(RNG_SEED)
        phantom = measure(session_cache, degraded_cache, alpha, 1.0, GAP, rng)

        row = {
            "label": label, "checkpoint": path, "epoch": loaded.get("epoch"),
            "heatmap_separation": separation,
            "localization": loc, "phantom": phantom,
        }
        rows.append(row)
        ls = "*" if loc["excludes_zero"] else " "
        ps = "*" if phantom["excludes_zero"] else " "
        print(f"  {label:<12} {separation:>+11.5f} {loc['effect']:>+12.4f}{ls}"
              f"[{loc['ci_low']:+.4f},{loc['ci_high']:+.4f}]".rjust(22) +
              f"{phantom['effect']:>+15.4f}{ps}"
              f"[{phantom['ci_low']:+.4f},{phantom['ci_high']:+.4f}]".rjust(22))

    print("\nSHAPE")
    sig_loc = [r["label"] for r in rows if r["localization"]["excludes_zero"]]
    sig_ph = [r["label"] for r in rows if r["phantom"]["excludes_zero"]]
    print(f"  localization score significant at: {sig_loc or 'nowhere'}")
    print(f"  phantom score significant at:      {sig_ph or 'nowhere'}")
    if sig_loc and len(sig_loc) < len(rows):
        print("  -> signal is confined to a band of detector quality, not monotone in it")
    agree = set(sig_loc) == set(sig_ph)
    print(f"  the two structurally different scores agree on where signal exists: {agree}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(
            {
                "note": ("Monitorable effect vs detector quality. x-axis is heatmap "
                         "object-vs-empty separation, the monitoring-relevant quality "
                         "measure; loss and epoch decouple from it after epoch 12."),
                "alpha": alpha, "gap": GAP, "rng_seed": RNG_SEED,
                "n_sessions": len(sessions), "n_degraded_frames": len(degraded_indices),
                "rows": rows,
            },
            f, indent=2,
        )
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
