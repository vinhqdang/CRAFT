"""
Does monitoring difficulty scale inversely with detector quality?

The 40-frame gate diagnostic showed the onset jump in m(t) shrinking as
training progressed (epoch 0: +0.248, epoch 4: +0.124). The hypothesis
worth testing is that this is real and mechanistic rather than noise: a
better-trained detector degrades more *gracefully* under snow, so there is
genuinely less nominal-vs-degraded error differential left for the monitor
to detect. If true, that is a reportable coupling with direct deployment
consequences -- improving the detector makes the monitor's job harder.

Two things this script does that the gate diagnostic did not:

1. **A curve, not two points.** Every available checkpoint is measured, so
   the trend can be seen across the whole quality range rather than
   inferred from its endpoints.
2. **Bootstrap confidence intervals on a large frame sample.** 40 frames is
   too thin to separate a +0.40 -> +0.28 shift from sampling noise, so the
   jump is resampled over a substantially larger set and reported as an
   interval.

On the "subtract the content artifact" decomposition: the zeros branch is
genuinely checkpoint-independent (it depends only on the targets), and this
script verifies that rather than assuming it. But subtracting its jump from
a real checkpoint's is a heuristic, not an exact decomposition -- the two
branches are thresholded at different calibrated quantiles (q_hat ~1.7 for
a trained checkpoint versus ~10.2 for zeros), so the content artifact does
not enter additively. Both the raw jump and the artifact-subtracted value
are reported, with the raw jump treated as primary.

Usage:
    python diagnose_quality_vs_signal.py --dataset snowy \
        --zip-path ../../../data/ROADVIEW5k.zip \
        --checkpoints ../../../checkpoints/snowy_scenes_fixed/checkpoint_final.pth \
                      ../../../checkpoints/snowy_scenes_focal/checkpoint_epoch0.pth \
                      ../../../checkpoints/snowy_scenes_focal/checkpoint_epoch4.pth \
        --n-frames 150
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

from conformal_monitor.calibration import (
    calibrate_quantile,
    frame_miscoverage_rate,
    object_nonconformity_scores,
)
from conformal_monitor.evaluate import _to_device, match_mask_from_heatmap

from experiment_common import DATASET_DEFAULTS, build_real_setting

N_BOOTSTRAP = 2000
RNG_SEED = 20260907


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["snowy", "cadc"], required=True)
    parser.add_argument("--zip-path")
    parser.add_argument("--data-root")
    parser.add_argument("--checkpoint", required=True,
                        help="any checkpoint; only used to build the dataset splits")
    parser.add_argument("--checkpoints", nargs="+", required=True,
                        help="checkpoints to measure, in quality order")
    parser.add_argument("--labels", nargs="+", default=None,
                        help="display labels, one per checkpoint")
    parser.add_argument("--split", default="val")
    parser.add_argument("--bev-size", type=int, default=128)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--n-frames", type=int, default=150,
                        help="frames sampled per regime")
    parser.add_argument("--out", default=None)
    return parser.parse_args()


@torch.no_grad()
def _per_frame_scores(model, subset, n_frames, zero_prediction=False):
    """Per-frame nonconformity score arrays, plus detector-quality stats."""
    device = next(model.parameters()).device
    per_frame, separations, box_stds = [], [], []

    for i in range(min(n_frames, len(subset))):
        sample = subset[i]
        image = sample["image"].unsqueeze(0)
        pointcloud = sample["pointcloud"].unsqueeze(0)
        targets = {k: v.unsqueeze(0) for k, v in sample["targets"].items()}
        image, pointcloud, targets = _to_device(image, pointcloud, targets, device)

        out = model(image, pointcloud)
        box_pred = torch.zeros_like(out["B"]) if zero_prediction else out["B"]
        match_mask = match_mask_from_heatmap(targets["H"])
        per_frame.append(object_nonconformity_scores(box_pred, targets["B"], match_mask))

        gt = match_mask.bool()
        peak = out["H"].amax(dim=1, keepdim=True)
        if gt.any():
            separations.append(float(peak[gt].mean().item() - peak[~gt].mean().item()))
        box_stds.append(float(out["B"].std().item()))

    quality = {
        "heatmap_separation": float(np.mean(separations)) if separations else float("nan"),
        "box_pred_std": float(np.mean(box_stds)),
    }
    return per_frame, quality


def _bootstrap_jump(nominal_m, degraded_m, rng):
    """Percentile bootstrap CI for (mean degraded m) - (mean nominal m),
    resampling frames independently within each regime."""
    nominal_m = np.asarray(nominal_m)
    degraded_m = np.asarray(degraded_m)
    draws = np.empty(N_BOOTSTRAP)
    for b in range(N_BOOTSTRAP):
        n = rng.choice(nominal_m, size=nominal_m.size, replace=True)
        d = rng.choice(degraded_m, size=degraded_m.size, replace=True)
        draws[b] = d.mean() - n.mean()
    return {
        "jump": float(degraded_m.mean() - nominal_m.mean()),
        "ci_low": float(np.percentile(draws, 2.5)),
        "ci_high": float(np.percentile(draws, 97.5)),
        "nominal_m": float(nominal_m.mean()),
        "degraded_m": float(degraded_m.mean()),
    }


def measure_checkpoint(setting, checkpoint_path, config, alpha, n_frames, device, rng,
                       zero_prediction=False):
    model = CRAFX_Net(config).to(device)
    loaded = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(loaded["model_state_dict"])
    model.eval()

    calibration, _ = _per_frame_scores(model, setting.calibration_set, len(setting.calibration_set),
                                       zero_prediction)
    q_hat = calibrate_quantile(np.concatenate(calibration), alpha)

    nominal_scores, quality = _per_frame_scores(model, setting.nominal_set, n_frames, zero_prediction)
    degraded_scores, _ = _per_frame_scores(model, setting.degraded_set, n_frames, zero_prediction)

    nominal_m = [frame_miscoverage_rate(s, q_hat) for s in nominal_scores]
    degraded_m = [frame_miscoverage_rate(s, q_hat) for s in degraded_scores]

    result = _bootstrap_jump(nominal_m, degraded_m, rng)
    result.update({"q_hat": q_hat, "epoch": loaded.get("epoch"), **quality})
    return result


def main():
    args = parse_args()
    setting = build_real_setting(args)
    defaults = DATASET_DEFAULTS[args.dataset]
    alpha = defaults["alpha"]
    device = torch.device(args.device)
    config = setting.model.config if hasattr(setting.model, "config") else None
    rng = np.random.default_rng(RNG_SEED)

    labels = args.labels or [os.path.basename(os.path.dirname(p)) + "/" + os.path.basename(p)
                             for p in args.checkpoints]
    if len(labels) != len(args.checkpoints):
        raise ValueError("--labels must have one entry per checkpoint")

    print(f"\nMeasuring {len(args.checkpoints)} checkpoints on {args.n_frames} frames per regime, "
          f"{N_BOOTSTRAP} bootstrap draws\n")

    # The zeros branch depends only on the targets, so it should be identical
    # for every checkpoint. Verified across two checkpoints rather than assumed.
    zeros_reference = measure_checkpoint(
        setting, args.checkpoints[0], config, alpha, args.n_frames, device, rng, zero_prediction=True
    )
    zeros_check = measure_checkpoint(
        setting, args.checkpoints[-1], config, alpha, args.n_frames, device, rng, zero_prediction=True
    )
    artifact = zeros_reference["jump"]
    identical = np.isclose(zeros_reference["jump"], zeros_check["jump"], atol=1e-9)
    print(f"content artifact (zeros branch): jump={artifact:+.5f} "
          f"[{zeros_reference['ci_low']:+.5f}, {zeros_reference['ci_high']:+.5f}]")
    print(f"  checkpoint-independent: {identical} "
          f"(first={zeros_reference['jump']:+.8f}, last={zeros_check['jump']:+.8f})\n")

    rows = []
    header = f"  {'checkpoint':<22} {'sep':>9} {'q_hat':>8} {'jump':>9} {'95% CI':>20} {'-artifact':>10}"
    print(header)
    for label, path in zip(labels, args.checkpoints):
        r = measure_checkpoint(setting, path, config, alpha, args.n_frames, device, rng)
        r["label"] = label
        r["jump_minus_artifact"] = r["jump"] - artifact
        rows.append(r)
        print(f"  {label:<22} {r['heatmap_separation']:>+9.5f} {r['q_hat']:>8.3f} "
              f"{r['jump']:>+9.5f} [{r['ci_low']:+.4f},{r['ci_high']:+.4f}] "
              f"{r['jump_minus_artifact']:>+10.5f}")

    # Do the endpoint intervals overlap? If they do, the apparent trend is
    # not separable from sampling noise at this sample size.
    first, last = rows[0], rows[-1]
    overlap = not (first["ci_low"] > last["ci_high"] or last["ci_low"] > first["ci_high"])
    print(f"\n  {first['label']} vs {last['label']}: "
          f"CIs {'OVERLAP -> trend not separable from noise' if overlap else 'DISJOINT -> trend is real'}")

    out_path = args.out or f"../manuscript/quality_vs_signal_{args.dataset}.json"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(
            {
                "note": "Detector quality vs. monitorable onset signal, bootstrap CIs.",
                "dataset": args.dataset,
                "n_frames_per_regime": args.n_frames,
                "n_bootstrap": N_BOOTSTRAP,
                "rng_seed": RNG_SEED,
                "alpha": alpha,
                "content_artifact_jump": artifact,
                "content_artifact_checkpoint_independent": bool(identical),
                "checkpoints": rows,
                "endpoint_cis_overlap": bool(overlap),
            },
            f, indent=2,
        )
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
