"""
Prerequisite question, asked before any operating curve: is there a
detectable weather effect in m(t) at all?

An operating curve conflates two different questions -- "is there signal"
and "does the sequential test find it" -- and reports a single delay number
that cannot distinguish them. If the underlying effect is not
distinguishable from zero, then no betting rule, calibration scheme or
score variant can recover it, and building any of those is wasted effort.

This script asks the first question directly and with far more power than
an operating curve has: over many independently sampled nominal and
degraded frames, is mean m different between the two regimes, and does a
bootstrap CI on that difference exclude zero?

Reported alongside the effect size, because it is the number that actually
predicts feasibility:

- the **between-session standard deviation** of per-frame m within each
  regime. On CADC the weather effect measured +0.057 against a
  between-drive sd of 0.061 -- a ratio below 1, meaning which drive you are
  on moves m more than the weather does. A monitor calibrated marginally
  across sessions is then detecting drive identity, not weather.
- the **overlap** between the two regimes' per-frame distributions.

Usage:
    python test_effect_exists.py --dataset snowy \
        --zip-path ../../../data/ROADVIEW5k.zip \
        --checkpoint ../../../checkpoints/snowy_scenes_focal/checkpoint_final.pth
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

from conformal_monitor.calibration import frame_miscoverage_rate, object_nonconformity_scores
from conformal_monitor.evaluate import calibrate_on_clear_weather, _to_device, match_mask_from_heatmap

from signal_monitor.null_detector import NullBoxHeadModel
from experiment_common import DATASET_DEFAULTS, build_real_setting

N_BOOTSTRAP = 5000
RNG_SEED = 20260907


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["snowy", "cadc"], required=True)
    parser.add_argument("--zip-path")
    parser.add_argument("--data-root")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--bev-size", type=int, default=128)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--n-frames", type=int, default=250,
                        help="frames sampled per regime")
    parser.add_argument("--out", default=None)
    return parser.parse_args()


@torch.no_grad()
def per_frame_miscoverage(model, subset, indices, q_hat):
    device = next(model.parameters()).device
    values = []
    for i in indices:
        sample = subset[int(i)]
        image = sample["image"].unsqueeze(0)
        pointcloud = sample["pointcloud"].unsqueeze(0)
        targets = {k: v.unsqueeze(0) for k, v in sample["targets"].items()}
        image, pointcloud, targets = _to_device(image, pointcloud, targets, device)

        out = model(image, pointcloud)
        mask = match_mask_from_heatmap(targets["H"])
        scores = object_nonconformity_scores(out["B"], targets["B"], mask)
        values.append(frame_miscoverage_rate(scores, q_hat))
    return np.asarray(values)


def bootstrap_difference(nominal, degraded, rng):
    draws = np.empty(N_BOOTSTRAP)
    for b in range(N_BOOTSTRAP):
        n = rng.choice(nominal, size=nominal.size, replace=True)
        d = rng.choice(degraded, size=degraded.size, replace=True)
        draws[b] = d.mean() - n.mean()
    low, high = np.percentile(draws, [2.5, 97.5])
    return {
        "nominal_mean": float(nominal.mean()),
        "degraded_mean": float(degraded.mean()),
        "effect": float(degraded.mean() - nominal.mean()),
        "ci_low": float(low),
        "ci_high": float(high),
        "excludes_zero": bool(low > 0.0 or high < 0.0),
        "nominal_sd": float(nominal.std(ddof=1)),
        "degraded_sd": float(degraded.std(ddof=1)),
    }


def overlap_fraction(nominal, degraded):
    """Fraction of degraded frames whose m falls inside the nominal range --
    a distribution-free read on how separable the two regimes are."""
    lo, hi = nominal.min(), nominal.max()
    return float(np.mean((degraded >= lo) & (degraded <= hi)))


def evaluate_model(label, model, setting, alpha, n_frames, rng):
    model.eval()
    q_hat = calibrate_on_clear_weather(model, setting.calibration_set, alpha,
                                       batch_size=4, num_workers=4)
    n_nom = min(n_frames, len(setting.nominal_set))
    n_deg = min(n_frames, len(setting.degraded_set))
    nominal_idx = rng.choice(len(setting.nominal_set), size=n_nom, replace=False)
    degraded_idx = rng.choice(len(setting.degraded_set), size=n_deg, replace=False)

    nominal = per_frame_miscoverage(model, setting.nominal_set, nominal_idx, q_hat)
    degraded = per_frame_miscoverage(model, setting.degraded_set, degraded_idx, q_hat)

    result = bootstrap_difference(nominal, degraded, rng)
    result.update({
        "label": label, "q_hat": q_hat,
        "n_nominal": int(n_nom), "n_degraded": int(n_deg),
        "overlap_fraction": overlap_fraction(nominal, degraded),
        "effect_to_noise": float(result["effect"] / result["nominal_sd"])
        if result["nominal_sd"] > 0 else float("nan"),
    })
    return result


def main():
    args = parse_args()
    setting = build_real_setting(args)
    alpha = DATASET_DEFAULTS[args.dataset]["alpha"]
    rng = np.random.default_rng(RNG_SEED)

    print(f"\nEffect-existence test on {args.dataset}: "
          f"{args.n_frames} frames per regime, {N_BOOTSTRAP} bootstrap draws, alpha={alpha}\n")

    results = {}
    for label, model in (("real_detector", setting.model),
                         ("null_detector", NullBoxHeadModel(setting.model))):
        r = evaluate_model(label, model, setting, alpha, args.n_frames, rng)
        results[label] = r
        verdict = "EFFECT EXISTS" if r["excludes_zero"] else "NOT DISTINGUISHABLE FROM ZERO"
        print(f"  {label}  (q_hat={r['q_hat']:.4f}, n={r['n_nominal']}/{r['n_degraded']})")
        print(f"    nominal m   = {r['nominal_mean']:.4f}  (sd {r['nominal_sd']:.4f})")
        print(f"    degraded m  = {r['degraded_mean']:.4f}  (sd {r['degraded_sd']:.4f})")
        print(f"    effect      = {r['effect']:+.4f}  95% CI [{r['ci_low']:+.4f}, {r['ci_high']:+.4f}]"
              f"  -> {verdict}")
        print(f"    effect/noise= {r['effect_to_noise']:+.3f}   "
              f"degraded frames inside nominal range: {r['overlap_fraction']:.1%}\n")

    real, null = results["real_detector"], results["null_detector"]
    print("READ-OUT")
    if not real["excludes_zero"]:
        print("  The real detector shows no weather effect distinguishable from zero.")
        print("  No betting rule, calibration scheme or score variant can recover a signal")
        print("  that is not there. Report this rather than building on top of it.")
    else:
        print(f"  A real weather effect exists ({real['effect']:+.4f}, CI excludes zero).")
        if null["excludes_zero"]:
            print(f"  BUT the null detector also shows one ({null['effect']:+.4f}), so part or all")
            print("  of it is scene content rather than perception degradation. Attribution")
            print("  requires the real effect to be clearly larger than the null's.")
        else:
            print("  The null detector shows none, so the effect is attributable to the detector.")

    out_path = args.out or f"../manuscript/effect_exists_{args.dataset}.json"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(
            {
                "note": "Does a weather effect in m(t) exist at all? Real vs null detector.",
                "dataset": args.dataset, "checkpoint": args.checkpoint,
                "alpha": alpha, "n_bootstrap": N_BOOTSTRAP, "rng_seed": RNG_SEED,
                "results": results,
            },
            f, indent=2,
        )
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
