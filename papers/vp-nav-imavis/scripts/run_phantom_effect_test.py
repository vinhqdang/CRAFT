"""
Lever 3: does the phantom-aware score find weather signal that the
matched-cell score is structurally blind to?

Every previous attempt varied *how we bet* on the same measured quantity.
This changes *what is measured*. The existing nonconformity score is box
error restricted to ground-truth object cells, so it cannot see phantom
detections in empty space -- which is snow's documented physical mechanism
(spurious near-range LiDAR returns). It is entirely possible that box error
at object cells is flat under snow, because a converged detector localizes
real objects well regardless, while activation in empty space rises
sharply; only the second is visible to the phantom score.

The converged checkpoint makes this newly testable: it produces 4.7x more
high-confidence phantom activations than the 5-epoch one (0.047% vs 0.010%
of empty cells above 0.5).

Protocol matches the effect-existence gate exactly -- session-conditional
quantiles, session-paired scoring, session-level bootstrap, temporal-gap
control -- so the numbers are directly comparable against the pre-fixed
reference of +0.074 effect / 0.85 effect-to-noise. The phantom weight is
swept, with w=0 reproducing the localization-only score as an internal
control.

Both score families are calibrated per session at level alpha, and the
frame-level rate is their convex combination

    m = (1 - w) * m_loc + w * m_phantom,

so each component has expectation at most alpha under the nominal
distribution and so does the combination.

On the null arm: `NullAllHeadsModel` is used rather than
`NullBoxHeadModel`, because the phantom score reads the heatmap and zeroing
only the box head would leave the detector's full heatmap signal intact.
The consequence is that the null's phantom component is degenerate by
construction (a constant heatmap makes every phantom score identical), so
for w > 0 the null is a floor rather than an informative comparison, and
the gap sweep and CIs carry the weight.

Usage:
    python run_phantom_effect_test.py --data-root ../../../data/cadcd \
        --checkpoint ../../../checkpoints/cadc_converged/checkpoint_final.pth \
        --label converged
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

from signal_monitor.null_detector import NullAllHeadsModel
from signal_monitor.phantom_score import phantom_nonconformity_scores
from experiment_common import DATASET_DEFAULTS, build_real_setting, _drives_for_category

PHANTOM_WEIGHTS = [0.0, 0.25, 0.5, 0.75, 1.0]
GAPS = [0, 10, 20]
N_CALIBRATION_PER_SESSION = 20
N_BOOTSTRAP = 3000
RNG_SEED = 20260908


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["cadc"], default="cadc")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--label", required=True, help="checkpoint label for the output file")
    parser.add_argument("--split", default="val")
    parser.add_argument("--bev-size", type=int, default=128)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-frames-per-session", type=int, default=60)
    parser.add_argument("--max-degraded-frames", type=int, default=300)
    parser.add_argument("--out", default=None)
    return parser.parse_args()


@torch.no_grad()
def cache_both_scores(model, dataset, indices):
    """Per-frame (localization scores, phantom scores), computed once."""
    device = next(model.parameters()).device
    loc, phantom = [], []
    for i in indices:
        sample = dataset[int(i)]
        image = sample["image"].unsqueeze(0)
        pointcloud = sample["pointcloud"].unsqueeze(0)
        targets = {k: v.unsqueeze(0) for k, v in sample["targets"].items()}
        image, pointcloud, targets = _to_device(image, pointcloud, targets, device)
        out = model(image, pointcloud)
        mask = match_mask_from_heatmap(targets["H"])
        loc.append(object_nonconformity_scores(out["B"], targets["B"], mask))
        phantom.append(phantom_nonconformity_scores(out["H"], mask))
    return loc, phantom


def combined_m(loc_scores, phantom_scores, q_loc, q_phantom, w):
    m_loc = frame_miscoverage_rate(loc_scores, q_loc)
    if w == 0.0:
        return m_loc
    m_phantom = frame_miscoverage_rate(phantom_scores, q_phantom)
    return (1.0 - w) * m_loc + w * m_phantom


def measure(session_cache, degraded_cache, alpha, w, gap, rng):
    """Session-paired effect for one phantom weight and gap."""
    effects, nominals = [], []
    degraded_loc, degraded_phantom = degraded_cache

    for loc, phantom in session_cache.values():
        cal_loc, cal_phantom = loc[:N_CALIBRATION_PER_SESSION], phantom[:N_CALIBRATION_PER_SESSION]
        mon_loc = loc[N_CALIBRATION_PER_SESSION + gap:]
        mon_phantom = phantom[N_CALIBRATION_PER_SESSION + gap:]
        if len(cal_loc) < 10 or len(mon_loc) < 5:
            continue

        q_loc = calibrate_quantile(np.concatenate(cal_loc), alpha)
        q_phantom = calibrate_quantile(np.concatenate(cal_phantom), alpha)

        nominal_m = float(np.mean([
            combined_m(l, p, q_loc, q_phantom, w) for l, p in zip(mon_loc, mon_phantom)
        ]))
        degraded_m = float(np.mean([
            combined_m(l, p, q_loc, q_phantom, w)
            for l, p in zip(degraded_loc, degraded_phantom)
        ]))
        effects.append(degraded_m - nominal_m)
        nominals.append(nominal_m)

    if not effects:
        return None
    effects = np.asarray(effects)
    nominals = np.asarray(nominals)
    draws = np.array([rng.choice(effects, effects.size, replace=True).mean()
                      for _ in range(N_BOOTSTRAP)])
    lo, hi = float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))
    sd = float(nominals.std(ddof=1)) if nominals.size > 1 else float("nan")
    effect = float(effects.mean())
    return {
        "phantom_weight": w, "gap": gap, "n_sessions": int(effects.size),
        "nominal_mean": float(nominals.mean()),
        "degraded_mean": float(nominals.mean() + effect),
        "effect": effect, "ci_low": lo, "ci_high": hi,
        "excludes_zero": bool(lo > 0.0 or hi < 0.0),
        "between_session_sd": sd,
        "effect_to_noise": effect / sd if sd and sd > 0 else float("nan"),
    }


def main():
    args = parse_args()
    args.zip_path = None
    setting = build_real_setting(args)
    alpha = DATASET_DEFAULTS[args.dataset]["alpha"]
    rng = np.random.default_rng(RNG_SEED)

    sessions = _drives_for_category(setting.dataset, "bare")
    degraded_sessions = _drives_for_category(setting.dataset, "covered")
    degraded_indices = [i for k in sorted(degraded_sessions) for i in degraded_sessions[k]]
    pick = np.random.default_rng(RNG_SEED)
    if len(degraded_indices) > args.max_degraded_frames:
        degraded_indices = sorted(pick.choice(degraded_indices, args.max_degraded_frames, replace=False))

    print(f"\nPhantom-score effect test [{args.label}]: {len(sessions)} sessions, "
          f"{len(degraded_indices)} degraded frames, alpha={alpha}")
    print("Reference to beat (localization score, 5-epoch): effect +0.0742, eff/noise 0.849\n")

    all_results = {}
    for arm_label, model in (("real_detector", setting.model),
                             ("null_all_heads", NullAllHeadsModel(setting.model))):
        model.eval()
        session_cache = {
            key: cache_both_scores(model, setting.dataset, idxs[:args.max_frames_per_session])
            for key, idxs in sessions.items()
        }
        degraded_cache = cache_both_scores(model, setting.dataset, degraded_indices)

        print(f"=== {arm_label} ===")
        print(f"  {'w':>5} {'gap':>4} {'nominal':>9} {'degraded':>9} {'effect':>9} "
              f"{'95% CI':>22} {'eff/noise':>10}")
        rows = []
        for w in PHANTOM_WEIGHTS:
            for gap in GAPS:
                r = measure(session_cache, degraded_cache, alpha, w, gap, rng)
                if r is None:
                    continue
                rows.append(r)
                star = "*" if r["excludes_zero"] else " "
                print(f"  {w:>5.2f} {gap:>4} {r['nominal_mean']:>9.4f} {r['degraded_mean']:>9.4f} "
                      f"{r['effect']:>+9.4f}{star} [{r['ci_low']:+.4f},{r['ci_high']:+.4f}] "
                      f"{r['effect_to_noise']:>10.3f}")
            print()
        all_results[arm_label] = rows

    real = all_results["real_detector"]
    significant = [r for r in real if r["excludes_zero"]]
    print("READ-OUT")
    if significant:
        best = max(significant, key=lambda r: abs(r["effect_to_noise"]))
        print(f"  Significant at w={best['phantom_weight']}, gap={best['gap']}: "
              f"effect {best['effect']:+.4f} [{best['ci_low']:+.4f},{best['ci_high']:+.4f}], "
              f"eff/noise {best['effect_to_noise']:.3f}")
        survives = [r for r in significant if r["gap"] == max(GAPS)]
        print(f"  Survives the largest gap: {bool(survives)}")
    else:
        print("  NO phantom weight at any gap produces an effect whose CI excludes zero.")
        print("  The phantom component does not expose weather signal the localization")
        print("  score misses, on this checkpoint.")

    out_path = args.out or f"../manuscript/phantom_effect_{args.label}.json"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(
            {
                "note": "Phantom-aware score effect-existence test, session-paired, gap-controlled.",
                "checkpoint": args.checkpoint, "label": args.label, "alpha": alpha,
                "phantom_weights": PHANTOM_WEIGHTS, "gaps": GAPS,
                "n_bootstrap": N_BOOTSTRAP, "rng_seed": RNG_SEED,
                "reference_localization_5epoch": {"effect": 0.0742, "effect_to_noise": 0.849},
                "results": all_results,
            },
            f, indent=2,
        )
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
