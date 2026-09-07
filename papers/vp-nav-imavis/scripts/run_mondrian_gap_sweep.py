"""
Does the CADC weather effect survive a temporal gap between the
calibration and monitored portions of each session?

Session-conditional calibration requires a within-session split (each drive
calibrates on its own leading frames), which reverses the earlier
whole-drive protocol. That earlier protocol was chosen deliberately to
avoid temporal-adjacency leakage, and the within-session split reintroduces
it: calibration and monitored frames now sit adjacent in time inside the
same drive, and adjacent driving frames are highly correlated. The quantile
is then fitted on data nearly identical to what it scores, making
nominal-side miscoverage optimistically low and *widening* the apparent
nominal-to-degraded gap -- manufacturing part of the effect being measured,
in the flattering direction.

Inserting a gap of N frames between the two portions breaks the adjacency
without damaging the deployment story: a vehicle can calibrate on the first
stretch of a drive, wait, then begin monitoring. This script sweeps N and
reports the effect at each value.

Reading it:
- effect roughly flat across gap sizes -> adjacency was not inflating it,
  the effect is real;
- effect shrinking as the gap grows -> adjacency was contributing, and the
  size of the shrink is how much.

The null detector is run at every gap size too. If within-session splitting
manufactures an apparent effect through content or adjacency rather than
detector error, the null arm should show it as well -- the control does
double duty here.

Usage:
    python run_mondrian_gap_sweep.py --data-root ../../../data/cadcd \
        --checkpoint ../../../checkpoints/cadc_focal/checkpoint_final.pth
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

from signal_monitor.mondrian import (
    per_session_miscoverage,
    session_conditional_quantiles,
)
from signal_monitor.null_detector import NullBoxHeadModel
from experiment_common import DATASET_DEFAULTS, build_real_setting, _drives_for_category

GAPS = [0, 10, 20, 40]
N_CALIBRATION_PER_SESSION = 20
N_BOOTSTRAP = 3000
RNG_SEED = 20260907


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["cadc"], default="cadc")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--bev-size", type=int, default=128)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-frames-per-session", type=int, default=40)
    parser.add_argument("--out", default="../manuscript/mondrian_gap_sweep_cadc.json")
    return parser.parse_args()


def bootstrap_ci(values, rng, n_bootstrap=N_BOOTSTRAP):
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return (float("nan"), float("nan"))
    draws = np.array([rng.choice(values, size=values.size, replace=True).mean()
                      for _ in range(n_bootstrap)])
    return float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def measure(model, setting, nominal_sessions, degraded_sessions, alpha, gap, max_frames, rng):
    """Session-conditional effect at one gap size."""
    quantiles, monitoring, pooled = session_conditional_quantiles(
        model, setting.dataset, nominal_sessions, alpha,
        n_calibration_per_session=N_CALIBRATION_PER_SESSION, temporal_gap=gap,
    )
    nominal_by_session = per_session_miscoverage(
        model, setting.dataset, monitoring, quantiles, max_frames=max_frames
    )
    # Degraded sessions carry no calibration data by construction -- they are
    # the anomaly. Each is scored against the mean nominal-session quantile,
    # standing in for "the session the vehicle believes it is in".
    nominal_q = float(np.mean(list(quantiles.values())))
    degraded_by_session = per_session_miscoverage(
        model, setting.dataset, degraded_sessions,
        {k: nominal_q for k in degraded_sessions}, max_frames=max_frames,
    )

    nominal_values = np.array([v for v in nominal_by_session.values() if np.isfinite(v)])
    degraded_values = np.array([v for v in degraded_by_session.values() if np.isfinite(v)])

    effect = float(degraded_values.mean() - nominal_values.mean())
    combined = np.concatenate([nominal_values - nominal_values.mean(),
                               degraded_values - degraded_values.mean()])
    draws = np.array([
        rng.choice(degraded_values, degraded_values.size, replace=True).mean()
        - rng.choice(nominal_values, nominal_values.size, replace=True).mean()
        for _ in range(N_BOOTSTRAP)
    ])
    lo, hi = float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))

    between_session_sd = float(nominal_values.std(ddof=1)) if nominal_values.size > 1 else float("nan")
    return {
        "gap": gap,
        "n_nominal_sessions": int(nominal_values.size),
        "n_degraded_sessions": int(degraded_values.size),
        "nominal_mean": float(nominal_values.mean()),
        "degraded_mean": float(degraded_values.mean()),
        "effect": effect,
        "ci_low": lo,
        "ci_high": hi,
        "excludes_zero": bool(lo > 0.0 or hi < 0.0),
        "between_session_sd": between_session_sd,
        "effect_to_noise": effect / between_session_sd if between_session_sd > 0 else float("nan"),
        "worst_session_deviation_from_alpha": float(
            max(abs(v - alpha) for v in nominal_by_session.values() if np.isfinite(v))
        ),
        "pooled_quantile": pooled,
        "mean_session_quantile": nominal_q,
    }


def main():
    args = parse_args()
    args.zip_path = None
    setting = build_real_setting(args)
    alpha = DATASET_DEFAULTS[args.dataset]["alpha"]
    rng = np.random.default_rng(RNG_SEED)

    nominal_sessions = _drives_for_category(setting.dataset, "bare")
    degraded_sessions = _drives_for_category(setting.dataset, "covered")
    print(f"\n{len(nominal_sessions)} nominal sessions, {len(degraded_sessions)} degraded sessions")
    print(f"calibration frames per session: {N_CALIBRATION_PER_SESSION}, "
          f"monitored frames capped at {args.max_frames_per_session}\n")

    all_results = {}
    for label, model in (("real_detector", setting.model),
                         ("null_detector", NullBoxHeadModel(setting.model))):
        model.eval()
        print(f"=== {label} ===")
        print(f"  {'gap':>4} {'nominal':>9} {'degraded':>9} {'effect':>9} {'95% CI':>22} "
              f"{'sd':>7} {'eff/noise':>10} {'worst|m-a|':>11}")
        rows = []
        for gap in GAPS:
            r = measure(model, setting, nominal_sessions, degraded_sessions,
                        alpha, gap, args.max_frames_per_session, rng)
            rows.append(r)
            star = "*" if r["excludes_zero"] else " "
            print(f"  {gap:>4} {r['nominal_mean']:>9.4f} {r['degraded_mean']:>9.4f} "
                  f"{r['effect']:>+9.4f}{star} [{r['ci_low']:+.4f},{r['ci_high']:+.4f}] "
                  f"{r['between_session_sd']:>7.4f} {r['effect_to_noise']:>10.3f} "
                  f"{r['worst_session_deviation_from_alpha']:>11.4f}")
        all_results[label] = rows
        print()

    real = all_results["real_detector"]
    first, last = real[0], real[-1]
    shrink = first["effect"] - last["effect"]
    print("READ-OUT")
    print(f"  effect at gap=0:  {first['effect']:+.4f}  (eff/noise {first['effect_to_noise']:.3f})")
    print(f"  effect at gap={last['gap']:<2}: {last['effect']:+.4f}  (eff/noise {last['effect_to_noise']:.3f})")
    print(f"  change: {-shrink:+.4f}")
    if not last["excludes_zero"]:
        print("  The effect does NOT survive the largest gap - adjacency was carrying it.")
    elif abs(shrink) < 0.25 * abs(first["effect"]):
        print("  Effect roughly stable across gaps - adjacency was not inflating it materially.")
    else:
        print("  Effect shrinks materially with the gap - adjacency contributed; use the "
              "gapped number as the honest estimate.")
    null_flags = [r["excludes_zero"] for r in all_results["null_detector"]]
    print(f"  null detector shows an effect at any gap: {any(null_flags)}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(
            {
                "note": "Session-conditional effect vs temporal gap between calibration and monitoring.",
                "checkpoint": args.checkpoint, "alpha": alpha,
                "n_calibration_per_session": N_CALIBRATION_PER_SESSION,
                "max_frames_per_session": args.max_frames_per_session,
                "gaps": GAPS, "n_bootstrap": N_BOOTSTRAP, "rng_seed": RNG_SEED,
                "results": all_results,
            },
            f, indent=2,
        )
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
