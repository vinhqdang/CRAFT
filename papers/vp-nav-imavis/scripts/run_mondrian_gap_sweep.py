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
stretch of a drive, wait, then begin monitoring. This script sweeps N.

## Session-paired scoring

Each nominal session s gets its own quantile q_s from its own leading
frames. The effect for that session is then

    effect_s = mean_degraded_m(q_s) - mean_nominal_m(q_s | s's held-out frames)

i.e. *the same quantile scores both sides*. This matters: an earlier
version of this script scored degraded frames against the mean of all
session quantiles while scoring each nominal session against its own. That
is asymmetric, and biased toward zero -- the mean session quantile is
larger than the pooled one, which suppresses degraded miscoverage and
shrinks the measured gap. It also contradicted the design documented in
`signal_monitor/mondrian.py`, which specifies scoring degraded frames
against the quantile of the session the stream started in. Session-paired
scoring is both the documented design and the deployment-faithful one: a
vehicle calibrates on its current session's clear portion and alarms when
incoming frames stop matching *that* calibration.

Bootstrapping is over sessions, not frames, since sessions are the
independent unit here -- frames within a drive are heavily correlated and
bootstrapping over them would understate the interval badly.

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

from conformal_monitor.calibration import (
    calibrate_quantile,
    frame_miscoverage_rate,
    object_nonconformity_scores,
)
from conformal_monitor.evaluate import _to_device, match_mask_from_heatmap

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
    parser.add_argument("--max-frames-per-session", type=int, default=60)
    parser.add_argument("--max-degraded-frames", type=int, default=300)
    parser.add_argument("--out", default="../manuscript/mondrian_gap_sweep_cadc.json")
    return parser.parse_args()


@torch.no_grad()
def cache_scores(model, dataset, indices):
    """
    Per-frame nonconformity score arrays, computed once.

    Scores do not depend on the quantile, so caching them lets the whole
    gap x session-quantile sweep run without re-invoking the model -- the
    difference between a few thousand forward passes and a few hundred
    thousand.
    """
    device = next(model.parameters()).device
    out_scores = []
    for i in indices:
        sample = dataset[int(i)]
        image = sample["image"].unsqueeze(0)
        pointcloud = sample["pointcloud"].unsqueeze(0)
        targets = {k: v.unsqueeze(0) for k, v in sample["targets"].items()}
        image, pointcloud, targets = _to_device(image, pointcloud, targets, device)
        out = model(image, pointcloud)
        mask = match_mask_from_heatmap(targets["H"])
        out_scores.append(object_nonconformity_scores(out["B"], targets["B"], mask))
    return out_scores


def measure(nominal_cache, degraded_scores, alpha, gap, rng, calibration_mode="mondrian"):
    """
    Session-paired effect at one gap size, bootstrapped over sessions.

    `calibration_mode` selects which quantile scores each session:
      "mondrian" -> that session's own quantile (session-conditional)
      "marginal" -> one pooled quantile from every session's calibration
                    frames (standard split conformal)

    Both modes run over the identical sessions, identical frames, identical
    gap and identical cached scores, so the comparison isolates the
    calibration scheme and nothing else. Without this matched baseline an
    effect/noise number from the Mondrian arm cannot be called an
    improvement over anything.
    """
    pooled_q = None
    if calibration_mode == "marginal":
        pooled_calibration = [
            s for scores in nominal_cache.values() for s in scores[:N_CALIBRATION_PER_SESSION]
        ]
        pooled_q = calibrate_quantile(np.concatenate(pooled_calibration), alpha)

    per_session_effect, per_session_nominal, quantiles = [], [], []

    for key, scores in nominal_cache.items():
        calibration = scores[:N_CALIBRATION_PER_SESSION]
        monitored = scores[N_CALIBRATION_PER_SESSION + gap:]
        if len(calibration) < 10 or len(monitored) < 5:
            continue
        q_s = pooled_q if pooled_q is not None else calibrate_quantile(
            np.concatenate(calibration), alpha
        )
        nominal_m = float(np.mean([frame_miscoverage_rate(s, q_s) for s in monitored]))
        # Same quantile scores the degraded side -- the session the vehicle
        # believes it is in.
        degraded_m = float(np.mean([frame_miscoverage_rate(s, q_s) for s in degraded_scores]))
        per_session_effect.append(degraded_m - nominal_m)
        per_session_nominal.append(nominal_m)
        quantiles.append(q_s)

    effects = np.asarray(per_session_effect)
    nominals = np.asarray(per_session_nominal)
    if effects.size == 0:
        return None

    draws = np.array([rng.choice(effects, effects.size, replace=True).mean()
                      for _ in range(N_BOOTSTRAP)])
    lo, hi = float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))
    sd = float(nominals.std(ddof=1)) if nominals.size > 1 else float("nan")
    effect = float(effects.mean())

    return {
        "gap": gap,
        "calibration_mode": calibration_mode,
        "n_sessions": int(effects.size),
        "nominal_mean": float(nominals.mean()),
        "degraded_mean": float(nominals.mean() + effect),
        "effect": effect,
        "ci_low": lo,
        "ci_high": hi,
        "excludes_zero": bool(lo > 0.0 or hi < 0.0),
        "between_session_sd": sd,
        "effect_to_noise": effect / sd if sd and sd > 0 else float("nan"),
        "worst_session_deviation_from_alpha": float(max(abs(nominals - alpha))),
        "mean_session_quantile": float(np.mean(quantiles)),
    }


def main():
    args = parse_args()
    args.zip_path = None
    setting = build_real_setting(args)
    alpha = DATASET_DEFAULTS[args.dataset]["alpha"]
    rng = np.random.default_rng(RNG_SEED)

    nominal_sessions = _drives_for_category(setting.dataset, "bare")
    degraded_sessions = _drives_for_category(setting.dataset, "covered")
    degraded_indices = [i for k in sorted(degraded_sessions) for i in degraded_sessions[k]]
    rng_pick = np.random.default_rng(RNG_SEED)
    if len(degraded_indices) > args.max_degraded_frames:
        degraded_indices = list(rng_pick.choice(degraded_indices, args.max_degraded_frames, replace=False))

    print(f"\n{len(nominal_sessions)} nominal sessions, {len(degraded_indices)} degraded frames")
    print(f"calibration frames/session: {N_CALIBRATION_PER_SESSION}, "
          f"monitored capped at {args.max_frames_per_session}, bootstrap over sessions\n")

    all_results = {}
    for label, model in (("real_detector", setting.model),
                         ("null_detector", NullBoxHeadModel(setting.model))):
        model.eval()
        nominal_cache = {
            key: cache_scores(model, setting.dataset, idxs[:args.max_frames_per_session])
            for key, idxs in nominal_sessions.items()
        }
        degraded_scores = cache_scores(model, setting.dataset, degraded_indices)

        rows = []
        for mode in ("marginal", "mondrian"):
            print(f"=== {label} / {mode} calibration ===")
            print(f"  {'gap':>4} {'n':>3} {'nominal':>9} {'degraded':>9} {'effect':>9} "
                  f"{'95% CI':>22} {'sd':>7} {'eff/noise':>10} {'worst|m-a|':>11}")
            for gap in GAPS:
                r = measure(nominal_cache, degraded_scores, alpha, gap, rng, calibration_mode=mode)
                if r is None:
                    continue
                rows.append(r)
                star = "*" if r["excludes_zero"] else " "
                print(f"  {gap:>4} {r['n_sessions']:>3} {r['nominal_mean']:>9.4f} "
                      f"{r['degraded_mean']:>9.4f} {r['effect']:>+9.4f}{star} "
                      f"[{r['ci_low']:+.4f},{r['ci_high']:+.4f}] {r['between_session_sd']:>7.4f} "
                      f"{r['effect_to_noise']:>10.3f} {r['worst_session_deviation_from_alpha']:>11.4f}")
            print()
        all_results[label] = rows

    real = [r for r in all_results["real_detector"] if r["calibration_mode"] == "mondrian"]
    marginal = [r for r in all_results["real_detector"] if r["calibration_mode"] == "marginal"]
    if marginal and real:
        print("MONDRIAN vs MARGINAL, matched sessions/frames/gap")
        print(f"  {'gap':>4} {'marginal eff/noise':>20} {'mondrian eff/noise':>20} "
              f"{'marginal worst|m-a|':>21} {'mondrian worst|m-a|':>21}")
        for m, d in zip(marginal, real):
            print(f"  {m['gap']:>4} {m['effect_to_noise']:>20.3f} {d['effect_to_noise']:>20.3f} "
                  f"{m['worst_session_deviation_from_alpha']:>21.4f} "
                  f"{d['worst_session_deviation_from_alpha']:>21.4f}")
        print()
    if real:
        first, last = real[0], real[-1]
        shrink = first["effect"] - last["effect"]
        print("READ-OUT")
        print(f"  gap=0:  effect {first['effect']:+.4f}  eff/noise {first['effect_to_noise']:.3f}"
              f"  {'(CI excludes 0)' if first['excludes_zero'] else '(CI includes 0)'}")
        print(f"  gap={last['gap']:<2}: effect {last['effect']:+.4f}  eff/noise {last['effect_to_noise']:.3f}"
              f"  {'(CI excludes 0)' if last['excludes_zero'] else '(CI includes 0)'}")
        if not last["excludes_zero"]:
            print("  Effect does NOT survive the largest gap.")
        elif abs(shrink) < 0.25 * abs(first["effect"]):
            print("  Effect stable across gaps - adjacency was not inflating it materially.")
        else:
            print("  Effect shrinks materially with the gap - use the gapped number as honest.")
        null_rows = all_results["null_detector"]
        print(f"  null detector shows an effect at any gap: "
              f"{any(r['excludes_zero'] for r in null_rows)}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(
            {
                "note": ("Session-paired session-conditional effect vs temporal gap. "
                         "Same quantile scores both regimes; bootstrap over sessions."),
                "checkpoint": args.checkpoint, "alpha": alpha,
                "n_calibration_per_session": N_CALIBRATION_PER_SESSION,
                "max_frames_per_session": args.max_frames_per_session,
                "n_degraded_frames": len(degraded_indices),
                "gaps": GAPS, "n_bootstrap": N_BOOTSTRAP, "rng_seed": RNG_SEED,
                "results": all_results,
            },
            f, indent=2,
        )
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
