"""
Lever 2: does averaging m over blocks of k frames make a block of degraded
frames distinguishable from a block of nominal ones?

## Which noise blocking actually reduces

Lever 1 measured effect against the **between-session** standard deviation
of session-mean m. Block aggregation cannot move that number at all:
averaging m within a session leaves the session's mean unchanged, so both
the effect and the between-session sd are exactly invariant to k. A sweep
on that metric would print the same value for every block size.

What blocking reduces is the **within-session, per-block** variability of
m -- the noise governing whether a given stretch of frames looks degraded,
which is what a sequential test actually consumes. Averaging k frames cuts
that by roughly sqrt(k) when frames are independent, and by less when they
are temporally correlated, which driving frames certainly are. Measuring
how much less is the point of the sweep.

So this script reports **per-block discriminability**:

    d(k) = [mean degraded block m - mean nominal block m] / sd(nominal block m)

At k=1 this is per-frame discriminability. The lever-1 session-level
statistic is reported alongside at every k, where it should stay constant
-- an arithmetic invariant that doubles as a correctness check on the
blocking implementation.

Everything else matches lever 1: session-paired Mondrian quantiles, the
marginal arm under identical conditions, the null-detector row, the
temporal-gap control, and session-level bootstrap CIs.

Usage:
    python run_block_sweep.py --data-root ../../../data/cadcd \
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

from conformal_monitor.calibration import calibrate_quantile, frame_miscoverage_rate

from signal_monitor.null_detector import NullBoxHeadModel
from experiment_common import DATASET_DEFAULTS, build_real_setting, _drives_for_category
from run_mondrian_gap_sweep import cache_scores, N_CALIBRATION_PER_SESSION

BLOCK_SIZES = [1, 2, 4, 8]
GAP = 10
N_BOOTSTRAP = 3000
RNG_SEED = 20260908


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
    parser.add_argument("--out", default="../manuscript/block_sweep_cadc.json")
    return parser.parse_args()


def block_means(values, k):
    """Non-overlapping block means; a trailing partial block is dropped so
    every block averages exactly k frames and blocks are comparable."""
    values = np.asarray(values, dtype=float)
    n_blocks = values.size // k
    if n_blocks == 0:
        return np.zeros(0)
    return values[: n_blocks * k].reshape(n_blocks, k).mean(axis=1)


def measure(nominal_cache, degraded_scores, alpha, k, rng, calibration_mode):
    pooled_q = None
    if calibration_mode == "marginal":
        pooled_calibration = [
            s for scores in nominal_cache.values() for s in scores[:N_CALIBRATION_PER_SESSION]
        ]
        pooled_q = calibrate_quantile(np.concatenate(pooled_calibration), alpha)

    session_effects, session_means = [], []
    all_nominal_blocks, all_degraded_blocks = [], []

    for scores in nominal_cache.values():
        calibration = scores[:N_CALIBRATION_PER_SESSION]
        monitored = scores[N_CALIBRATION_PER_SESSION + GAP:]
        if len(calibration) < 10 or len(monitored) < k:
            continue
        q_s = pooled_q if pooled_q is not None else calibrate_quantile(
            np.concatenate(calibration), alpha
        )

        nominal_m = [frame_miscoverage_rate(s, q_s) for s in monitored]
        degraded_m = [frame_miscoverage_rate(s, q_s) for s in degraded_scores]

        nominal_blocks = block_means(nominal_m, k)
        degraded_blocks = block_means(degraded_m, k)
        if nominal_blocks.size == 0 or degraded_blocks.size == 0:
            continue

        all_nominal_blocks.append(nominal_blocks)
        all_degraded_blocks.append(degraded_blocks)
        session_effects.append(float(np.mean(degraded_m) - np.mean(nominal_m)))
        session_means.append(float(np.mean(nominal_m)))

    if not session_effects:
        return None

    nominal_blocks = np.concatenate(all_nominal_blocks)
    degraded_blocks = np.concatenate(all_degraded_blocks)
    block_effect = float(degraded_blocks.mean() - nominal_blocks.mean())
    block_noise = float(nominal_blocks.std(ddof=1))

    effects = np.asarray(session_effects)
    draws = np.array([rng.choice(effects, effects.size, replace=True).mean()
                      for _ in range(N_BOOTSTRAP)])
    lo, hi = float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))

    session_sd = float(np.asarray(session_means).std(ddof=1))
    # Fraction of degraded blocks above the nominal blocks' 80th percentile:
    # a threshold-free read on separability that does not assume normality.
    threshold = float(np.percentile(nominal_blocks, 80))
    separability = float(np.mean(degraded_blocks > threshold))

    return {
        "block_size": k,
        "calibration_mode": calibration_mode,
        "n_sessions": int(effects.size),
        "n_nominal_blocks": int(nominal_blocks.size),
        "n_degraded_blocks": int(degraded_blocks.size),
        "block_effect": block_effect,
        "block_noise": block_noise,
        "block_discriminability": block_effect / block_noise if block_noise > 0 else float("nan"),
        "separability_above_p80": separability,
        "session_effect": float(effects.mean()),
        "session_ci_low": lo,
        "session_ci_high": hi,
        "session_excludes_zero": bool(lo > 0.0 or hi < 0.0),
        "session_effect_to_noise": float(effects.mean() / session_sd) if session_sd > 0 else float("nan"),
    }


def main():
    args = parse_args()
    args.zip_path = None
    setting = build_real_setting(args)
    alpha = DATASET_DEFAULTS[args.dataset]["alpha"]
    rng = np.random.default_rng(RNG_SEED)

    nominal_sessions = _drives_for_category(setting.dataset, "bare")
    degraded_sessions = _drives_for_category(setting.dataset, "covered")
    degraded_indices = [i for key in sorted(degraded_sessions) for i in degraded_sessions[key]]
    pick = np.random.default_rng(RNG_SEED)
    if len(degraded_indices) > args.max_degraded_frames:
        degraded_indices = sorted(pick.choice(degraded_indices, args.max_degraded_frames, replace=False))

    print(f"\nBlock sweep on CADC: gap={GAP}, {len(nominal_sessions)} sessions, "
          f"{len(degraded_indices)} degraded frames, alpha={alpha}\n")

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
            print(f"=== {label} / {mode} ===")
            print(f"  {'k':>3} {'blk effect':>11} {'blk noise':>10} {'discrim':>9} "
                  f"{'sep>p80':>8} {'session eff':>12} {'session e/n':>12}")
            for k in BLOCK_SIZES:
                r = measure(nominal_cache, degraded_scores, alpha, k, rng, mode)
                if r is None:
                    continue
                rows.append(r)
                print(f"  {k:>3} {r['block_effect']:>+11.4f} {r['block_noise']:>10.4f} "
                      f"{r['block_discriminability']:>9.3f} {r['separability_above_p80']:>8.1%} "
                      f"{r['session_effect']:>+12.4f} {r['session_effect_to_noise']:>12.3f}")
            print()
        all_results[label] = rows

    real_mondrian = [r for r in all_results["real_detector"] if r["calibration_mode"] == "mondrian"]
    print("READ-OUT (real detector, Mondrian)")
    base = real_mondrian[0]
    for r in real_mondrian:
        ratio = r["block_discriminability"] / base["block_discriminability"] if base["block_discriminability"] else float("nan")
        print(f"  k={r['block_size']:<2} discriminability {r['block_discriminability']:.3f}  "
              f"({ratio:.2f}x vs k=1, sqrt(k) would predict {np.sqrt(r['block_size']):.2f}x)")

    session_stats = {round(r["session_effect_to_noise"], 6) for r in real_mondrian}
    print(f"\n  session-level effect/noise constant across k: {len(session_stats) == 1} "
          f"(values: {sorted(session_stats)})")
    print("  (it must be -- averaging within a session cannot change the session mean;"
          " this is the blocking implementation's correctness check)")

    null_mondrian = [r for r in all_results["null_detector"] if r["calibration_mode"] == "mondrian"]
    print(f"\n  null detector discriminability by k: "
          f"{[round(r['block_discriminability'], 3) for r in null_mondrian]}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(
            {
                "note": ("Block aggregation sweep. Per-block discriminability is the metric "
                         "blocking can move; the session-level statistic is invariant to k "
                         "by construction and is reported as a correctness check."),
                "checkpoint": args.checkpoint, "alpha": alpha, "gap": GAP,
                "block_sizes": BLOCK_SIZES, "n_bootstrap": N_BOOTSTRAP, "rng_seed": RNG_SEED,
                "results": all_results,
            },
            f, indent=2,
        )
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
