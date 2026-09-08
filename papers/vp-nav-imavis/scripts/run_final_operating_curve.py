"""
The operating curve the paper's claim rests on.

Specific question: marginal calibration produced a false-alarm rate of 1.00
and an unusable monitor on CADC. Does Mondrian session-conditional
calibration plus block aggregation give controlled false alarms with real
detection, on the same data?

Arms, all on the fixed randomized harness with a shared seed list so every
replicate is the identical frame draw across arms:

    baseline      marginal pooled q_hat        k=1
    ours          Mondrian session-conditional k=4
    null_marginal null detector, marginal      k=4
    null_mondrian null detector, Mondrian      k=4

The null arms are the control that makes the result defensible: a monitor
with no functioning box head sees only ground-truth scene content, so if it
matches the real arm the result is content-driven rather than perception
driven. This is not a formality here -- the null detector previously
reproduced both datasets' published headline numbers exactly.

Every delay comparison is gated on false-alarm control: a monitor whose FA
exceeds delta has no valid delay to compare, because it "detects" early by
alarming indiscriminately, including before the onset exists.

Usage:
    python run_final_operating_curve.py --data-root ../../../data/cadcd \
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
from torch.utils.data import Subset

from conformal_monitor.betting import AGRAPABettor
from conformal_monitor.calibration import calibrate_quantile

from signal_monitor.mondrian import operating_curve_per_stream_quantile
from signal_monitor.null_detector import NullBoxHeadModel
from signal_monitor.randomized_stream import RandomizedOnsetStream, stream_seeds
from experiment_common import DATASET_DEFAULTS, build_real_setting, _drives_for_category
from run_mondrian_gap_sweep import cache_scores, N_CALIBRATION_PER_SESSION

GAP = 10
N_REPLICATES = 36          # 2 per session across 18 sessions
SCENE_LENGTH = 20
ONSET_FRAME = 8
BASE_SEED = 20260908


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
    parser.add_argument("--out", default="../manuscript/final_operating_curve_cadc.json")
    return parser.parse_args()


def session_quantiles(model, dataset, sessions, alpha, max_frames):
    """Per-session quantile and pooled quantile, from each session's own
    leading frames, with the monitored remainder returned alongside."""
    per_session_q, monitoring, pooled_scores = {}, {}, []
    for key, idxs in sessions.items():
        capped = idxs[:max_frames]
        calibration_idx = capped[:N_CALIBRATION_PER_SESSION]
        monitoring[key] = capped[N_CALIBRATION_PER_SESSION + GAP:]
        scores = cache_scores(model, dataset, calibration_idx)
        if len(scores) < 10 or len(monitoring[key]) < ONSET_FRAME:
            monitoring.pop(key, None)
            continue
        per_session_q[key] = calibrate_quantile(np.concatenate(scores), alpha)
        pooled_scores.extend(scores)
    pooled = calibrate_quantile(np.concatenate(pooled_scores), alpha)
    return per_session_q, monitoring, pooled


def build_specs(dataset, monitoring, per_session_q, pooled, degraded_indices, seeds, mode):
    """One (stream_factory, quantile) pair per replicate, for onset and
    clear streams. Replicates cycle over sessions so every session is
    represented, and both modes reuse the identical seed list and session
    assignment -- so arms differ only in the quantile."""
    keys = sorted(monitoring.keys())
    degraded_subset = Subset(dataset, degraded_indices)
    onset_specs, clear_specs = [], []

    for i, seed in enumerate(seeds):
        key = keys[i % len(keys)]
        nominal_subset = Subset(dataset, monitoring[key])
        q = pooled if mode == "marginal" else per_session_q[key]

        onset_specs.append((
            lambda ns=nominal_subset, ds=degraded_subset, s=seed: RandomizedOnsetStream(
                ns, ds, ONSET_FRAME, SCENE_LENGTH, s),
            q,
        ))
        clear_specs.append((
            lambda ns=nominal_subset, s=seed: RandomizedOnsetStream(
                ns, ns, ONSET_FRAME, SCENE_LENGTH, s),
            q,
        ))
    return onset_specs, clear_specs


def print_curve(name, curve):
    print(f"\n  {name}")
    print(f"    {'delta':>6} {'FA':>6} {'FA 95% CI':>18} {'delay':>7} {'delay 95% CI':>20} "
          f"{'cens':>7} {'FA ok':>6}")
    for p in curve:
        d = p["mean_detection_delay"]
        delay = f"{d:.1f}" if d is not None else "--"
        dci = p["detection_delay_ci"]
        dci_s = f"[{dci[0]:.1f},{dci[1]:.1f}]" if dci[0] is not None else "--"
        fci = p["false_alarm_ci"]
        print(f"    {p['delta']:>6.2f} {p['false_alarm_rate']:>6.2f} "
              f"[{fci[0]:.2f},{fci[1]:.2f}]".ljust(0) +
              f"{'':>4}{delay:>7} {dci_s:>20} "
              f"{p['n_censored']:>3}/{p['n_onset_replicates']:<3} "
              f"{'yes' if p['controls_false_alarms'] else 'NO':>6}")


def main():
    args = parse_args()
    args.zip_path = None
    setting = build_real_setting(args)
    d = DATASET_DEFAULTS[args.dataset]
    alpha, deltas = d["alpha"], d["deltas"]

    sessions = _drives_for_category(setting.dataset, "bare")
    degraded_sessions = _drives_for_category(setting.dataset, "covered")
    degraded_indices = [i for k in sorted(degraded_sessions) for i in degraded_sessions[k]]
    pick = np.random.default_rng(BASE_SEED)
    if len(degraded_indices) > args.max_degraded_frames:
        degraded_indices = sorted(pick.choice(degraded_indices, args.max_degraded_frames, replace=False))

    seeds = stream_seeds(N_REPLICATES, base_seed=BASE_SEED)
    print(f"\nCADC final operating curve: {N_REPLICATES} replicates, scene_length={SCENE_LENGTH}, "
          f"onset={ONSET_FRAME}, gap={GAP}, alpha={alpha}")
    print(f"{len(sessions)} sessions, {len(degraded_indices)} degraded frames, "
          f"paired seeds shared across all arms\n")

    arms = [
        ("baseline (marginal, k=1)", setting.model, "marginal", 1),
        ("ours (mondrian, k=4)", setting.model, "mondrian", 4),
        ("null (marginal, k=4)", NullBoxHeadModel(setting.model), "marginal", 4),
        ("null (mondrian, k=4)", NullBoxHeadModel(setting.model), "mondrian", 4),
    ]

    results = {}
    for name, model, mode, block in arms:
        model.eval()
        per_session_q, monitoring, pooled = session_quantiles(
            model, setting.dataset, sessions, alpha, args.max_frames_per_session
        )
        onset_specs, clear_specs = build_specs(
            setting.dataset, monitoring, per_session_q, pooled, degraded_indices, seeds, mode
        )
        curve = operating_curve_per_stream_quantile(
            model, alpha, deltas, onset_specs, clear_specs,
            lambda: AGRAPABettor(alpha), block_size=block,
        )
        results[name] = curve
        print_curve(name, curve)

    print("\n\nVERDICT (FA-gated: a monitor with FA > delta has no valid delay)")
    baseline = results["baseline (marginal, k=1)"]
    ours = results["ours (mondrian, k=4)"]
    null_m = results["null (mondrian, k=4)"]
    for delta in deltas:
        b = next(p for p in baseline if p["delta"] == delta)
        o = next(p for p in ours if p["delta"] == delta)
        n = next(p for p in null_m if p["delta"] == delta)
        line = f"  delta={delta:.2f}: baseline FA={b['false_alarm_rate']:.2f}" \
               f"{'(ok)' if b['controls_false_alarms'] else '(BROKEN)'}, " \
               f"ours FA={o['false_alarm_rate']:.2f}" \
               f"{'(ok)' if o['controls_false_alarms'] else '(BROKEN)'}"
        if o["controls_false_alarms"] and o["mean_detection_delay"] is not None:
            line += f", ours delay={o['mean_detection_delay']:.1f}"
            if n["controls_false_alarms"] and n["mean_detection_delay"] is not None:
                line += f" vs null {n['mean_detection_delay']:.1f}"
                line += " -> ATTRIBUTABLE" if o["mean_detection_delay"] < n["mean_detection_delay"] \
                    else " -> NOT better than null"
            else:
                line += ", null does not alarm -> ATTRIBUTABLE"
        elif o["controls_false_alarms"]:
            line += ", ours controls FA but never alarms"
        print(line)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(
            {
                "note": "CADC final operating curve: marginal k=1 vs Mondrian k=4, with null controls.",
                "checkpoint": args.checkpoint, "alpha": alpha, "deltas": deltas,
                "n_replicates": N_REPLICATES, "scene_length": SCENE_LENGTH,
                "onset_frame": ONSET_FRAME, "gap": GAP, "base_seed": BASE_SEED,
                "n_degraded_frames": len(degraded_indices),
                "results": results,
            },
            f, indent=2,
        )
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
