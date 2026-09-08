"""
Lever 4: does a longer monitoring window convert a valid-but-rarely-firing
monitor into one that detects reliably?

The 20-frame curve showed Mondrian+blocking controlling false alarms while
censoring 30/36 to 36/36 replicates, with surviving alarms landing on the
window's final frame. That is the signature of insufficient accumulation
time rather than of a method that does not work, and the window was never
chosen on its merits: 20 frames at CADC's 10 Hz is two seconds, inherited
from the original setup.

This sweeps scene_length with the onset placed proportionally (40% of the
window), so post-onset accumulation scales with the window rather than
staying pinned. Censoring is the headline metric: detection *rate* is what
currently limits the claim, not delay.

Every arm gets the identical window, seeds and sessions at every length --
a win from giving ourselves more accumulation time than the baseline would
be worthless -- and the null detector runs at every length, because a
detector-free monitor that starts alarming on longer windows would change
the interpretation entirely.

Note the trade-off this sweep is measuring: a longer window mechanically
raises the false-alarm rate for *every* arm, since clear streams get more
frames in which to cross the threshold and Ville's inequality bounds the
alarm probability over all stopping times. Censoring should fall and FA
should rise; the question is whether ours stays under delta while censoring
drops.

## Why this runs from cached scores

Nonconformity scores depend on neither the quantile, the block size nor the
window length. CADC reads an image and a LiDAR sweep per frame at roughly
0.4s, so driving the sweep from the model directly needs ~39,000 frame
reads (over four hours, and the first attempt was killed by the task time
limit mid-way). Scoring every frame once per checkpoint reduces that to
~2,800. `RandomizedOnsetStream` indexes any pool, so lists of cached score
arrays substitute for frame datasets directly -- same seeded
without-replacement sampling, same paired design, no second code path that
could drift from the tested one. An equivalence test asserts the cached
path is arithmetically identical to the model path at every block size.

Usage:
    python run_window_sweep.py --data-root ../../../data/cadcd \
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

from conformal_monitor.betting import AGRAPABettor
from conformal_monitor.calibration import calibrate_quantile

from signal_monitor.mondrian import operating_curve_from_score_streams
from signal_monitor.null_detector import NullBoxHeadModel
from signal_monitor.randomized_stream import RandomizedOnsetStream, stream_seeds
from experiment_common import DATASET_DEFAULTS, build_real_setting, _drives_for_category
from run_mondrian_gap_sweep import cache_scores, N_CALIBRATION_PER_SESSION

GAP = 10
N_REPLICATES = 36
ONSET_FRACTION = 0.4
BASE_SEED = 20260908
SCENE_LENGTHS = [20, 40, 60, 80]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["cadc"], default="cadc")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--bev-size", type=int, default=128)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-frames-per-session", type=int, default=250)
    parser.add_argument("--max-degraded-frames", type=int, default=300)
    parser.add_argument("--scene-lengths", type=int, nargs="+", default=SCENE_LENGTHS)
    parser.add_argument("--out", default="../manuscript/window_sweep_cadc.json")
    return parser.parse_args()


def build_score_cache(model, dataset, sessions, degraded_indices, alpha, max_frames):
    """Score every frame once for this checkpoint; reused at every window."""
    per_session_q, monitored_scores, pooled_scores = {}, {}, []
    for key, idxs in sessions.items():
        capped = idxs[:max_frames]
        calibration_idx = capped[:N_CALIBRATION_PER_SESSION]
        monitored_idx = capped[N_CALIBRATION_PER_SESSION + GAP:]
        cal_scores = cache_scores(model, dataset, calibration_idx)
        if len(cal_scores) < 10 or not monitored_idx:
            continue
        monitored_scores[key] = cache_scores(model, dataset, monitored_idx)
        per_session_q[key] = calibrate_quantile(np.concatenate(cal_scores), alpha)
        pooled_scores.extend(cal_scores)
    pooled = calibrate_quantile(np.concatenate(pooled_scores), alpha) if pooled_scores else float("nan")
    degraded_scores = cache_scores(model, dataset, degraded_indices)
    return per_session_q, monitored_scores, pooled, degraded_scores


def build_specs(monitored_scores, per_session_q, pooled, degraded_scores, seeds,
                mode, scene_length, onset_frame):
    """One (stream_factory, quantile) pair per replicate, over cached scores."""
    keys = sorted(k for k in monitored_scores if len(monitored_scores[k]) >= onset_frame)
    if not keys:
        return [], [], 0
    onset_specs, clear_specs = [], []
    for i, seed in enumerate(seeds):
        key = keys[i % len(keys)]
        pool = monitored_scores[key]
        q = pooled if mode == "marginal" else per_session_q[key]
        onset_specs.append((
            lambda ns=pool, ds=degraded_scores, s=seed: RandomizedOnsetStream(
                ns, ds, onset_frame, scene_length, s), q))
        clear_specs.append((
            lambda ns=pool, s=seed: RandomizedOnsetStream(
                ns, ns, onset_frame, scene_length, s), q))
    return onset_specs, clear_specs, len(keys)


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

    print("\nScoring all frames once per checkpoint (reused across every window length)...")
    caches = {}
    for arm_label, model in (("real", setting.model), ("null", NullBoxHeadModel(setting.model))):
        model.eval()
        caches[arm_label] = build_score_cache(
            model, setting.dataset, sessions, degraded_indices, alpha, args.max_frames_per_session
        )
        print(f"  {arm_label}: {len(caches[arm_label][1])} sessions cached, "
              f"{len(caches[arm_label][3])} degraded frames")

    arms = [
        ("baseline (marginal, k=1)", "real", "marginal", 1),
        ("ours (mondrian, k=4)", "real", "mondrian", 4),
        ("null (mondrian, k=4)", "null", "mondrian", 4),
    ]

    all_results = {}
    for scene_length in args.scene_lengths:
        onset_frame = int(round(ONSET_FRACTION * scene_length))
        print(f"\n{'=' * 78}")
        print(f"WINDOW {scene_length} frames ({scene_length / 10:.0f}s at 10Hz), "
              f"onset at frame {onset_frame}, {scene_length - onset_frame} frames post-onset")
        print("=" * 78)

        per_length = {}
        for name, cache_key, mode, block in arms:
            per_session_q, monitored_scores, pooled, degraded_scores = caches[cache_key]
            onset_specs, clear_specs, n_usable = build_specs(
                monitored_scores, per_session_q, pooled, degraded_scores,
                seeds, mode, scene_length, onset_frame
            )
            if not onset_specs:
                print(f"\n  {name}: no session has {onset_frame} nominal frames "
                      f"after calibration+gap -- skipping")
                continue
            curve = operating_curve_from_score_streams(
                alpha, deltas, onset_specs, clear_specs,
                lambda: AGRAPABettor(alpha), block_size=block,
            )
            per_length[name] = curve

            print(f"\n  {name}   ({n_usable} sessions usable)")
            print(f"    {'delta':>6} {'FA':>6} {'FA CI':>14} {'detected':>10} "
                  f"{'delay':>8} {'delay CI':>16} {'FA ok':>6}")
            for p in curve:
                n_rep = p["n_onset_replicates"]
                detected = n_rep - p["n_censored"]
                dl = p["mean_detection_delay"]
                delay = f"{dl:.1f}" if dl is not None else "--"
                dci = p["detection_delay_ci"]
                dci_s = f"[{dci[0]:.1f},{dci[1]:.1f}]" if dci[0] is not None else "--"
                fci = p["false_alarm_ci"]
                fa_ci_s = f"[{fci[0]:.2f},{fci[1]:.2f}]"
                print(f"    {p['delta']:>6.2f} {p['false_alarm_rate']:>6.2f} {fa_ci_s:>14} "
                      f"{detected:>4}/{n_rep:<5} {delay:>8} {dci_s:>16} "
                      f"{'yes' if p['controls_false_alarms'] else 'NO':>6}")

        all_results[str(scene_length)] = per_length
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(
                {
                    "note": ("Window-length sweep on cached scores. Onset at 40% of the "
                             "window so post-onset accumulation scales. All arms share "
                             "window, seeds and sessions at every length."),
                    "checkpoint": args.checkpoint, "alpha": alpha, "deltas": deltas,
                    "n_replicates": N_REPLICATES, "onset_fraction": ONSET_FRACTION,
                    "gap": GAP, "base_seed": BASE_SEED,
                    "n_degraded_frames": len(degraded_indices),
                    "results": all_results,
                },
                f, indent=2,
            )

    print(f"\n\n{'=' * 78}")
    print("DETECTION RATE BY WINDOW -- the headline metric")
    print("=" * 78)
    for arm_name in ("ours (mondrian, k=4)", "baseline (marginal, k=1)", "null (mondrian, k=4)"):
        print(f"\n  {arm_name}")
        print(f"    {'window':>7} {'d=0.30':>20} {'d=0.10':>20} {'d=0.05':>20}")
        for length in args.scene_lengths:
            curve = all_results.get(str(length), {}).get(arm_name)
            if not curve:
                continue
            cells = []
            for p in curve:
                n_rep = p["n_onset_replicates"]
                detected = n_rep - p["n_censored"]
                flag = "" if p["controls_false_alarms"] else " FA!"
                cells.append(f"{detected}/{n_rep} ({detected / n_rep:.0%}){flag}")
            print(f"    {length:>7} " + " ".join(f"{c:>20}" for c in cells))

    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
