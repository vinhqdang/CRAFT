"""
Lever 4: does a longer monitoring window convert a valid-but-rarely-firing
monitor into one that detects reliably?

The 20-frame curve showed Mondrian+blocking controlling false alarms
(FA=0.00 at delta<=0.10) while censoring 32/36 and 34/36 replicates, with
every surviving alarm landing on frame 19 -- the last frame of the window.
That is the signature of insufficient accumulation time rather than of a
method that does not work. The window was also never chosen on its merits:
20 frames at CADC's 10 Hz is two seconds, inherited from the original
setup.

This sweeps scene_length with the onset placed proportionally (40% of the
window), so post-onset accumulation time scales with the window instead of
staying pinned at 12 frames. Censoring is the headline metric here, not
delay: detection *rate* is what currently limits the claim.

Every arm gets the identical window at every length -- a win from giving
ourselves more accumulation time than the baseline would be worthless --
and the null detector is run at every length, because if longer windows let
a detector-free monitor start alarming, the interpretation changes
completely.

Results are written after each window length, so a run that hits the
environment's task time limit still leaves the completed lengths on disk.

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
from torch.utils.data import Subset

from conformal_monitor.betting import AGRAPABettor
from conformal_monitor.calibration import calibrate_quantile

from signal_monitor.mondrian import operating_curve_per_stream_quantile
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
    parser.add_argument("--max-frames-per-session", type=int, default=250,
                        help="effectively uncapped; longer windows need more nominal frames")
    parser.add_argument("--max-degraded-frames", type=int, default=300)
    parser.add_argument("--scene-lengths", type=int, nargs="+", default=SCENE_LENGTHS)
    parser.add_argument("--out", default="../manuscript/window_sweep_cadc.json")
    return parser.parse_args()


def session_quantiles(model, dataset, sessions, alpha, max_frames, min_nominal):
    per_session_q, monitoring, pooled_scores = {}, {}, []
    for key, idxs in sessions.items():
        capped = idxs[:max_frames]
        calibration_idx = capped[:N_CALIBRATION_PER_SESSION]
        monitored = capped[N_CALIBRATION_PER_SESSION + GAP:]
        scores = cache_scores(model, dataset, calibration_idx)
        if len(scores) < 10 or len(monitored) < min_nominal:
            continue
        monitoring[key] = monitored
        per_session_q[key] = calibrate_quantile(np.concatenate(scores), alpha)
        pooled_scores.extend(scores)
    pooled = calibrate_quantile(np.concatenate(pooled_scores), alpha) if pooled_scores else float("nan")
    return per_session_q, monitoring, pooled


def build_specs(dataset, monitoring, per_session_q, pooled, degraded_indices, seeds,
                mode, scene_length, onset_frame):
    keys = sorted(monitoring.keys())
    degraded_subset = Subset(dataset, degraded_indices)
    onset_specs, clear_specs = [], []
    for i, seed in enumerate(seeds):
        key = keys[i % len(keys)]
        nominal_subset = Subset(dataset, monitoring[key])
        q = pooled if mode == "marginal" else per_session_q[key]
        onset_specs.append((
            lambda ns=nominal_subset, ds=degraded_subset, s=seed: RandomizedOnsetStream(
                ns, ds, onset_frame, scene_length, s), q))
        clear_specs.append((
            lambda ns=nominal_subset, s=seed: RandomizedOnsetStream(
                ns, ns, onset_frame, scene_length, s), q))
    return onset_specs, clear_specs


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
    all_results = {}

    for scene_length in args.scene_lengths:
        onset_frame = int(round(ONSET_FRACTION * scene_length))
        print(f"\n{'=' * 78}")
        print(f"WINDOW {scene_length} frames ({scene_length / 10:.0f}s at 10Hz), "
              f"onset at frame {onset_frame}, {scene_length - onset_frame} frames post-onset")
        print("=" * 78)

        arms = [
            ("baseline (marginal, k=1)", setting.model, "marginal", 1),
            ("ours (mondrian, k=4)", setting.model, "mondrian", 4),
            ("null (mondrian, k=4)", NullBoxHeadModel(setting.model), "mondrian", 4),
        ]
        per_length = {}
        for name, model, mode, block in arms:
            model.eval()
            per_session_q, monitoring, pooled = session_quantiles(
                model, setting.dataset, sessions, alpha, args.max_frames_per_session, onset_frame
            )
            if not monitoring:
                print(f"  {name}: no session has {onset_frame} nominal frames after "
                      f"calibration+gap -- skipping this window length")
                continue
            onset_specs, clear_specs = build_specs(
                setting.dataset, monitoring, per_session_q, pooled, degraded_indices,
                seeds, mode, scene_length, onset_frame
            )
            curve = operating_curve_per_stream_quantile(
                model, alpha, deltas, onset_specs, clear_specs,
                lambda: AGRAPABettor(alpha), block_size=block,
            )
            per_length[name] = curve

            print(f"\n  {name}   ({len(monitoring)} sessions usable)")
            print(f"    {'delta':>6} {'FA':>6} {'FA CI':>14} {'detected':>10} "
                  f"{'delay':>7} {'delay CI':>16} {'FA ok':>6}")
            for p in curve:
                n_rep = p["n_onset_replicates"]
                detected = n_rep - p["n_censored"]
                dl = p["mean_detection_delay"]
                delay = f"{dl:.1f}" if dl is not None else "--"
                dci = p["detection_delay_ci"]
                dci_s = f"[{dci[0]:.1f},{dci[1]:.1f}]" if dci[0] is not None else "--"
                fci = p["false_alarm_ci"]
                print(f"    {p['delta']:>6.2f} {p['false_alarm_rate']:>6.2f} "
                      f"[{fci[0]:.2f},{fci[1]:.2f}]".ljust(35) +
                      f"{detected:>3}/{n_rep:<3}{'':>4}{delay:>7} {dci_s:>16} "
                      f"{'yes' if p['controls_false_alarms'] else 'NO':>6}")

        all_results[str(scene_length)] = per_length
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(
                {
                    "note": ("Window-length sweep. Onset placed at 40% of the window so "
                             "post-onset accumulation scales with window length. All arms "
                             "share window, seeds and sessions at every length."),
                    "checkpoint": args.checkpoint, "alpha": alpha, "deltas": deltas,
                    "n_replicates": N_REPLICATES, "onset_fraction": ONSET_FRACTION,
                    "gap": GAP, "base_seed": BASE_SEED,
                    "n_degraded_frames": len(degraded_indices),
                    "results": all_results,
                },
                f, indent=2,
            )
        print(f"\n  [written through window {scene_length}]")

    print(f"\n\n{'=' * 78}\nDETECTION RATE BY WINDOW (ours, mondrian k=4) -- the headline metric")
    print("=" * 78)
    print(f"  {'window':>7} {'delta=0.30':>22} {'delta=0.10':>22} {'delta=0.05':>22}")
    for length, per_length in all_results.items():
        curve = per_length.get("ours (mondrian, k=4)")
        if not curve:
            continue
        cells = []
        for p in curve:
            n_rep = p["n_onset_replicates"]
            detected = n_rep - p["n_censored"]
            ok = "" if p["controls_false_alarms"] else " FA!"
            cells.append(f"{detected}/{n_rep} ({detected / n_rep:.0%}){ok}")
        print(f"  {length:>7} " + " ".join(f"{c:>22}" for c in cells))

    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
