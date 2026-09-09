"""
The decisive re-measurement: is there a weather effect on drives the
detector never trained on?

Everything previously measured on CADC is void, because the detector and
the monitor were built over the same unsplit dataset -- every calibration,
nominal and degraded frame had been a training frame, so the conformal
scores were in-sample residuals and the calibrated quantile bounded
nothing out-of-sample.

This runs the effect-existence test on the drive-disjoint evaluation split
with two corrections applied at once:

1. **Held-out drives.** The detector trained on `split="train"`; every
   frame here comes from `split="eval"`. Session-conditional calibration
   still splits each evaluation drive internally into a calibration prefix,
   a temporal gap, and a monitored remainder -- that inner split is nested
   inside the evaluation drives and never reaches training data.
2. **Two-level bootstrap.** Nominal sessions and degraded drives are both
   resampled. The nominal-only interval is reported alongside so the
   inflation caused by the correction is visible rather than asserted.

The null detector runs on the same footing throughout: if the effect is
content rather than detector error, it appears in both arms.

Report this before running any curve or operating curve.

Usage:
    python run_split_effect_test.py --data-root ../../../data/cadcd \
        --checkpoint ../../../checkpoints/cadc_split/checkpoint_final.pth \
        --label split_ep5
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

from signal_monitor.effect_estimation import two_level_effect
from signal_monitor.null_detector import NullBoxHeadModel
from experiment_common import DATASET_DEFAULTS, build_real_setting, _drives_for_category
from run_mondrian_gap_sweep import cache_scores

GAPS = [0, 10, 20]
N_CALIBRATION_PER_SESSION = 20
N_BOOTSTRAP = 3000
RNG_SEED = 20260908


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["cadc"], default="cadc")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--bev-size", type=int, default=128)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-frames-per-drive", type=int, default=60)
    parser.add_argument("--out", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    args.zip_path = None
    setting = build_real_setting(args)
    alpha = DATASET_DEFAULTS[args.dataset]["alpha"]

    nominal_drives = _drives_for_category(setting.dataset, "bare")
    degraded_drives = _drives_for_category(setting.dataset, "covered")
    print(f"\nEVAL SPLIT ONLY -- detector never saw these drives")
    print(f"  {len(nominal_drives)} nominal (bare) drives  ->  bootstrap sessions")
    print(f"  {len(degraded_drives)} degraded (covered) drives -> resampled as the second level")
    print(f"  alpha={alpha}, {N_CALIBRATION_PER_SESSION} calibration frames/session, "
          f"{N_BOOTSTRAP} bootstrap draws\n")

    all_results = {}
    for arm_label, model in (("real_detector", setting.model),
                             ("null_detector", NullBoxHeadModel(setting.model))):
        model.eval()
        session_scores = {}
        for key, idxs in nominal_drives.items():
            capped = idxs[:args.max_frames_per_drive]
            scores = cache_scores(model, setting.dataset, capped)
            # Calibration prefix and monitored remainder must be disjoint
            # frame sets: passing the same array for both (as this used to
            # do) let `gap=0` score the calibration frames a second time as
            # "monitored", inflating the effect in the direction this test
            # exists to detect. two_level_effect's own `gap` then trims
            # further from the front of the (already-past-calibration)
            # monitored slice.
            calibration_slice = scores[:N_CALIBRATION_PER_SESSION]
            monitored_slice = scores[N_CALIBRATION_PER_SESSION:]
            session_scores[f"{key[0]}_{key[1]}"] = (calibration_slice, monitored_slice)
        degraded_by_drive = {
            f"{key[0]}_{key[1]}": cache_scores(
                model, setting.dataset, idxs[:args.max_frames_per_drive])
            for key, idxs in degraded_drives.items()
        }

        rows = []
        for mode in ("marginal", "mondrian"):
            print(f"=== {arm_label} / {mode} ===")
            print(f"  {'gap':>4} {'nom':>8} {'deg':>8} {'effect':>9} "
                  f"{'two-level 95% CI':>24} {'nominal-only CI':>24} {'infl':>6}")
            for gap in GAPS:
                r = two_level_effect(
                    session_scores, degraded_by_drive, alpha,
                    n_calibration=N_CALIBRATION_PER_SESSION, gap=gap,
                    rng=np.random.default_rng(RNG_SEED),
                    calibration_mode=mode, n_bootstrap=N_BOOTSTRAP,
                )
                if r is None:
                    print(f"  {gap:>4}  (insufficient monitored frames)")
                    continue
                rows.append(r)
                star = "*" if r["excludes_zero"] else " "
                star_n = "*" if r["excludes_zero_nominal_only"] else " "
                print(f"  {gap:>4} {r['nominal_mean']:>8.4f} {r['degraded_mean']:>8.4f} "
                      f"{r['effect']:>+9.4f}{star} "
                      f"[{r['ci_low']:+.4f},{r['ci_high']:+.4f}]".rjust(24) +
                      f"[{r['ci_low_nominal_only']:+.4f},{r['ci_high_nominal_only']:+.4f}]{star_n}".rjust(25) +
                      f" {r['ci_width_inflation']:>5.2f}x")
            print()
        all_results[arm_label] = rows

    real = [r for r in all_results["real_detector"] if r["calibration_mode"] == "mondrian"]
    null = [r for r in all_results["null_detector"] if r["calibration_mode"] == "mondrian"]
    print("READ-OUT (Mondrian arm)")
    print(f"  reference to beat, pre-split and nominal-only bootstrap: "
          f"effect +0.0742, CI [+0.0196,+0.1168]")
    if real:
        any_sig = any(r["excludes_zero"] for r in real)
        print(f"  real detector, held-out drives: "
              f"effect {real[0]['effect']:+.4f} CI [{real[0]['ci_low']:+.4f},{real[0]['ci_high']:+.4f}]"
              f" (gap 0); significant at any gap: {any_sig}")
        print(f"  null detector: significant at any gap: "
              f"{any(r['excludes_zero'] for r in null)}")
        if not any_sig:
            print("\n  No weather effect distinguishable from zero on held-out drives.")
            print("  The pre-split +0.074 does not survive removing train/eval overlap")
            print("  and correcting the bootstrap. Report this; do not run the curve.")
        else:
            print("\n  Effect survives. Proceed to the curve, reporting first.")

    out_path = args.out or f"../manuscript/split_effect_{args.label}.json"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(
            {
                "note": ("Effect-existence on drive-disjoint held-out drives, "
                         "two-level bootstrap over nominal sessions and degraded drives."),
                "checkpoint": args.checkpoint, "label": args.label, "alpha": alpha,
                "n_nominal_drives": len(nominal_drives),
                "n_degraded_drives": len(degraded_drives),
                "n_calibration_per_session": N_CALIBRATION_PER_SESSION,
                "gaps": GAPS, "n_bootstrap": N_BOOTSTRAP, "rng_seed": RNG_SEED,
                "results": all_results,
            },
            f, indent=2,
        )
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
