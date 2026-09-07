"""
Null-detector control: the full operating curve with no functioning
detector, reported side by side with the real detector's.

See `signal_monitor/null_detector.py` for why this is needed rather than
optional. Short version: on CADC the ground-truth scene-content artifact
points the same direction as the real degradation signal, so a good
detection number there could be riding on the splice rather than on
perception degradation. Snowy Scenes acts as the positive control -- its
content artifact opposes the real signal, so the null detector should do
badly there.

Both curves use the identical protocol, replicate streams, calibration set
and bettor. The only difference is whether the box head's output is real or
identically zero, and the null detector calibrates its own quantile on its
own scores, exactly as a deployment with a broken box head would.

Usage:
    python run_zeros_baseline.py --dataset cadc \
        --data-root ../../../data/cadcd \
        --checkpoint ../../../checkpoints/cadc_focal/checkpoint_final.pth
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "conformal-snow-icra2027")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch

from conformal_monitor.betting import AGRAPABettor
from conformal_monitor.evaluate import calibrate_on_clear_weather, operating_curve
from conformal_monitor.real_snow_stream import RealSnowOnsetStream

from signal_monitor.null_detector import NullBoxHeadModel
from experiment_common import DATASET_DEFAULTS, build_real_setting


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["snowy", "cadc"], required=True)
    parser.add_argument("--zip-path")
    parser.add_argument("--data-root")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--bev-size", type=int, default=128)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out", default=None)
    return parser.parse_args()


# A monitor is only allowed into a delay comparison if it actually controls
# its false-alarm budget. Comparing delays across monitors with different
# false-alarm rates is meaningless: an uncontrolled monitor "detects" early
# by alarming at everything, including before the onset it is supposed to be
# detecting.
def _controls_false_alarms(point, delta):
    return point["false_alarm_rate"] <= delta


def _attribution_verdict(real, null, delta):
    """
    Compare the real detector against the null detector at one delta.

    Delay is only a meaningful comparison between two monitors that both
    control false alarms; otherwise the faster-looking one may simply be
    firing indiscriminately. An earlier version of this function compared
    `mean_detection_delay` directly and reported a monitor with
    false_alarm_rate=1.00 and a NEGATIVE delay (alarming before onset) as
    "real faster - attributable", which is exactly backwards.
    """
    real_ok = _controls_false_alarms(real, delta)
    null_ok = _controls_false_alarms(null, delta)
    rd, nd = real["mean_detection_delay"], null["mean_detection_delay"]

    if not real_ok:
        return (f"REAL MONITOR BROKEN (FA={real['false_alarm_rate']:.2f} > delta={delta:.2f}"
                f"{', delay ' + format(rd, '.1f') + ' is pre-onset' if rd is not None and rd < 0 else ''})"
                " - no attribution possible until false alarms are controlled")
    if not null_ok:
        return "real controls false alarms, null does not - cleanly attributable to the detector"
    if rd is None and nd is None:
        return "both control false alarms, neither alarms - no detection either way"
    if nd is None:
        return "ONLY the real detector alarms, both FA-controlled - cleanly attributable"
    if rd is None:
        return "ONLY the null detector alarms - result is pure content artifact"
    if rd < nd:
        return f"real faster by {nd - rd:.1f} frames, both FA-controlled - attributable"
    if rd > nd:
        return f"NULL faster by {rd - nd:.1f} frames - result is content-driven"
    return "identical - indistinguishable from content artifact"


def _print_curve(name, curve):
    print(f"\n  {name}")
    for point in curve:
        delay = point["mean_detection_delay"]
        delay_str = f"{delay:.1f}" if delay is not None else "-- (censored)"
        print(f"    delta={point['delta']:.2f}  FA={point['false_alarm_rate']:.2f}  "
              f"delay={delay_str}  censored={point['n_censored']}/5")


def main():
    args = parse_args()
    setting = build_real_setting(args)
    d = DATASET_DEFAULTS[args.dataset]
    alpha, deltas = d["alpha"], d["deltas"]

    def make_onset_stream():
        return RealSnowOnsetStream(setting.nominal_set, setting.degraded_set,
                                   onset_frame=d["onset_frame"], scene_length=d["scene_length"])

    def make_clear_stream():
        return RealSnowOnsetStream(setting.nominal_set, setting.nominal_set,
                                   onset_frame=d["onset_frame"], scene_length=d["scene_length"])

    def bettor_factory():
        return AGRAPABettor(alpha)

    results, quantiles = {}, {}
    for name, model in (("real_detector", setting.model),
                        ("null_detector", NullBoxHeadModel(setting.model))):
        model.eval()
        q_hat = calibrate_on_clear_weather(model, setting.calibration_set, alpha,
                                           batch_size=4, num_workers=4)
        quantiles[name] = q_hat
        print(f"\n=== {name} (q_hat={q_hat:.6f}) ===")
        curve = operating_curve(
            model, q_hat, alpha, deltas, make_onset_stream, make_clear_stream, bettor_factory,
            n_onset_replicates=d["n_onset_replicates"], n_clear_replicates=d["n_clear_replicates"],
        )
        results[name] = curve
        _print_curve(name, curve)

    print("\n\nATTRIBUTION")
    for delta in deltas:
        real = next(p for p in results["real_detector"] if p["delta"] == delta)
        null = next(p for p in results["null_detector"] if p["delta"] == delta)
        print(f"  delta={delta:.2f}: {_attribution_verdict(real, null, delta)}")

    out_path = args.out or f"../manuscript/zeros_baseline_{args.dataset}.json"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(
            {
                "note": ("Null-detector control: operating curve with the box head's output "
                         "zeroed, versus the real detector, identical protocol."),
                "dataset": args.dataset,
                "checkpoint": args.checkpoint,
                "q_hat": quantiles,
                "config": {k: v for k, v in d.items()},
                "results": results,
            },
            f, indent=2, default=str,
        )
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
