"""
Runs the conformal_monitor operating-curve evaluation against the REAL
Snowy Scenes archive using a TRAINED CRAFX_Net checkpoint (README's stated
critical path -- "the head-to-head operating curve the paper's central
claim depends on needs a trained detector before the numbers mean
anything"). Supersedes run_operating_curve_experiment.py's untrained/mock
smoke run: same pipeline, but on real held-out frames with real trained
weights, so these numbers are the paper's first real evidence rather than
a mechanics check.

Snowy Scenes has no clear-weather split (see real_snow_stream.py), so the
"clear" replicates used for the false-alarm-rate control are built from
`accumulated`-category frames on both sides of the splice (stationary,
no real transition) rather than a true clear/no-snow condition -- the same
substitution the README's first live run used.

Usage:
    python run_real_operating_curve.py \
        --zip-path ../../../data/ROADVIEW5k.zip \
        --checkpoint ../../../checkpoints/snowy_scenes/checkpoint_final.pth \
        --split val --device cuda
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from torch.utils.data import Subset

from craf_x.config import CRAFXConfig
from craf_x.datasets.snowy_scenes_dataset import SNOWY_SCENES_NUM_CLASSES, CRAFXSnowyScenesDataset
from craf_x.models.crafx_net import CRAFX_Net

from conformal_monitor.betting import AGRAPABettor, CCPInformedBettor
from conformal_monitor.evaluate import calibrate_on_clear_weather, operating_curve
from conformal_monitor.real_snow_stream import RealSnowOnsetStream, category_indices

CONFORMAL_ALPHA = 0.2
SCENE_LENGTH = 20
ONSET_FRAME = 8
DELTAS = [0.3, 0.1, 0.05]
N_ONSET_REPLICATES = 5
N_CLEAR_REPLICATES = 5
KAPPA = 2.0
N_CALIBRATION_FRAMES = 40


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zip-path", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="val", help="Snowy Scenes split to evaluate on (not the training split).")
    parser.add_argument("--bev-size", type=int, default=128, help="Must match the checkpoint's training bev size.")
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu",
        help="Defaults to 'cuda' if available, else 'cpu'.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    config = CRAFXConfig(bev_h=args.bev_size, bev_w=args.bev_size, num_classes=SNOWY_SCENES_NUM_CLASSES)
    dataset = CRAFXSnowyScenesDataset(zip_path=args.zip_path, split=args.split, config=config)

    device = torch.device(args.device)
    model = CRAFX_Net(config).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print(f"Loaded checkpoint {args.checkpoint} (epoch {checkpoint.get('epoch')})")

    acc_indices = category_indices(dataset, "accumulated")
    fall_indices = category_indices(dataset, "falling")
    print(f"{args.split} split: {len(acc_indices)} accumulated frames, {len(fall_indices)} falling frames")

    min_needed = N_CALIBRATION_FRAMES + SCENE_LENGTH
    if len(acc_indices) < min_needed:
        raise ValueError(
            f"Need at least {min_needed} accumulated frames ({N_CALIBRATION_FRAMES} calibration + "
            f"{SCENE_LENGTH} nominal-stream) in split '{args.split}', found {len(acc_indices)}."
        )
    if len(fall_indices) < SCENE_LENGTH:
        raise ValueError(f"Need at least {SCENE_LENGTH} falling frames in split '{args.split}', found {len(fall_indices)}.")

    # Calibration frames are disjoint from the nominal-stream frames so the
    # monitored stream's "clean" portion isn't the same data the quantile
    # was fit on.
    calibration_set = Subset(dataset, acc_indices[:N_CALIBRATION_FRAMES])
    nominal_set = Subset(dataset, acc_indices[N_CALIBRATION_FRAMES:])
    degraded_set = Subset(dataset, fall_indices)

    print(f"Calibrating on {len(calibration_set)} held-out accumulated frames...")
    q_hat = calibrate_on_clear_weather(model, calibration_set, CONFORMAL_ALPHA, batch_size=4, num_workers=4)
    print(f"q_hat = {q_hat:.4f}")

    def make_onset_stream():
        return RealSnowOnsetStream(nominal_set, degraded_set, onset_frame=ONSET_FRAME, scene_length=SCENE_LENGTH)

    def make_clear_stream():
        # No real clear-weather split exists (see real_snow_stream.py) -- use
        # accumulated frames on both sides of the splice so the stream is
        # stationary throughout. This is the false-alarm-rate control (a
        # "no true onset" replicate), not a real clear-weather condition.
        return RealSnowOnsetStream(nominal_set, nominal_set, onset_frame=ONSET_FRAME, scene_length=SCENE_LENGTH)

    bettor_factories = {
        "covariate_blind_agrapa": lambda: AGRAPABettor(CONFORMAL_ALPHA),
        "ccp_informed": lambda: CCPInformedBettor(AGRAPABettor(CONFORMAL_ALPHA), kappa=KAPPA),
    }

    results = {}
    for name, factory in bettor_factories.items():
        print(f"\nRunning operating curve for bettor: {name}")
        start = time.time()
        curve = operating_curve(
            model, q_hat, CONFORMAL_ALPHA, DELTAS,
            onset_stream_factory=make_onset_stream,
            clear_stream_factory=make_clear_stream,
            bettor_factory=factory,
            n_onset_replicates=N_ONSET_REPLICATES,
            n_clear_replicates=N_CLEAR_REPLICATES,
        )
        elapsed = time.time() - start
        results[name] = curve
        print(f"  ({elapsed:.1f}s)")
        for point in curve:
            print(
                f"  delta={point['delta']:.2f}  "
                f"false_alarm_rate={point['false_alarm_rate']:.2f}  "
                f"mean_detection_delay={point['mean_detection_delay']}  "
                f"n_censored={point['n_censored']}"
            )

    out_path = os.path.join(os.path.dirname(__file__), "..", "real_operating_curve_run.json")
    with open(out_path, "w") as f:
        json.dump(
            {
                "note": (
                    "Real Snowy Scenes data + trained CRAFX_Net checkpoint -- "
                    "first real evidence, not a mechanics smoke run. 'Clear' "
                    "replicates use accumulated frames on both sides of the "
                    "splice (no real clear-weather split exists)."
                ),
                "checkpoint": args.checkpoint,
                "checkpoint_epoch": checkpoint.get("epoch"),
                "split": args.split,
                "config": {
                    "conformal_alpha": CONFORMAL_ALPHA,
                    "scene_length": SCENE_LENGTH,
                    "onset_frame": ONSET_FRAME,
                    "deltas": DELTAS,
                    "n_onset_replicates": N_ONSET_REPLICATES,
                    "n_clear_replicates": N_CLEAR_REPLICATES,
                    "kappa": KAPPA,
                    "n_calibration_frames": N_CALIBRATION_FRAMES,
                    "q_hat": q_hat,
                    "bev_size": args.bev_size,
                },
                "results": results,
            },
            f,
            indent=2,
        )
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
