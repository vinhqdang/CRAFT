"""
Real operating-curve evaluation of the synthetic-snow-corruption checkpoint
(checkpoints/snowy_scenes_snow_corrupted/checkpoint_final.pth), using the
exact same protocol/defaults as Table 1 (kappa=2.0, 40 calibration frames,
scene length 20, alpha=0.2, deltas=[0.3,0.1,0.05], 5+5 replicates) for a
direct, apples-to-apples comparison against the existing real result.

A local copy, not a modification of
papers/conformal-snow-icra2027/scripts/run_real_operating_curve.py, per
the coordination constraint against touching that paper's files.

Usage:
    python evaluate_snow_corrupted.py \
        --zip-path ../../../data/ROADVIEW5k.zip \
        --checkpoint ../../../checkpoints/snowy_scenes_snow_corrupted/checkpoint_final.pth \
        --split val --device cuda
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "conformal-snow-icra2027")))

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
    parser.add_argument("--split", default="val")
    parser.add_argument("--bev-size", type=int, default=128)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out", default="../manuscript/snow_corrupted_operating_curve.json")
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
    print(f"{args.split} split: {len(acc_indices)} accumulated, {len(fall_indices)} falling frames")

    calibration_set = Subset(dataset, acc_indices[:N_CALIBRATION_FRAMES])
    nominal_set = Subset(dataset, acc_indices[N_CALIBRATION_FRAMES:])
    degraded_set = Subset(dataset, fall_indices)

    print(f"Calibrating on {len(calibration_set)} held-out accumulated frames...")
    q_hat = calibrate_on_clear_weather(model, calibration_set, CONFORMAL_ALPHA, batch_size=4, num_workers=4)
    print(f"q_hat = {q_hat:.4f}")

    # Also directly sample the CCP disagreement across a real onset stream,
    # same diagnostic used for the original checkpoint, to see whether the
    # ordering is now correct (higher disagreement in `falling` than
    # `accumulated`) rather than only looking at the downstream operating
    # curve.
    stream = RealSnowOnsetStream(nominal_set, degraded_set, onset_frame=ONSET_FRAME, scene_length=SCENE_LENGTH)
    disagreements = []
    with torch.no_grad():
        for t in range(len(stream)):
            sample = stream[t]
            image = sample["image"].unsqueeze(0).to(device)
            pointcloud = sample["pointcloud"].unsqueeze(0).to(device)
            out = model(image, pointcloud)
            disagreements.append((t, "nominal" if t < ONSET_FRAME else "degraded", float((1.0 - out["S"]).mean().item())))
    print("Per-frame CCP disagreement across the real onset stream:")
    for t, regime, d in disagreements:
        print(f"  t={t:2d} {regime:8s} disagreement={d:.6f}")
    nominal_mean = sum(d for _, r, d in disagreements if r == "nominal") / sum(1 for _, r, _ in disagreements if r == "nominal")
    degraded_mean = sum(d for _, r, d in disagreements if r == "degraded") / sum(1 for _, r, _ in disagreements if r == "degraded")
    print(f"Mean disagreement: nominal={nominal_mean:.4f}, degraded={degraded_mean:.4f} "
          f"({'CORRECT direction' if degraded_mean > nominal_mean else 'WRONG direction'})")

    def make_onset_stream():
        return RealSnowOnsetStream(nominal_set, degraded_set, onset_frame=ONSET_FRAME, scene_length=SCENE_LENGTH)

    def make_clear_stream():
        return RealSnowOnsetStream(nominal_set, nominal_set, onset_frame=ONSET_FRAME, scene_length=SCENE_LENGTH)

    bettor_factories = {
        "covariate_blind_agrapa": lambda: AGRAPABettor(CONFORMAL_ALPHA),
        "ccp_informed": lambda: CCPInformedBettor(AGRAPABettor(CONFORMAL_ALPHA), kappa=KAPPA),
    }

    results = {}
    for name, factory in bettor_factories.items():
        print(f"\nRunning operating curve for bettor: {name}")
        curve = operating_curve(
            model, q_hat, CONFORMAL_ALPHA, DELTAS,
            onset_stream_factory=make_onset_stream,
            clear_stream_factory=make_clear_stream,
            bettor_factory=factory,
            n_onset_replicates=N_ONSET_REPLICATES,
            n_clear_replicates=N_CLEAR_REPLICATES,
        )
        results[name] = curve
        for point in curve:
            print(
                f"  delta={point['delta']:.2f}  "
                f"false_alarm_rate={point['false_alarm_rate']:.2f}  "
                f"mean_detection_delay={point['mean_detection_delay']}  "
                f"n_censored={point['n_censored']}"
            )

    out_path = args.out
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(
            {
                "note": "Real Snowy Scenes evaluation of the synthetic-snow-corruption CCP checkpoint.",
                "checkpoint": args.checkpoint,
                "checkpoint_epoch": checkpoint.get("epoch"),
                "split": args.split,
                "disagreement_check": {
                    "nominal_mean": nominal_mean,
                    "degraded_mean": degraded_mean,
                    "correct_direction": degraded_mean > nominal_mean,
                    "per_frame": disagreements,
                },
                "config": {
                    "conformal_alpha": CONFORMAL_ALPHA, "scene_length": SCENE_LENGTH, "onset_frame": ONSET_FRAME,
                    "deltas": DELTAS, "n_onset_replicates": N_ONSET_REPLICATES,
                    "n_clear_replicates": N_CLEAR_REPLICATES, "kappa": KAPPA,
                    "n_calibration_frames": N_CALIBRATION_FRAMES, "q_hat": q_hat, "bev_size": args.bev_size,
                },
                "results": results,
            },
            f, indent=2,
        )
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
