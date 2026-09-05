"""
Real sensitivity analysis for the VP-NAV journal paper (papers/vp-nav-imavis/),
against the actual corrected Snowy Scenes checkpoint -- not projected numbers.
Loads the model and dataset once, then sweeps:
  (a) kappa (CCP-informed bettor's covariate gain) in {0.5, 1.0, 2.0, 4.0, 8.0}
  (b) calibration-set size (n_calibration_frames) in {20, 40, 80}
  (c) scene length in {20, 40}

Reuses conformal_monitor's real evaluation primitives directly (does not
duplicate or modify papers/conformal-snow-icra2027/scripts/run_real_operating_curve.py
-- imports the same library code it does, from a read-only path insert).

Usage:
    python sensitivity_analysis.py \
        --zip-path ../../../data/ROADVIEW5k.zip \
        --checkpoint ../../../checkpoints/snowy_scenes_fixed/checkpoint_final.pth \
        --split val --device cuda \
        --out ../manuscript/sensitivity_results.json
"""
import argparse
import json
import os
import sys
import time

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
ONSET_FRAME = 8
DELTAS = [0.3, 0.1, 0.05]
N_ONSET_REPLICATES = 5
N_CLEAR_REPLICATES = 5


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zip-path", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--bev-size", type=int, default=128)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out", default="sensitivity_results.json")
    return parser.parse_args()


def run_one(model, dataset, acc_indices, fall_indices, n_calib, scene_length, kappa, device):
    calibration_set = Subset(dataset, acc_indices[:n_calib])
    nominal_set = Subset(dataset, acc_indices[n_calib:])
    degraded_set = Subset(dataset, fall_indices)

    q_hat = calibrate_on_clear_weather(model, calibration_set, CONFORMAL_ALPHA, batch_size=4, num_workers=4)

    def make_onset_stream():
        return RealSnowOnsetStream(nominal_set, degraded_set, onset_frame=ONSET_FRAME, scene_length=scene_length)

    def make_clear_stream():
        return RealSnowOnsetStream(nominal_set, nominal_set, onset_frame=ONSET_FRAME, scene_length=scene_length)

    bettor_factories = {
        "covariate_blind_agrapa": lambda: AGRAPABettor(CONFORMAL_ALPHA),
        "ccp_informed": lambda: CCPInformedBettor(AGRAPABettor(CONFORMAL_ALPHA), kappa=kappa),
    }

    results = {}
    for name, factory in bettor_factories.items():
        curve = operating_curve(
            model, q_hat, CONFORMAL_ALPHA, DELTAS,
            onset_stream_factory=make_onset_stream,
            clear_stream_factory=make_clear_stream,
            bettor_factory=factory,
            n_onset_replicates=N_ONSET_REPLICATES,
            n_clear_replicates=N_CLEAR_REPLICATES,
        )
        results[name] = curve
    return {"q_hat": q_hat, "results": results}


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

    all_results = {"kappa_sweep": {}, "calibration_size_sweep": {}, "scene_length_sweep": {}}

    # (a) kappa sweep, fixed n_calib=40, scene_length=20
    for kappa in [0.5, 1.0, 2.0, 4.0, 8.0]:
        print(f"\n=== kappa={kappa} ===")
        t0 = time.time()
        out = run_one(model, dataset, acc_indices, fall_indices, n_calib=40, scene_length=20, kappa=kappa, device=device)
        print(f"  ({time.time()-t0:.1f}s) q_hat={out['q_hat']:.3f}")
        for pt in out["results"]["ccp_informed"]:
            print(f"  ccp_informed delta={pt['delta']:.2f} delay={pt['mean_detection_delay']} fa={pt['false_alarm_rate']:.2f} censored={pt['n_censored']}")
        all_results["kappa_sweep"][str(kappa)] = out

    # (b) calibration-set-size sweep, fixed kappa=2.0, scene_length=20
    for n_calib in [20, 40, 80]:
        print(f"\n=== n_calibration_frames={n_calib} ===")
        t0 = time.time()
        out = run_one(model, dataset, acc_indices, fall_indices, n_calib=n_calib, scene_length=20, kappa=2.0, device=device)
        print(f"  ({time.time()-t0:.1f}s) q_hat={out['q_hat']:.3f}")
        for pt in out["results"]["covariate_blind_agrapa"]:
            print(f"  blind delta={pt['delta']:.2f} delay={pt['mean_detection_delay']} fa={pt['false_alarm_rate']:.2f} censored={pt['n_censored']}")
        all_results["calibration_size_sweep"][str(n_calib)] = out

    # (c) scene-length sweep, fixed kappa=2.0, n_calib=40
    for scene_length in [20, 40]:
        print(f"\n=== scene_length={scene_length} ===")
        t0 = time.time()
        out = run_one(model, dataset, acc_indices, fall_indices, n_calib=40, scene_length=scene_length, kappa=2.0, device=device)
        print(f"  ({time.time()-t0:.1f}s) q_hat={out['q_hat']:.3f}")
        for pt in out["results"]["covariate_blind_agrapa"]:
            print(f"  blind delta={pt['delta']:.2f} delay={pt['mean_detection_delay']} fa={pt['false_alarm_rate']:.2f} censored={pt['n_censored']}")
        all_results["scene_length_sweep"][str(scene_length)] = out

    with open(args.out, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
