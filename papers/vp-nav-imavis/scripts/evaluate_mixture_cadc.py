"""
Real mixture-betting evaluation on CADC: adds the equal-weight mixture of
the covariate-blind and CCP-informed bettors to the existing real
operating-curve comparison, using the same protocol/defaults/checkpoint
and the same drive-stratified calibration split as
run_real_operating_curve_cadc.py, for a direct, apples-to-apples
comparison against the existing real CADC result.

Usage:
    python evaluate_mixture_cadc.py \
        --data-root ../../../data/cadcd \
        --checkpoint ../../../checkpoints/cadc/checkpoint_final.pth \
        --device cuda
"""
import argparse
import json
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "conformal-snow-icra2027")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from torch.utils.data import Subset

from craf_x.config import CRAFXConfig
from craf_x.datasets.cadc_dataset import CADC_NUM_CLASSES, CRAFXCADCDataset
from craf_x.models.crafx_net import CRAFX_Net

from conformal_monitor.betting import AGRAPABettor, CCPInformedBettor
from conformal_monitor.evaluate import calibrate_on_clear_weather
from conformal_monitor.real_snow_stream import RealSnowOnsetStream, category_indices

from mixture_monitor.mixture_betting import mixture_operating_curve

CONFORMAL_ALPHA = 0.2
SCENE_LENGTH = 20
ONSET_FRAME = 8
DELTAS = [0.3, 0.1, 0.05]
N_ONSET_REPLICATES = 5
N_CLEAR_REPLICATES = 5
KAPPA = 2.0
CALIBRATION_FRACTION = 0.5


def _drives_for_category(dataset, category):
    """Identical logic to run_real_operating_curve_cadc.py's own helper
    (not imported, since that script has no importable functions of its
    own -- everything lives in main() -- ported instead of duplicated by
    reference)."""
    drives = defaultdict(list)
    prefix = f"{category}_"
    for i, sid in enumerate(dataset.sample_indices):
        if not sid.startswith(prefix):
            continue
        parts = sid[len(prefix):].split("_")
        date = "_".join(parts[:3])
        drive = parts[3]
        drives[(date, drive)].append(i)
    return dict(drives)


def _split_calibration_and_nominal(bare_drives):
    by_date = defaultdict(list)
    for date, drive in sorted(bare_drives.keys()):
        by_date[date].append((date, drive))

    calibration_drives, nominal_drives = [], []
    for date, drives in by_date.items():
        n_cal = max(1, round(len(drives) * CALIBRATION_FRACTION))
        calibration_drives.extend(drives[:n_cal])
        nominal_drives.extend(drives[n_cal:])
    return calibration_drives, nominal_drives


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--bev-size", type=int, default=128)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out", default="../manuscript/mixture_operating_curve_cadc.json")
    return parser.parse_args()


def main():
    args = parse_args()
    config = CRAFXConfig(bev_h=args.bev_size, bev_w=args.bev_size, num_classes=CADC_NUM_CLASSES)
    dataset = CRAFXCADCDataset(data_root=args.data_root, config=config)

    device = torch.device(args.device)
    model = CRAFX_Net(config).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print(f"Loaded checkpoint {args.checkpoint} (epoch {checkpoint.get('epoch')})")

    bare_drives = _drives_for_category(dataset, "bare")
    covered_indices = category_indices(dataset, "covered")
    print(f"{len(bare_drives)} bare drives, {len(covered_indices)} covered frames")

    calibration_drives, nominal_drives = _split_calibration_and_nominal(bare_drives)
    calibration_indices = [i for d in calibration_drives for i in bare_drives[d]]
    nominal_indices = [i for d in nominal_drives for i in bare_drives[d]]
    print(f"Calibration drives: {calibration_drives} ({len(calibration_indices)} frames)")
    print(f"Nominal-stream drives: {nominal_drives} ({len(nominal_indices)} frames)")

    calibration_set = Subset(dataset, calibration_indices)
    nominal_set = Subset(dataset, nominal_indices)
    degraded_set = Subset(dataset, covered_indices)

    print(f"Calibrating on {len(calibration_set)} held-out bare (whole-drive) frames...")
    q_hat = calibrate_on_clear_weather(model, calibration_set, CONFORMAL_ALPHA, batch_size=4, num_workers=4)
    print(f"q_hat = {q_hat:.4f}")

    def make_onset_stream():
        return RealSnowOnsetStream(nominal_set, degraded_set, onset_frame=ONSET_FRAME, scene_length=SCENE_LENGTH)

    def make_clear_stream():
        return RealSnowOnsetStream(nominal_set, nominal_set, onset_frame=ONSET_FRAME, scene_length=SCENE_LENGTH)

    bettor_factories = {
        "covariate_blind_agrapa": lambda: AGRAPABettor(CONFORMAL_ALPHA),
        "ccp_informed": lambda: CCPInformedBettor(AGRAPABettor(CONFORMAL_ALPHA), kappa=KAPPA),
    }
    weights = {"covariate_blind_agrapa": 0.5, "ccp_informed": 0.5}

    print("\nRunning mixture operating curve (equal-weight blind+ccp_informed)...")
    curves = mixture_operating_curve(
        model, q_hat, CONFORMAL_ALPHA, DELTAS,
        onset_stream_factory=make_onset_stream, clear_stream_factory=make_clear_stream,
        bettor_factories=bettor_factories, weights=weights,
        n_onset_replicates=N_ONSET_REPLICATES, n_clear_replicates=N_CLEAR_REPLICATES,
    )
    for name, curve in curves.items():
        print(f"\n{name}:")
        for point in curve:
            print(
                f"  delta={point['delta']:.2f}  false_alarm_rate={point['false_alarm_rate']:.2f}  "
                f"mean_detection_delay={point['mean_detection_delay']}  n_censored={point['n_censored']}"
            )

    out_path = args.out
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(
            {
                "note": "Real CADC mixture-betting evaluation (equal-weight blind+ccp_informed).",
                "checkpoint": args.checkpoint,
                "checkpoint_epoch": checkpoint.get("epoch"),
                "calibration_drives": calibration_drives,
                "nominal_drives": nominal_drives,
                "weights": weights,
                "config": {
                    "conformal_alpha": CONFORMAL_ALPHA, "scene_length": SCENE_LENGTH, "onset_frame": ONSET_FRAME,
                    "deltas": DELTAS, "n_onset_replicates": N_ONSET_REPLICATES,
                    "n_clear_replicates": N_CLEAR_REPLICATES, "kappa": KAPPA,
                    "calibration_fraction": CALIBRATION_FRACTION, "q_hat": q_hat, "bev_size": args.bev_size,
                },
                "results": curves,
            },
            f, indent=2,
        )
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
