"""
Runs the conformal_monitor operating-curve evaluation against the REAL CADC
archive using a TRAINED CRAFX_Net checkpoint (CADC counterpart of
run_real_operating_curve.py's Snowy Scenes run).

Unlike Snowy Scenes (whose accumulated/falling/highway splits carry no real
severity label -- see real_snow_stream.py), CADC's own devkit ships genuine
per-drive road-condition metadata (cadc_dataset_route_stats.csv's "Road snow
cover" column, encoded into CRAFXCADCDataset's sample_indices as the "bare"
vs. "covered" category prefix -- see cadc_dataset.py's _DATE_CATEGORY). That
real drive-level split lets calibration and nominal-stream frames be drawn
from *disjoint whole bare drives* rather than a frame-count split within one
pooled category -- avoiding the temporal-adjacency leakage the Snowy Scenes
driver accepted (there, every bare-equivalent frame came from the same
handful of sequences, just split by count).

Usage:
    python run_real_operating_curve_cadc.py \
        --data-root ../../../data/cadcd \
        --checkpoint ../../../checkpoints/cadc/checkpoint_final.pth \
        --device cuda
"""
import argparse
import json
import os
import sys
import time
from collections import defaultdict

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from torch.utils.data import Subset

from craf_x.config import CRAFXConfig
from craf_x.datasets.cadc_dataset import CADC_NUM_CLASSES, CRAFXCADCDataset
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
N_CALIBRATION_DRIVES = 2  # held-out whole bare drives, disjoint from the nominal-stream drives


def _drives_for_category(dataset, category):
    """dict[(date, drive)] -> sorted list of frame indices, restricted to one road-condition category.

    Parses sample_indices of the form f"{category}_{date}_{drive}_{frame:010d}"
    where date is itself "YYYY_MM_DD" (3 underscore-separated parts).
    """
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


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--bev-size", type=int, default=128, help="Must match the checkpoint's training bev size.")
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu",
        help="Defaults to 'cuda' if available, else 'cpu'.",
    )
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

    if len(bare_drives) < N_CALIBRATION_DRIVES + 1:
        raise ValueError(
            f"Need at least {N_CALIBRATION_DRIVES + 1} bare drives (>= {N_CALIBRATION_DRIVES} for "
            f"calibration + >=1 for the nominal stream), found {len(bare_drives)}."
        )
    if len(covered_indices) < SCENE_LENGTH:
        raise ValueError(f"Need at least {SCENE_LENGTH} covered frames, found {len(covered_indices)}.")

    sorted_drives = sorted(bare_drives.keys())
    calibration_drives = sorted_drives[:N_CALIBRATION_DRIVES]
    nominal_drives = sorted_drives[N_CALIBRATION_DRIVES:]
    calibration_indices = [i for d in calibration_drives for i in bare_drives[d]]
    nominal_indices = [i for d in nominal_drives for i in bare_drives[d]]
    print(f"Calibration drives: {calibration_drives} ({len(calibration_indices)} frames)")
    print(f"Nominal-stream drives: {nominal_drives} ({len(nominal_indices)} frames)")

    if len(nominal_indices) < SCENE_LENGTH:
        raise ValueError(
            f"Held-out nominal bare drives give only {len(nominal_indices)} frames, need >= {SCENE_LENGTH}."
        )

    calibration_set = Subset(dataset, calibration_indices)
    nominal_set = Subset(dataset, nominal_indices)
    degraded_set = Subset(dataset, covered_indices)

    print(f"Calibrating on {len(calibration_set)} held-out bare (whole-drive) frames...")
    q_hat = calibrate_on_clear_weather(model, calibration_set, CONFORMAL_ALPHA, batch_size=4, num_workers=4)
    print(f"q_hat = {q_hat:.4f}")

    def make_onset_stream():
        return RealSnowOnsetStream(nominal_set, degraded_set, onset_frame=ONSET_FRAME, scene_length=SCENE_LENGTH)

    def make_clear_stream():
        # No frames are held back a third way -- the false-alarm-rate control
        # reuses nominal_set on both sides of the splice, same substitution
        # run_real_operating_curve.py makes for Snowy Scenes.
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

    out_path = os.path.join(os.path.dirname(__file__), "..", "real_operating_curve_cadc_run.json")
    with open(out_path, "w") as f:
        json.dump(
            {
                "note": (
                    "Real CADC data (bare vs. snow-covered road condition, from the "
                    "devkit's own per-drive metadata) + trained CRAFX_Net checkpoint. "
                    "Calibration and nominal-stream frames come from disjoint whole "
                    "bare drives (not a frame-count split within pooled frames), "
                    "avoiding the temporal-adjacency leakage the Snowy Scenes driver "
                    "accepted."
                ),
                "checkpoint": args.checkpoint,
                "checkpoint_epoch": checkpoint.get("epoch"),
                "calibration_drives": calibration_drives,
                "nominal_drives": nominal_drives,
                "config": {
                    "conformal_alpha": CONFORMAL_ALPHA,
                    "scene_length": SCENE_LENGTH,
                    "onset_frame": ONSET_FRAME,
                    "deltas": DELTAS,
                    "n_onset_replicates": N_ONSET_REPLICATES,
                    "n_clear_replicates": N_CLEAR_REPLICATES,
                    "kappa": KAPPA,
                    "n_calibration_drives": N_CALIBRATION_DRIVES,
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
