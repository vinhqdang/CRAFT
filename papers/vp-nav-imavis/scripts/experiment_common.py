"""
Shared real-dataset setup for the signal-strength ablation scripts.

Every variant in this ablation (phantom-aware score, e-value merging,
lambda-mixture betting) must be evaluated against the exact calibration
protocol, splits and defaults that produced the already-reported
covariate-blind baseline numbers on each dataset, or the comparison is not
apples-to-apples. That protocol differs between the two datasets:

- Snowy Scenes: the first N_CALIBRATION_FRAMES `accumulated` frames
  calibrate; the remaining `accumulated` frames are the nominal stream;
  `falling` frames are the degraded stream.
- CADC: `bare` drives are split by whole drive, stratified across
  collection date (the drive-to-drive heterogeneity fix -- an unstratified
  split produces a 1.00 false-alarm rate); `covered` frames are the
  degraded stream.

Both protocols are reproduced here from the existing evaluation scripts so
each ablation variant inherits them identically, rather than each script
re-deriving them and risking silent drift.
"""
import os
import sys
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch
from torch.utils.data import Dataset, Subset

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "conformal-snow-icra2027")))

from craf_x.config import CRAFXConfig
from craf_x.models.crafx_net import CRAFX_Net

from conformal_monitor.real_snow_stream import category_indices

# Per-dataset protocol constants, matching the existing real evaluations.
DATASET_DEFAULTS: Dict[str, dict] = {
    "snowy": {
        "alpha": 0.2,
        "scene_length": 20,
        "onset_frame": 8,
        "deltas": [0.3, 0.1, 0.05],
        "n_onset_replicates": 5,
        "n_clear_replicates": 5,
        "kappa": 2.0,
        "n_calibration_frames": 40,
        "nominal_category": "accumulated",
        "degraded_category": "falling",
    },
    "cadc": {
        "alpha": 0.2,
        "scene_length": 20,
        "onset_frame": 8,
        "deltas": [0.3, 0.1, 0.05],
        "n_onset_replicates": 5,
        "n_clear_replicates": 5,
        "kappa": 2.0,
        "calibration_fraction": 0.5,
        "nominal_category": "bare",
        "degraded_category": "covered",
    },
}


@dataclass
class RealSetting:
    """Everything an ablation script needs to run one dataset's evaluation."""

    model: CRAFX_Net
    dataset: Dataset
    calibration_set: Subset
    nominal_set: Subset
    degraded_set: Subset
    checkpoint_epoch: object
    provenance: dict


def _load_model(checkpoint_path: str, config: CRAFXConfig, device: torch.device) -> Tuple[CRAFX_Net, object]:
    model = CRAFX_Net(config).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint.get("epoch")


def _drives_for_category(dataset, category: str) -> Dict[Tuple[str, str], List[int]]:
    """Group a CADC category's frame indices by (collection date, drive)."""
    drives: Dict[Tuple[str, str], List[int]] = defaultdict(list)
    prefix = f"{category}_"
    for i, sid in enumerate(dataset.sample_indices):
        if not sid.startswith(prefix):
            continue
        parts = sid[len(prefix):].split("_")
        date = "_".join(parts[:3])
        drive = parts[3]
        drives[(date, drive)].append(i)
    return dict(drives)


def _split_calibration_and_nominal(
    bare_drives: Dict[Tuple[str, str], List[int]], calibration_fraction: float
) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]:
    """Whole-drive split, stratified across collection date (the CADC fix)."""
    by_date: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
    for date, drive in sorted(bare_drives.keys()):
        by_date[date].append((date, drive))

    calibration_drives: List[Tuple[str, str]] = []
    nominal_drives: List[Tuple[str, str]] = []
    for date, drives in by_date.items():
        n_cal = max(1, round(len(drives) * calibration_fraction))
        calibration_drives.extend(drives[:n_cal])
        nominal_drives.extend(drives[n_cal:])
    return calibration_drives, nominal_drives


def _build_snowy(args, defaults: dict, device: torch.device) -> RealSetting:
    from craf_x.datasets.snowy_scenes_dataset import (
        SNOWY_SCENES_NUM_CLASSES,
        CRAFXSnowyScenesDataset,
    )

    if not args.zip_path:
        raise ValueError("--zip-path is required for dataset=snowy")

    config = CRAFXConfig(bev_h=args.bev_size, bev_w=args.bev_size, num_classes=SNOWY_SCENES_NUM_CLASSES)
    dataset = CRAFXSnowyScenesDataset(zip_path=args.zip_path, split=args.split, config=config)
    model, epoch = _load_model(args.checkpoint, config, device)

    n_calib = defaults["n_calibration_frames"]
    nominal_indices = category_indices(dataset, defaults["nominal_category"])
    degraded_indices = category_indices(dataset, defaults["degraded_category"])

    return RealSetting(
        model=model,
        dataset=dataset,
        calibration_set=Subset(dataset, nominal_indices[:n_calib]),
        nominal_set=Subset(dataset, nominal_indices[n_calib:]),
        degraded_set=Subset(dataset, degraded_indices),
        checkpoint_epoch=epoch,
        provenance={
            "split": args.split,
            "n_calibration_frames": n_calib,
            "n_nominal_frames": len(nominal_indices) - n_calib,
            "n_degraded_frames": len(degraded_indices),
        },
    )


def _build_cadc(args, defaults: dict, device: torch.device) -> RealSetting:
    from craf_x.datasets.cadc_dataset import CADC_NUM_CLASSES, CRAFXCADCDataset

    if not args.data_root:
        raise ValueError("--data-root is required for dataset=cadc")

    config = CRAFXConfig(bev_h=args.bev_size, bev_w=args.bev_size, num_classes=CADC_NUM_CLASSES)
    dataset = CRAFXCADCDataset(data_root=args.data_root, config=config)
    model, epoch = _load_model(args.checkpoint, config, device)

    bare_drives = _drives_for_category(dataset, defaults["nominal_category"])
    degraded_indices = category_indices(dataset, defaults["degraded_category"])
    calibration_drives, nominal_drives = _split_calibration_and_nominal(
        bare_drives, defaults["calibration_fraction"]
    )
    calibration_indices = [i for d in calibration_drives for i in bare_drives[d]]
    nominal_indices = [i for d in nominal_drives for i in bare_drives[d]]

    return RealSetting(
        model=model,
        dataset=dataset,
        calibration_set=Subset(dataset, calibration_indices),
        nominal_set=Subset(dataset, nominal_indices),
        degraded_set=Subset(dataset, degraded_indices),
        checkpoint_epoch=epoch,
        provenance={
            "calibration_drives": calibration_drives,
            "nominal_drives": nominal_drives,
            "n_calibration_frames": len(calibration_indices),
            "n_nominal_frames": len(nominal_indices),
            "n_degraded_frames": len(degraded_indices),
        },
    )


def build_real_setting(args) -> RealSetting:
    """Load the model and build the calibration/nominal/degraded splits for
    whichever real dataset `args.dataset` names, using that dataset's own
    established protocol."""
    device = torch.device(args.device)
    defaults = DATASET_DEFAULTS[args.dataset]
    setting = _build_snowy(args, defaults, device) if args.dataset == "snowy" else _build_cadc(
        args, defaults, device
    )
    print(
        f"Loaded {args.dataset} checkpoint {args.checkpoint} (epoch {setting.checkpoint_epoch}); "
        f"{len(setting.calibration_set)} calibration / {len(setting.nominal_set)} nominal / "
        f"{len(setting.degraded_set)} degraded frames"
    )
    return setting
