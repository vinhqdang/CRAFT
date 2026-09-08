"""
Retraining entrypoint for both real datasets after the detection-loss fix.

The previous checkpoints were trained under a placeholder detection loss
(MSE on a >99.99%-zero heatmap, unmasked L1 on box regression) whose
optimum is a constant, and both collapsed accordingly -- see
`papers/vp-nav-imavis/scripts/diagnose_detector_collapse.py` for the
evidence. `craf_x/utils/losses.py` now uses penalty-reduced focal loss on
Gaussian-splatted targets plus object-masked L1, and
`craf_x/utils/targets.py` produces the matching targets.

Kept separate from `tools/train.py` so the shared entrypoint is untouched,
and it writes to its own checkpoint directory so the collapsed checkpoints
survive for the before/after comparison.

Each epoch logs head-health metrics alongside the loss -- heatmap peak
separation between ground-truth object cells and empty cells, and the
spread of the box head's output -- because those, not the loss value, are
what the acceptance gate actually turns on.

Usage (short smoke run first, to confirm the heads learn at all):
    python train_detector.py --dataset snowy --zip-path ../../../data/ROADVIEW5k.zip \
        --max-steps 150 --output-dir ../../../checkpoints/snowy_scenes_smoke

Full run:
    python train_detector.py --dataset snowy --zip-path ../../../data/ROADVIEW5k.zip \
        --epochs 5 --output-dir ../../../checkpoints/snowy_scenes_focal
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

import torch
from torch.utils.data import DataLoader

from craf_x.config import CRAFXConfig
from craf_x.models.crafx_net import CRAFX_Net
from craf_x.training.adversarial import act_training_step
from craf_x.utils.targets import MONITOR_MATCH_THRESHOLD


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["snowy", "cadc"], required=True)
    parser.add_argument("--zip-path", help="Snowy Scenes ROADVIEW5k.zip")
    parser.add_argument("--data-root", help="CADC data root")
    parser.add_argument("--split", default="train")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=None,
                        help="stop after this many steps (smoke run)")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--bev-size", type=int, default=128)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--resume", default=None,
                        help="checkpoint to resume weights from; --epochs then counts "
                             "ADDITIONAL epochs, numbered on from the checkpoint's own")
    return parser.parse_args()


def build_dataset(args, config_num_classes=None):
    if args.dataset == "snowy":
        from craf_x.datasets.snowy_scenes_dataset import (
            SNOWY_SCENES_NUM_CLASSES,
            CRAFXSnowyScenesDataset,
        )
        if not args.zip_path:
            raise ValueError("--zip-path is required for dataset=snowy")
        config = CRAFXConfig(
            bev_h=args.bev_size, bev_w=args.bev_size, num_classes=SNOWY_SCENES_NUM_CLASSES
        )
        return CRAFXSnowyScenesDataset(zip_path=args.zip_path, split=args.split, config=config), config

    from craf_x.datasets.cadc_dataset import CADC_NUM_CLASSES, CRAFXCADCDataset
    if not args.data_root:
        raise ValueError("--data-root is required for dataset=cadc")
    config = CRAFXConfig(bev_h=args.bev_size, bev_w=args.bev_size, num_classes=CADC_NUM_CLASSES)
    # split="train": whole-drive partition. Evaluation drives are never seen
    # in training, so the conformal calibration downstream is out-of-sample.
    return CRAFXCADCDataset(data_root=args.data_root, config=config, split="train"), config


def _move_batch_to_device(batch, device):
    batch["image"] = batch["image"].to(device)
    batch["pointcloud"] = batch["pointcloud"].to(device)
    batch["m"] = batch["m"].to(device)
    batch["targets"] = {k: v.to(device) for k, v in batch["targets"].items()}
    return batch


@torch.no_grad()
def head_health(model, batch):
    """
    The metrics the acceptance gate turns on: can the heatmap separate a
    ground-truth object cell from an empty one, and does the box head
    produce anything other than a constant?
    """
    model.eval()
    out = model(batch["image"], batch["pointcloud"])
    peak = out["H"].amax(dim=1, keepdim=True)
    gt_cells = (batch["targets"]["H"].amax(dim=1, keepdim=True) >= MONITOR_MATCH_THRESHOLD)
    model.train()

    if not gt_cells.any():
        return None
    return {
        "heatmap_peak_at_objects": float(peak[gt_cells].mean().item()),
        "heatmap_peak_at_empty": float(peak[~gt_cells].mean().item()),
        "separation": float(peak[gt_cells].mean().item() - peak[~gt_cells].mean().item()),
        "box_pred_std": float(out["B"].std().item()),
        "box_pred_abs_max": float(out["B"].abs().max().item()),
    }


def train(args):
    dataset, config = build_dataset(args)
    if len(dataset) < 2:
        raise ValueError(f"Dataset has only {len(dataset)} sample(s); need at least 2.")

    device = torch.device(args.device)
    model = CRAFX_Net(config).to(device)

    # Resuming matters here for a practical reason: a full run exceeds the
    # ~1h background-task ceiling in this environment, so training is done
    # in chunks that pick up from the last per-epoch checkpoint. Only the
    # weights are carried over -- Adam's moments are not checkpointed, so a
    # resumed chunk re-warms them over its first few steps.
    start_epoch = 0
    if args.resume:
        resumed = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(resumed["model_state_dict"])
        start_epoch = int(resumed.get("epoch", -1)) + 1
        print(f"Resumed weights from {args.resume} (epoch {resumed.get('epoch')}); "
              f"continuing at epoch {start_epoch}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, drop_last=True,
        num_workers=args.num_workers,
    )
    if len(loader) == 0:
        raise ValueError(f"batch_size={args.batch_size} leaves no full batches for {len(dataset)} samples.")

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Training {args.dataset} on {device}: {len(dataset)} samples, "
          f"{len(loader)} batches/epoch, lr={args.lr}")

    history = []
    step = 0
    stop = False
    for epoch in range(start_epoch, start_epoch + args.epochs):
        model.train()
        epoch_start = time.time()
        epoch_loss, n_batches = 0.0, 0
        last_health = None

        for batch in loader:
            batch = _move_batch_to_device(batch, device)
            loss, metrics = act_training_step(
                model, batch["image"], batch["pointcloud"], batch["targets"], batch["m"], config
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += float(loss.item())
            n_batches += 1

            if step % args.log_every == 0:
                last_health = head_health(model, batch)
                health_str = (
                    f" sep={last_health['separation']:+.5f} "
                    f"box_std={last_health['box_pred_std']:.5f}"
                    if last_health else " (no objects in batch)"
                )
                print(f"  epoch {epoch} step {step} loss={loss.item():.4f}"
                      f" det={metrics.get('l_det_clean', float('nan')):.4f}{health_str}", flush=True)

            step += 1
            if args.max_steps is not None and step >= args.max_steps:
                stop = True
                break

        record = {
            "epoch": epoch,
            "mean_loss": epoch_loss / max(n_batches, 1),
            "seconds": time.time() - epoch_start,
            "health": last_health,
        }
        history.append(record)
        print(f"epoch {epoch}: mean_loss={record['mean_loss']:.4f} "
              f"({record['seconds']:.0f}s)", flush=True)

        torch.save(
            {"model_state_dict": model.state_dict(), "config": config, "epoch": epoch},
            os.path.join(args.output_dir, f"checkpoint_epoch{epoch}.pth"),
        )
        if stop:
            break

    torch.save(
        {"model_state_dict": model.state_dict(), "config": config, "epoch": history[-1]["epoch"]},
        os.path.join(args.output_dir, "checkpoint_final.pth"),
    )
    # Append rather than overwrite, so a chunked run keeps the whole record.
    history_path = os.path.join(args.output_dir, "training_history.json")
    previous = []
    if os.path.exists(history_path):
        with open(history_path) as f:
            previous = json.load(f).get("history", [])
    with open(history_path, "w") as f:
        json.dump(
            {"dataset": args.dataset, "args": vars(args), "history": previous + history},
            f, indent=2,
        )
    print(f"Wrote checkpoints and training_history.json to {args.output_dir}")


if __name__ == "__main__":
    train(parse_args())
