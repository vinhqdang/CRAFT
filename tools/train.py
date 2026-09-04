"""
CRAF-X training entrypoint.

Wires together a real dataset (KITTI or Snowy Scenes; extend
`_build_dataset` for others), CRAFX_Net, and the Adversarial Consistency
Training (ACT) step from `craf_x/training/adversarial.py` into an actual
epoch/optimizer loop with checkpointing -- the pieces needed to train on
real data, which previously only existed as a single training-step
function with no runner.

Usage:
    python tools/train.py --dataset kitti --data-root /path/to/kitti \
        --epochs 5 --batch-size 4 --device cuda --output-dir checkpoints/

    python tools/train.py --dataset snowy_scenes --zip-path /path/to/ROADVIEW5k.zip \
        --split train --epochs 1 --batch-size 4 --num-workers 4 --device cuda
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from torch.utils.data import DataLoader

from craf_x.config import CRAFXConfig
from craf_x.datasets.kitti_dataset import KITTI_CLASSES, CRAFXKittiDataset
from craf_x.datasets.snowy_scenes_dataset import SNOWY_SCENES_NUM_CLASSES, CRAFXSnowyScenesDataset
from craf_x.models.crafx_net import CRAFX_Net
from craf_x.training.adversarial import act_training_step


def _build_dataset(args):
    if args.dataset == "kitti":
        split = args.split or "training"
        config = CRAFXConfig(bev_h=args.bev_size, bev_w=args.bev_size, num_classes=len(KITTI_CLASSES))
        dataset = CRAFXKittiDataset(data_root=args.data_root, split=split, config=config)
        if not dataset.is_real:
            print(
                f"WARNING: {args.data_root}/{split} does not look like a real KITTI layout "
                "(expected image_2/, velodyne/, calib/, label_2/ subdirectories). "
                "Training will run against random placeholder tensors.",
                file=sys.stderr,
            )
        return dataset, config

    if args.dataset == "snowy_scenes":
        if not args.zip_path:
            raise ValueError("--zip-path is required for --dataset snowy_scenes")
        split = args.split or "train"
        config = CRAFXConfig(bev_h=args.bev_size, bev_w=args.bev_size, num_classes=SNOWY_SCENES_NUM_CLASSES)
        dataset = CRAFXSnowyScenesDataset(zip_path=args.zip_path, split=split, config=config)
        return dataset, config

    raise ValueError(f"Unknown dataset: {args.dataset}")


def _move_batch_to_device(batch, device):
    batch["image"] = batch["image"].to(device)
    batch["pointcloud"] = batch["pointcloud"].to(device)
    batch["m"] = batch["m"].to(device)
    batch["targets"] = {k: v.to(device) for k, v in batch["targets"].items()}
    return batch


def train(args):
    dataset, config = _build_dataset(args)
    if len(dataset) < 2:
        raise ValueError(
            f"Dataset has only {len(dataset)} sample(s); need at least 2 "
            "(the backbone's BatchNorm layers require batch_size > 1 in training mode)."
        )

    device = torch.device(args.device)
    model = CRAFX_Net(config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,  # avoid a final batch of size 1, which BatchNorm can't handle in training mode
        num_workers=args.num_workers,
    )
    if len(loader) == 0:
        raise ValueError(
            f"batch_size={args.batch_size} leaves no full batches for a dataset of size {len(dataset)} "
            "(drop_last=True). Use a smaller batch_size or a larger dataset."
        )

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Training on {device}, {len(dataset)} samples, {len(loader)} batches/epoch.")

    step = 0
    stop = False
    for epoch in range(args.epochs):
        model.train()
        epoch_start = time.time()
        for batch in loader:
            batch = _move_batch_to_device(batch, device)

            loss, metrics = act_training_step(
                model, batch["image"], batch["pointcloud"], batch["targets"], batch["m"], config
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step % args.log_every == 0:
                metrics_str = " ".join(f"{k}={v:.4f}" for k, v in metrics.items())
                print(f"epoch {epoch} step {step}: {metrics_str}")
            step += 1

            if args.max_steps is not None and step >= args.max_steps:
                stop = True
                break

        checkpoint_path = os.path.join(args.output_dir, f"checkpoint_epoch{epoch}.pth")
        torch.save({"model_state_dict": model.state_dict(), "config": config, "epoch": epoch}, checkpoint_path)
        print(f"epoch {epoch} done in {time.time() - epoch_start:.1f}s, saved {checkpoint_path}")

        if stop:
            print(f"Reached --max-steps={args.max_steps}, stopping early.")
            break

    final_path = os.path.join(args.output_dir, "checkpoint_final.pth")
    torch.save({"model_state_dict": model.state_dict(), "config": config, "epoch": args.epochs - 1}, final_path)
    print(f"Training complete. Final checkpoint: {final_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Train CRAF-X on a real dataset.")
    parser.add_argument("--dataset", choices=["kitti", "snowy_scenes"], default="kitti")
    parser.add_argument("--data-root", help="Root directory of the dataset (KITTI).")
    parser.add_argument("--zip-path", help="Path to the ROADVIEW5k.zip archive (Snowy Scenes).")
    parser.add_argument("--split", default=None, help="Defaults to 'training' for KITTI, 'train' for Snowy Scenes.")
    parser.add_argument("--bev-size", type=int, default=128, help="BEV grid height/width (also image resize size).")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader worker processes.")
    parser.add_argument("--max-steps", type=int, default=None, help="Stop after this many optimizer steps total.")
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--output-dir", default="checkpoints")
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu",
        help="Defaults to 'cuda' if available, else 'cpu'.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
