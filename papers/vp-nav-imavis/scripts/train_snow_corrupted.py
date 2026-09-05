"""
Standalone training entrypoint for the synthetic-snow-corruption CCP
experiment (craf_x/training/snow_corruption_ccp.py). Deliberately a
separate script from tools/train.py -- not a modification of it -- so
this experiment cannot interfere with anything already running against
that shared entrypoint (e.g. the concurrent CADC training run).

Trains CRAFX_Net on real Snowy Scenes data under
act_training_step_snow_corrupted instead of the existing
act_training_step, to test whether supervising CCP mismatch with
physically-motivated synthetic snow corruption (rather than PGD
adversarial perturbation) produces a covariate that actually helps the
CCP-informed bettor, per papers/vp-nav-imavis/manuscript/5discussion.tex.

Usage:
    python train_snow_corrupted.py --zip-path ../../../data/ROADVIEW5k.zip \
        --split train --epochs 5 --batch-size 4 --num-workers 8 \
        --device cuda --output-dir ../../../checkpoints/snowy_scenes_snow_corrupted
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

import torch
from torch.utils.data import DataLoader

from craf_x.config import CRAFXConfig
from craf_x.datasets.snowy_scenes_dataset import SNOWY_SCENES_NUM_CLASSES, CRAFXSnowyScenesDataset
from craf_x.models.crafx_net import CRAFX_Net
from craf_x.training.snow_corruption_ccp import act_training_step_snow_corrupted


def _move_batch_to_device(batch, device):
    batch["image"] = batch["image"].to(device)
    batch["pointcloud"] = batch["pointcloud"].to(device)
    batch["m"] = batch["m"].to(device)
    batch["targets"] = {k: v.to(device) for k, v in batch["targets"].items()}
    return batch


def train(args):
    config = CRAFXConfig(bev_h=args.bev_size, bev_w=args.bev_size, num_classes=SNOWY_SCENES_NUM_CLASSES)
    dataset = CRAFXSnowyScenesDataset(zip_path=args.zip_path, split=args.split, config=config)
    if len(dataset) < 2:
        raise ValueError(f"Dataset has only {len(dataset)} sample(s); need at least 2.")

    device = torch.device(args.device)
    model = CRAFX_Net(config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=args.num_workers,
    )
    if len(loader) == 0:
        raise ValueError(f"batch_size={args.batch_size} leaves no full batches for dataset size {len(dataset)}.")

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Training on {device}, {len(dataset)} samples, {len(loader)} batches/epoch, "
          f"snow_severity={args.snow_severity}.")

    step = 0
    stop = False
    for epoch in range(args.epochs):
        model.train()
        epoch_start = time.time()
        for batch in loader:
            batch = _move_batch_to_device(batch, device)

            loss, metrics = act_training_step_snow_corrupted(
                model, batch["image"], batch["pointcloud"], batch["targets"], batch["m"], config,
                snow_severity=args.snow_severity,
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
    parser = argparse.ArgumentParser(description="Train CRAFX_Net with synthetic-snow-corrupted CCP supervision.")
    parser.add_argument("--zip-path", required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--bev-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--snow-severity", type=float, default=0.6)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--output-dir", default="checkpoints_snow_corrupted")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
