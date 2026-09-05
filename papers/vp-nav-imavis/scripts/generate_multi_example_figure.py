"""
Multi-example qualitative figure for the VP-NAV journal paper: THREE real
accumulated (nominal) and THREE real falling (degraded) val-split frames
(not just one pair, unlike the ICRA paper's figure), run through the
corrected checkpoint, showing camera image, LiDAR BEV occupancy, and CCP
disagreement heatmap for each -- using the journal format's extra room to
tell the qualitative story across several real examples rather than one.

Usage:
    python generate_multi_example_figure.py \
        --zip-path ../../../data/ROADVIEW5k.zip \
        --checkpoint ../../../checkpoints/snowy_scenes_fixed/checkpoint_final.pth \
        --split val --device cuda \
        --out ../manuscript/figures/qualitative_multi.pdf
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "conformal-snow-icra2027")))

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
import matplotlib.pyplot as plt
import torch

from craf_x.config import CRAFXConfig
from craf_x.datasets.snowy_scenes_dataset import SNOWY_SCENES_NUM_CLASSES, CRAFXSnowyScenesDataset
from craf_x.models.crafx_net import CRAFX_Net
from craf_x.utils.visualization import plot_tensor_as_image, plot_heatmap
from conformal_monitor.real_snow_stream import category_indices

N_PER_CATEGORY = 3


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zip-path", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--bev-size", type=int, default=128)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out", default="../manuscript/figures/qualitative_multi.pdf")
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
    # Spread picks across the available pool rather than the first N
    # consecutive frames, for more representative variety.
    acc_picks = [acc_indices[i * len(acc_indices) // N_PER_CATEGORY] for i in range(N_PER_CATEGORY)]
    fall_picks = [fall_indices[i * len(fall_indices) // N_PER_CATEGORY] for i in range(N_PER_CATEGORY)]
    print(f"accumulated picks: {acc_picks}")
    print(f"falling picks: {fall_picks}")

    rows = [("accumulated (nominal)", idx) for idx in acc_picks] + [("falling (degraded)", idx) for idx in fall_picks]

    fig, axes = plt.subplots(len(rows), 3, figsize=(7.16, 1.55 * len(rows)))
    mean_disagreements = []

    with torch.no_grad():
        for row, (label, idx) in enumerate(rows):
            sample = dataset[idx]
            image = sample["image"].unsqueeze(0).to(device)
            pointcloud = sample["pointcloud"].unsqueeze(0).to(device)
            out = model(image, pointcloud)
            S = out["S"][0]
            disagreement = 1.0 - S
            mean_disagree = disagreement.mean().item()
            mean_disagreements.append((label, idx, mean_disagree))

            plot_tensor_as_image(axes[row, 0], sample["image"], f"{label} [{idx}]\ncamera")
            occupancy = sample["pointcloud"][0:1]
            plot_heatmap(axes[row, 1], occupancy, "LiDAR BEV\noccupancy", cmap="gray")
            im = plot_heatmap(axes[row, 2], disagreement, f"CCP disagreement\n(mean={mean_disagree:.3f})")
            fig.colorbar(im, ax=axes[row, 2], fraction=0.046, pad=0.04)

    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, dpi=200, bbox_inches="tight")
    print(f"Wrote {args.out}")

    print("\nSummary (mean CCP disagreement per frame):")
    for label, idx, d in mean_disagreements:
        print(f"  {label} [{idx}]: {d:.4f}")
    acc_mean = sum(d for l, i, d in mean_disagreements if "accumulated" in l) / N_PER_CATEGORY
    fall_mean = sum(d for l, i, d in mean_disagreements if "falling" in l) / N_PER_CATEGORY
    print(f"Category means: accumulated={acc_mean:.4f}, falling={fall_mean:.4f}")


if __name__ == "__main__":
    main()
