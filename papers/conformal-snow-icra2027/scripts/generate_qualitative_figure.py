"""
Generates the qualitative real-data figure for the manuscript (Section IV):
one `accumulated` and one `falling` frame from the real Snowy Scenes val
split, run through the trained (post-CCP-fix) checkpoint, showing the
camera image, LiDAR BEV occupancy channel, and CCP disagreement heatmap
side by side for each frame -- a visual counterpart to the numbers already
reported in Table I and the CCP-collapse/wrong-direction diagnostic in
Section IV-C/IV-F.

Usage:
    python generate_qualitative_figure.py \
        --zip-path ../../../data/ROADVIEW5k.zip \
        --checkpoint ../../../checkpoints/snowy_scenes_fixed/checkpoint_final.pth \
        --split val --device cuda \
        --out ../manuscript/figures/qualitative.pdf
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

import matplotlib
matplotlib.use("Agg")
# matplotlib's default PDF font embedding is Type 3, which IEEE/ICRA's
# submission portal rejects outright ("Type 3 font ... prevents the file
# from being accepted"). Type 42 (TrueType) is portal-compliant.
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
import matplotlib.pyplot as plt
import torch

from craf_x.config import CRAFXConfig
from craf_x.datasets.snowy_scenes_dataset import SNOWY_SCENES_NUM_CLASSES, CRAFXSnowyScenesDataset
from craf_x.models.crafx_net import CRAFX_Net
from craf_x.utils.visualization import plot_tensor_as_image, plot_heatmap

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from conformal_monitor.real_snow_stream import category_indices


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zip-path", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--bev-size", type=int, default=128)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out", default="../manuscript/figures/qualitative.pdf")
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

    acc_idx = category_indices(dataset, "accumulated")[0]
    fall_idx = category_indices(dataset, "falling")[0]
    print(f"Using accumulated frame index {acc_idx}, falling frame index {fall_idx}")

    fig, axes = plt.subplots(2, 3, figsize=(7.16, 3.55))
    row_labels = ["accumulated (nominal)", "falling (degraded)"]

    with torch.no_grad():
        for row, idx in enumerate([acc_idx, fall_idx]):
            sample = dataset[idx]
            image = sample["image"].unsqueeze(0).to(device)
            pointcloud = sample["pointcloud"].unsqueeze(0).to(device)
            out = model(image, pointcloud)
            S = out["S"][0]  # (1, H, W)
            disagreement = 1.0 - S
            mean_disagree = disagreement.mean().item()

            plot_tensor_as_image(axes[row, 0], sample["image"], f"{row_labels[row]}\ncamera")
            occupancy = sample["pointcloud"][0:1]  # channel 0 = occupancy, (1, H, W)
            plot_heatmap(axes[row, 1], occupancy, "LiDAR BEV\noccupancy", cmap="gray")
            im = plot_heatmap(
                axes[row, 2], disagreement,
                f"CCP disagreement\n(mean={mean_disagree:.3f})",
            )
            fig.colorbar(im, ax=axes[row, 2], fraction=0.046, pad=0.04)

    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, dpi=200, bbox_inches="tight")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
