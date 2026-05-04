# CRAF-X: Cross-modal Robust Adaptive Fusion with eXplainability

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**CRAF-X** is a novel defense-oriented fusion framework for 3D object detection in autonomous perception. It pioneers a verify-then-fuse paradigm by deploying a Cross-modal Consistency Probe (CCP) that natively detects geometric and semantic contradictions in the shared Bird's-Eye View (BEV) space, providing robustness against multi-modal adversarial attacks and severe sensor degradation. 

This repository contains the official PyTorch implementation, testing scripts, and pre-trained weights for the algorithms described in our manuscript.

## Citation

If you find this code or our research helpful in your work, please cite our paper published in **The Visual Computer**:

```bibtex
@article{crafx_under_review,
  title={{CRAF-X}: Cross-modal Robust Adaptive Fusion with eXplainability for Autonomous Perception},
  author={Anonymous Author},
  journal={Under review at The Visual Computer},
  year={2026}
}
```

## Key Algorithms & Components

1. **Cross-modal Consistency Probe (CCP)**: Acts as an intrinsic anomaly detector. It computes semantic alignment scores between LiDAR geometry and Camera features to dynamically detect adversarial patches or point-cloud displacements.
2. **Gated Adaptive Fusion Module (GAFM)**: Dynamically quarantines adversarial or degraded signals at the BEV grid-cell level. It shifts trust weights automatically when sensor dropout occurs.
3. **Adversarial Consistency Training (ACT)**: A joint objective combining detection loss, consistency contrastive loss, and Modal Attribution Regularization. It ensures the network produces interpretable, auditor-friendly spatial trust maps directly tied to its predictions.

Full implementations of these modules can be found in `craf_x/models/`.

## Dependencies and Requirements

This codebase has been tested under the following environment:
- OS: Ubuntu 20.04 / macOS
- Python: 3.9+
- PyTorch: 2.1.0+
- CUDA: 11.8 (for Linux GPU execution)

**Installation Steps:**
```bash
# Clone the repository
git clone https://github.com/vinhqdang/CRAFT.git
cd CRAFT

# Create and activate conda environment
conda create -n crafx python=3.9 -y
conda activate crafx

# Install dependencies
pip install -r requirements.txt

# (Optional) Compile custom CUDA ops for SparseConvNet
python setup.py develop
```

## Data Sets

CRAF-X is evaluated on the standard multi-modal autonomous driving benchmarks. You must download the datasets directly from their official providers and organize them into the `data/` directory.

- **nuScenes**: The primary robustness benchmark. Download the full dataset from [nuscenes.org](https://www.nuscenes.org/download) and extract to `data/nuscenes/`.
- **KITTI 3D Object Detection**: Used for secondary benchmarking. Download from [cvlibs.net](http://www.cvlibs.net/datasets/kitti/) and extract to `data/kitti/`.
- **Waymo Open Dataset**: Used for evaluating scalability. Download from [waymo.com/open/](https://waymo.com/open/download/) and extract to `data/waymo/`.

After downloading, run the dataset preparation scripts:
```bash
python tools/create_data.py nuscenes --root-path ./data/nuscenes --out-dir ./data/nuscenes --extra-tag nuscenes
```

## Reproducing Experiments

To test the architecture and generate evaluation visualizations locally:

```bash
# Run unit tests to verify modules
python tests/run_all.py

# Evaluate a pre-trained model under adversarial attack
python tests/run_evaluation.py --checkpoint checkpoints/crafx_nuscenes.pth --attack simultaneous_pgd
```

For more detailed guides on training your own models with Adversarial Consistency Training (ACT), please refer to the `docs/` folder (coming soon upon full open-source release).
