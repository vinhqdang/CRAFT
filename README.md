# CRAFT

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository is a multi-paper research codebase. It holds a shared PyTorch library, `craf_x/`, for cross-modal fusion research on 3D object detection in autonomous perception, plus one directory per paper built on top of it under [`papers/`](papers/).

## Layout

```
CRAFT/
├── craf_x/     # shared library: models, datasets, training, evaluation, utils
├── tests/      # shared unit tests and evaluation scripts for craf_x
└── papers/     # one subdirectory per paper (manuscript, planning notes, paper-specific scripts)
    ├── craf-x-tvc/               # CRAF-X, under review at The Visual Computer
    └── conformal-snow-icra2027/  # anytime-valid conformal monitoring, targeting ICRA 2027
```

See [`papers/README.md`](papers/README.md) for the list of papers and how to add a new one, and each paper's own `README.md` for its abstract, status, and citation.

## The `craf_x` library

`craf_x` implements the shared modeling components used across papers in this repository, currently centered on:

1. **Cross-modal Consistency Probe (CCP)**: computes semantic alignment scores between LiDAR geometry and Camera features to detect adversarial patches or point-cloud displacements.
2. **Gated Adaptive Fusion Module (GAFM)**: dynamically quarantines adversarial or degraded modality signals at the BEV grid-cell level.
3. **Adversarial Consistency Training (ACT)**: a joint training objective combining detection loss, consistency contrastive loss, and Modal Attribution Regularization.

Full implementations live in `craf_x/models/`. See [`papers/craf-x-tvc/README.md`](papers/craf-x-tvc/README.md) for the paper these components were introduced in.

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

Experiments in this repository are evaluated on standard multi-modal autonomous driving benchmarks. Download each dataset directly from its official provider and organize it under `data/`:

- **nuScenes**: Download from [nuscenes.org](https://www.nuscenes.org/download) and extract to `data/nuscenes/`.
- **KITTI 3D Object Detection**: Download from [cvlibs.net](http://www.cvlibs.net/datasets/kitti/) and extract to `data/kitti/`.
- **Waymo Open Dataset**: Download from [waymo.com/open/](https://waymo.com/open/download/) and extract to `data/waymo/`.

After downloading, run the dataset preparation scripts:
```bash
python tools/create_data.py nuscenes --root-path ./data/nuscenes --out-dir ./data/nuscenes --extra-tag nuscenes
```

## Running Tests

```bash
# Run unit tests to verify craf_x modules
python tests/run_all.py

# Evaluate a pre-trained model under adversarial attack
python tests/run_evaluation.py --checkpoint checkpoints/crafx_nuscenes.pth --attack simultaneous_pgd
```

For a specific paper's citation, results, and reproduction steps, see that paper's own README under `papers/`.
