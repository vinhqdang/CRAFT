# CRAF-X: Cross-modal Robust Adaptive Fusion with eXplainability

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20017952.svg)](https://doi.org/10.5281/zenodo.20017952)

**Status:** Under review at *The Visual Computer*.

**CRAF-X** is a defense-oriented fusion framework for 3D object detection in autonomous perception. It pioneers a verify-then-fuse paradigm by deploying a Cross-modal Consistency Probe (CCP) that natively detects geometric and semantic contradictions in the shared Bird's-Eye View (BEV) space, providing robustness against multi-modal adversarial attacks and severe sensor degradation.

This directory contains the manuscript, reviewer correspondence, and paper-specific planning notes for CRAF-X. The implementation itself lives in the shared [`craf_x`](../../craf_x) library at the repository root, since it is reused across papers in this repository — see the [top-level README](../../README.md) for setup and testing instructions.

## Contents

- `manuscript/` — LaTeX source and compiled PDF of the submitted manuscript.
- `plan.md` — related-work survey and algorithm design notes used while drafting the paper.
- `R1_point_to_point_response.txt` — point-to-point response to Round 1 reviewer comments.
- `scripts/` — figure-generation scripts specific to this manuscript (architecture diagram, result charts). Outputs are written to `manuscript/figures/`.

## Citation

If you find this code or research helpful, please cite:

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

Full implementations of these modules can be found in [`craf_x/models/`](../../craf_x/models).

## Data Sets

CRAF-X is evaluated on standard multi-modal autonomous driving benchmarks. Download each dataset directly from its official provider and organize it under a `data/` directory:

- **nuScenes**: The primary robustness benchmark. Download from [nuscenes.org](https://www.nuscenes.org/download) and extract to `data/nuscenes/`.
- **KITTI 3D Object Detection**: Used for secondary benchmarking. Download from [cvlibs.net](http://www.cvlibs.net/datasets/kitti/) and extract to `data/kitti/`.
- **Waymo Open Dataset**: Used for evaluating scalability. Download from [waymo.com/open/](https://waymo.com/open/download/) and extract to `data/waymo/`.

After downloading, run the dataset preparation scripts:

```bash
python tools/create_data.py nuscenes --root-path ./data/nuscenes --out-dir ./data/nuscenes --extra-tag nuscenes
```

## Reproducing Experiments

From the repository root:

```bash
# Run unit tests to verify modules
python tests/run_all.py

# Evaluate a pre-trained model under adversarial attack
python tests/run_evaluation.py --checkpoint checkpoints/crafx_nuscenes.pth --attack simultaneous_pgd

# Regenerate this paper's figures
python papers/craf-x-tvc/scripts/generate_charts.py
python papers/craf-x-tvc/scripts/generate_architecture_diagram.py
```
