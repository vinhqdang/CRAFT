import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "conformal-snow-icra2027")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "scripts")))

import pytest
import torch
import torch.nn as nn
from torch.utils.data import Dataset

from diagnose_detector_collapse import head_statistics, zero_prediction_ablation

ALPHA = 0.2
GRID = 8
N_CLASSES = 3
N_BOX = 6


class _FixedOutputModel(nn.Module):
    """
    Stub detector with directly controllable heads, so the diagnostic's own
    logic can be tested without depending on a real checkpoint.

    `box_scale=0` reproduces a collapsed box head (constant zero output);
    `heatmap_object_boost` controls whether the heatmap can separate a
    ground-truth object cell from an empty one.
    """

    def __init__(self, box_scale: float, heatmap_object_boost: float):
        super().__init__()
        self.box_scale = box_scale
        self.heatmap_object_boost = heatmap_object_boost
        # Gives the module a device to report via next(model.parameters()).
        self._anchor = nn.Parameter(torch.zeros(1))

    def forward(self, image, pointcloud):
        b = image.shape[0]
        device = image.device
        heatmap = torch.full((b, N_CLASSES, GRID, GRID), 0.04, device=device)
        # Put any object-cell boost on the diagonal, where the fixture's
        # ground-truth objects live.
        for i in range(GRID):
            heatmap[:, 0, i, i] += self.heatmap_object_boost
        # A box prediction that genuinely varies with the input.
        box = self.box_scale * pointcloud[:, :1].mean() * torch.ones(
            (b, N_BOX, GRID, GRID), device=device
        )
        return {"H": heatmap, "B": box, "S": torch.full((b, 1, GRID, GRID), 0.9, device=device)}


class _TinyDataset(Dataset):
    """Frames whose ground-truth boxes vary, so miscoverage is non-degenerate."""

    def __init__(self, n, box_value):
        self.n = n
        self.box_value = box_value

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        heatmap = torch.zeros(N_CLASSES, GRID, GRID)
        for i in range(GRID):
            heatmap[0, i, i] = 1.0
        return {
            "image": torch.full((3, GRID, GRID), 0.5),
            "pointcloud": torch.full((4, GRID, GRID), float(idx + 1)),
            "targets": {
                "H": heatmap,
                "B": torch.full((N_BOX, GRID, GRID), self.box_value * (idx + 1)),
                "V": torch.zeros(2, GRID, GRID),
            },
        }


class _Setting:
    def __init__(self, calibration_set, nominal_set, degraded_set):
        self.calibration_set = calibration_set
        self.nominal_set = nominal_set
        self.degraded_set = degraded_set


def _setting():
    return _Setting(_TinyDataset(6, 0.5), _TinyDataset(6, 0.5), _TinyDataset(6, 2.0))


def test_head_statistics_flags_no_object_empty_separation_when_collapsed():
    model = _FixedOutputModel(box_scale=0.0, heatmap_object_boost=0.0)
    stats = head_statistics(model, _TinyDataset(4, 0.5), n_frames=4)

    assert stats["object_vs_empty_separation"] == pytest.approx(0.0, abs=1e-9)
    # A collapsed heatmap sitting at the noise floor puts nothing above any
    # real detection threshold.
    for frac in stats["frac_empty_cells_above"].values():
        assert frac == 0.0


def test_head_statistics_detects_a_healthy_heatmap():
    model = _FixedOutputModel(box_scale=0.0, heatmap_object_boost=0.5)
    stats = head_statistics(model, _TinyDataset(4, 0.5), n_frames=4)

    assert stats["object_vs_empty_separation"] > 0.4
    assert stats["frac_empty_cells_above"]["0.05"] == 0.0  # empty cells stay at 0.04


def test_zero_prediction_ablation_is_identical_for_a_collapsed_box_head():
    # The decisive check: when the box head contributes nothing, zeroing it
    # must leave q_hat and both regimes' miscoverage untouched.
    model = _FixedOutputModel(box_scale=0.0, heatmap_object_boost=0.0)
    ablation = zero_prediction_ablation(model, _setting(), ALPHA, n_frames=6)

    real, zeroed = ablation["real_model"], ablation["zero_prediction"]
    assert real["q_hat"] == pytest.approx(zeroed["q_hat"])
    assert real["nominal_m"] == pytest.approx(zeroed["nominal_m"])
    assert real["degraded_m"] == pytest.approx(zeroed["degraded_m"])


def test_zero_prediction_ablation_differs_for_a_contributing_box_head():
    # Guard against the diagnostic being vacuous: a box head that genuinely
    # affects the residual must produce a detectable difference, otherwise
    # the "identical" result above would carry no information.
    # box_scale is deliberately not 1.0: with this fixture's targets at half
    # the pointcloud value, |pred - target| would then coincidentally equal
    # |0 - target| and the two branches would agree for the wrong reason.
    model = _FixedOutputModel(box_scale=2.0, heatmap_object_boost=0.0)
    ablation = zero_prediction_ablation(model, _setting(), ALPHA, n_frames=6)

    real, zeroed = ablation["real_model"], ablation["zero_prediction"]
    assert real["q_hat"] != pytest.approx(zeroed["q_hat"])


def test_the_two_gates_are_independent():
    # The box-head gate and the heatmap gate govern different downstream
    # variants and must not be conflated: all three call sites in
    # conformal_monitor.evaluate build the match mask from the GROUND-TRUTH
    # heatmap, so the baseline nonconformity score depends on the box head
    # alone. A model with a working box head but a flat heatmap must pass
    # the box-head gate and fail the heatmap gate.
    from diagnose_detector_collapse import (
        M_MATERIAL_EPSILON,
        Q_MATERIAL_EPSILON,
        SEPARATION_EPSILON,
    )

    working_box_flat_heatmap = _FixedOutputModel(box_scale=2.0, heatmap_object_boost=0.0)
    stats = head_statistics(working_box_flat_heatmap, _TinyDataset(4, 0.5), n_frames=4)
    ablation = zero_prediction_ablation(working_box_flat_heatmap, _setting(), ALPHA, n_frames=6)

    q_delta = abs(ablation["real_model"]["q_hat"] - ablation["zero_prediction"]["q_hat"])
    m_delta = abs(ablation["real_model"]["nominal_m"] - ablation["zero_prediction"]["nominal_m"])

    assert q_delta >= Q_MATERIAL_EPSILON or m_delta >= M_MATERIAL_EPSILON  # box gate passes
    assert stats["object_vs_empty_separation"] < SEPARATION_EPSILON  # heatmap gate fails
