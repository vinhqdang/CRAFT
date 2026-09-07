import pytest
import torch

from craf_x.utils.losses import compute_det_loss, masked_l1_loss, penalty_reduced_focal_loss
from craf_x.utils.targets import draw_gaussian, object_center_mask


def _gaussian_target(n_classes=2, size=16, centers=((8, 8),), radius=2):
    heatmap = torch.zeros(n_classes, size, size)
    for center in centers:
        draw_gaussian(heatmap[0], center, radius=radius)
    return heatmap


def test_focal_loss_is_lower_for_a_better_prediction():
    target = _gaussian_target()
    good = target.clone().clamp(1e-4, 1 - 1e-4).unsqueeze(0)
    bad = torch.full_like(good, 0.5)

    assert penalty_reduced_focal_loss(good, target.unsqueeze(0)) < penalty_reduced_focal_loss(
        bad, target.unsqueeze(0)
    )


def test_focal_loss_penalizes_the_constant_predictor_that_mse_rewarded():
    # The whole point of the change: a near-zero constant is what MSE drove
    # the old head to. Focal loss must prefer a prediction that actually
    # peaks at the object over that constant.
    target = _gaussian_target().unsqueeze(0)
    constant = torch.full_like(target, 0.04)
    peaked = target.clone().clamp(1e-4, 1 - 1e-4)

    assert penalty_reduced_focal_loss(peaked, target) < penalty_reduced_focal_loss(constant, target)


def test_focal_loss_handles_a_frame_with_no_objects():
    target = torch.zeros(1, 2, 8, 8)
    pred = torch.full_like(target, 0.01)

    loss = penalty_reduced_focal_loss(pred, target)
    assert torch.isfinite(loss)
    assert loss >= 0.0


def test_focal_loss_is_finite_at_saturated_predictions():
    # Guards the clamp: log(0) / log(1) must not leak through as inf/nan.
    target = _gaussian_target().unsqueeze(0)
    for value in (0.0, 1.0):
        loss = penalty_reduced_focal_loss(torch.full_like(target, value), target)
        assert torch.isfinite(loss)


def test_masked_l1_ignores_cells_outside_the_mask():
    pred = torch.zeros(1, 6, 8, 8)
    target = torch.zeros(1, 6, 8, 8)
    mask = torch.zeros(1, 1, 8, 8)
    mask[0, 0, 4, 4] = 1.0
    # Huge error outside the mask must not register at all.
    target[0, :, 0, 0] = 1000.0
    target[0, :, 4, 4] = 2.0

    assert masked_l1_loss(pred, target, mask) == pytest.approx(2.0)


def test_masked_l1_does_not_reward_predicting_zero_everywhere():
    # The failure mode that collapsed the old box head: with an unmasked
    # loss over a mostly-zero-target grid, the all-zero prediction wins.
    # Under the masked loss it must lose to a prediction that matches the
    # real targets at the object cells.
    target = torch.zeros(1, 6, 16, 16)
    mask = torch.zeros(1, 1, 16, 16)
    for cell in ((4, 4), (9, 11)):
        mask[0, 0, cell[0], cell[1]] = 1.0
        target[0, :, cell[0], cell[1]] = 3.0

    all_zero = torch.zeros_like(target)
    correct = target.clone()

    assert masked_l1_loss(correct, target, mask) < masked_l1_loss(all_zero, target, mask)
    assert masked_l1_loss(correct, target, mask) == pytest.approx(0.0)


def test_masked_l1_returns_zero_when_no_cells_are_supervised():
    pred = torch.ones(1, 6, 8, 8, requires_grad=True)
    target = torch.zeros(1, 6, 8, 8)
    mask = torch.zeros(1, 1, 8, 8)

    loss = masked_l1_loss(pred, target, mask)
    assert loss.item() == pytest.approx(0.0)
    loss.backward()  # must stay differentiable rather than detaching
    assert pred.grad is not None


def test_compute_det_loss_derives_the_object_mask_from_the_target_heatmap():
    heatmap = _gaussian_target().unsqueeze(0)
    targets = {
        "H": heatmap,
        "B": torch.zeros(1, 6, 16, 16),
        "V": torch.zeros(1, 2, 16, 16),
    }
    preds = {
        "H": torch.full_like(heatmap, 0.02),
        "B": torch.zeros(1, 6, 16, 16),
        "V": torch.zeros(1, 2, 16, 16),
    }

    explicit = compute_det_loss(preds, targets, object_mask=object_center_mask(heatmap))
    derived = compute_det_loss(preds, targets)
    assert derived == pytest.approx(explicit)


def test_compute_det_loss_prefers_a_detector_over_a_constant_predictor():
    # End-to-end statement of the fix: the trained-looking prediction must
    # score better than the degenerate constant one. Under the previous
    # MSE + unmasked-L1 loss the ordering was the other way round, which is
    # why both checkpoints collapsed.
    heatmap = _gaussian_target(centers=((5, 5), (11, 12)))
    mask = object_center_mask(heatmap)
    box_target = torch.zeros(6, 16, 16)
    box_target[:, mask[0].bool()] = 2.5

    targets = {"H": heatmap.unsqueeze(0), "B": box_target.unsqueeze(0), "V": torch.zeros(1, 2, 16, 16)}
    good = {
        "H": heatmap.clone().clamp(1e-4, 1 - 1e-4).unsqueeze(0),
        "B": box_target.clone().unsqueeze(0),
        "V": torch.zeros(1, 2, 16, 16),
    }
    collapsed = {
        "H": torch.full_like(heatmap, 0.04).unsqueeze(0),
        "B": torch.full_like(box_target, 0.001).unsqueeze(0),
        "V": torch.zeros(1, 2, 16, 16),
    }

    assert compute_det_loss(good, targets) < compute_det_loss(collapsed, targets)


def test_compute_det_loss_is_differentiable():
    heatmap = _gaussian_target().unsqueeze(0)
    targets = {"H": heatmap, "B": torch.zeros(1, 6, 16, 16), "V": torch.zeros(1, 2, 16, 16)}
    preds = {
        "H": torch.full_like(heatmap, 0.3).requires_grad_(True),
        "B": torch.zeros(1, 6, 16, 16, requires_grad=True),
        "V": torch.zeros(1, 2, 16, 16, requires_grad=True),
    }

    compute_det_loss(preds, targets).backward()
    assert preds["H"].grad is not None
    assert torch.isfinite(preds["H"].grad).all()
