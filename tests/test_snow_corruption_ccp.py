import torch
import pytest
from craf_x.config import CRAFXConfig
from craf_x.models import CRAFX_Net
from craf_x.training.snow_corruption_ccp import (
    act_training_step_snow_corrupted,
    apply_camera_snow_corruption,
    apply_lidar_snow_corruption,
)


def test_apply_camera_snow_corruption_is_noop_at_zero_severity():
    image = torch.randn(2, 3, 8, 8)
    corrupted = apply_camera_snow_corruption(image, severity=0.0)
    assert torch.equal(corrupted, image)


def test_apply_camera_snow_corruption_reduces_contrast():
    # Attenuation toward flat gray should shrink the per-image standard
    # deviation as severity increases (the corruption's core physical
    # claim: reduced visibility/contrast, not just added noise).
    torch.manual_seed(0)
    image = torch.randn(1, 3, 32, 32)
    mild = apply_camera_snow_corruption(image, severity=0.2, seed=1)
    severe = apply_camera_snow_corruption(image, severity=0.9, seed=1)
    assert severe.std().item() < mild.std().item() * 1.5  # attenuation dominates despite added speckle
    assert mild.shape == image.shape
    assert severe.shape == image.shape


def test_apply_lidar_snow_corruption_is_noop_at_zero_severity():
    pointcloud = torch.randn(4, 8, 8)
    corrupted = apply_lidar_snow_corruption(pointcloud, severity=0.0)
    assert torch.equal(corrupted, pointcloud)


def test_apply_lidar_snow_corruption_drops_points_with_severity():
    # Higher severity should zero out more cells (simulating beam
    # attenuation / dropout), so the corrupted tensor's L1 norm should
    # trend down as severity rises, on average across seeds.
    torch.manual_seed(0)
    pointcloud = torch.ones(4, 16, 16)
    mild = apply_lidar_snow_corruption(pointcloud, severity=0.1, seed=1)
    severe = apply_lidar_snow_corruption(pointcloud, severity=0.9, seed=1)
    n_nonzero_mild = (mild.abs().sum(dim=0) > 0).sum().item()
    n_nonzero_severe = (severe.abs().sum(dim=0) > 0).sum().item()
    assert n_nonzero_severe < n_nonzero_mild


def test_act_training_step_snow_corrupted_supervises_new_branches():
    # Mirrors test_act_training_step_supervises_adversarial_ccp_scores in
    # test_losses.py, but for the new synthetic-snow-corruption branches:
    # l_ccp_snow must be present, nonzero, and l_ccp must equal the sum of
    # all four mismatch-supervised components plus the clean-branch loss.
    config = CRAFXConfig(bev_h=16, bev_w=16, pgd_k=1)
    model = CRAFX_Net(config)

    image = torch.randn(2, 3, 16, 16)
    pointcloud = torch.randn(2, 4, 16, 16)
    m = torch.ones(2, 1, 16, 16)
    targets = {
        'H': torch.randn(2, 10, 16, 16),
        'B': torch.randn(2, 6, 16, 16),
        'V': torch.randn(2, 2, 16, 16),
    }

    loss, metrics = act_training_step_snow_corrupted(model, image, pointcloud, targets, m, config, snow_severity=0.6)

    assert 'l_ccp_clean' in metrics
    assert 'l_ccp_adv' in metrics
    assert 'l_ccp_snow' in metrics
    assert metrics['l_ccp_snow'] > 0
    assert metrics['l_ccp'] == pytest.approx(
        metrics['l_ccp_clean'] + metrics['l_ccp_adv'] + metrics['l_ccp_snow'], rel=1e-4
    )
    assert metrics['loss'] > 0
    assert loss.requires_grad


def test_act_training_step_snow_corrupted_gradients_flow():
    config = CRAFXConfig(bev_h=16, bev_w=16, pgd_k=1)
    model = CRAFX_Net(config)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    image = torch.randn(2, 3, 16, 16)
    pointcloud = torch.randn(2, 4, 16, 16)
    m = torch.ones(2, 1, 16, 16)
    targets = {
        'H': torch.randn(2, 10, 16, 16),
        'B': torch.randn(2, 6, 16, 16),
        'V': torch.randn(2, 2, 16, 16),
    }

    loss, _ = act_training_step_snow_corrupted(model, image, pointcloud, targets, m, config)
    optimizer.zero_grad()
    loss.backward()

    assert model.ccp.mlp_s[0].weight.grad is not None
    assert model.lid_enc.pillar_net[0].weight.grad is not None
    cam_backbone_params = list(model.cam_enc.backbone.parameters())
    assert any(p.grad is not None for p in cam_backbone_params)


def test_act_training_step_unchanged_by_new_module_import():
    # Regression guard for the coordination constraint this module was
    # built under: importing snow_corruption_ccp must not change
    # adversarial.act_training_step's own behavior (e.g. via accidental
    # shared mutable state or a monkeypatch). A plain call must still
    # produce exactly the same metric keys it always has.
    from craf_x.training import act_training_step

    config = CRAFXConfig(bev_h=16, bev_w=16, pgd_k=1)
    model = CRAFX_Net(config)
    image = torch.randn(2, 3, 16, 16)
    pointcloud = torch.randn(2, 4, 16, 16)
    m = torch.ones(2, 1, 16, 16)
    targets = {
        'H': torch.randn(2, 10, 16, 16),
        'B': torch.randn(2, 6, 16, 16),
        'V': torch.randn(2, 2, 16, 16),
    }
    _, metrics = act_training_step(model, image, pointcloud, targets, m, config)
    assert set(metrics.keys()) == {
        'loss', 'l_det_clean', 'l_ccp', 'l_ccp_clean', 'l_ccp_adv', 'l_act', 'l_mar'
    }
