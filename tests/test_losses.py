import torch
import pytest
from craf_x.config import CRAFXConfig
from craf_x.models import CRAFX_Net
from craf_x.training import act_training_step
from craf_x.utils.losses import compute_ccp_loss

def test_act_training_step():
    config = CRAFXConfig(bev_h=32, bev_w=32, pgd_k=2) # fast step
    model = CRAFX_Net(config)
    
    # Dummy inputs
    image = torch.randn(2, 3, 32, 32)
    pointcloud = torch.randn(2, 4, 32, 32)
    m = torch.randint(0, 2, (2, 1, 32, 32)).float()
    
    # Random targets
    targets = {
        'H': torch.randn(2, 10, 32, 32),
        'B': torch.randn(2, 6, 32, 32),
        'V': torch.randn(2, 2, 32, 32)
    }
    
    # Ensure model parameters require grad
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    loss, metrics = act_training_step(model, image, pointcloud, targets, m, config)
    
    optimizer.zero_grad()
    loss.backward()
    
    # Verify gradients flow into at least CCP and GAFM
    assert model.ccp.mlp_s[0].weight.grad is not None
    assert model.gafm.mlp_cross[0].weight.grad is not None
    
    assert metrics['loss'] > 0
    assert metrics['l_act'] > 0


def test_compute_ccp_loss_penalizes_high_score_under_mismatch():
    # A score confidently claiming "agree" (s≈1) under a mismatched mask
    # (m=0) should be penalized far more than one confidently claiming
    # "disagree" (s≈0) under the same mask.
    s_confidently_agrees = torch.full((2, 1, 4, 4), 0.999)
    s_confidently_disagrees = torch.full((2, 1, 4, 4), 0.001)
    zero_mask = torch.zeros(2, 1, 4, 4)

    loss_when_wrong = compute_ccp_loss(s_confidently_agrees, zero_mask)
    loss_when_right = compute_ccp_loss(s_confidently_disagrees, zero_mask)

    assert loss_when_wrong.item() > loss_when_right.item()


def test_act_training_step_supervises_adversarial_ccp_scores():
    # Regression guard for the CCP-collapse bug (see
    # papers/conformal-snow-icra2027/README.md, "First trained run and a
    # real negative result"): the trained CCP score collapsed to ≈constant
    # "agree" because compute_ccp_loss was only ever applied to the clean
    # branch. act_training_step must also apply it to the adversarial
    # branches with a mismatched mask, or nothing in training ever teaches
    # S to disagree.
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

    assert 'l_ccp_clean' in metrics
    assert 'l_ccp_adv' in metrics
    assert metrics['l_ccp_adv'] > 0
    assert metrics['l_ccp'] == pytest.approx(metrics['l_ccp_clean'] + metrics['l_ccp_adv'], rel=1e-4)
