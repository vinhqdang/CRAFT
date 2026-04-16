import torch
import pytest
from craf_x.config import CRAFXConfig
from craf_x.models import CRAFX_Net
from craf_x.training import act_training_step

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
