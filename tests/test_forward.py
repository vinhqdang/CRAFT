import torch
import pytest
from craf_x.config import CRAFXConfig
from craf_x.models import CRAFX_Net

def test_forward_pass():
    config = CRAFXConfig(bev_h=32, bev_w=32)
    model = CRAFX_Net(config)
    
    # Dummy inputs
    # Let batch_size = 2
    image = torch.randn(2, 3, 32, 32)
    pointcloud = torch.randn(2, 4, 32, 32)
    
    out = model(image, pointcloud)
    
    # Check shapes
    assert 'H' in out
    assert 'S' in out
    assert 'A' in out
    
    assert out['S'].shape == (2, 1, 32, 32)
    assert out['A'].shape == (2, 2, 32, 32)
    
    # A should be clamped
    assert torch.all(out['A'] >= 0.0)
    assert torch.all(out['A'] <= 1.0)
    
    assert out['F_fused'].shape == (2, config.bev_channels, 32, 32)
