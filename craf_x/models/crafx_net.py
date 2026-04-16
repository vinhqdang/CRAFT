import torch
import torch.nn as nn
from typing import Tuple, Dict

from .ccp import CrossModalConsistencyProbe
from .gafm import GatedAdaptiveFusionModule
from ..config import CRAFXConfig

class DummyEncoder(nn.Module):
    """A dummy encoder returning features shaped as expected."""
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.bn(self.conv(x)))

class DummyDetHead(nn.Module):
    """A dummy detection head."""
    def __init__(self, in_channels: int):
        super().__init__()
        self.heatmap_head = nn.Conv2d(in_channels, 10, kernel_size=1) # 10 classes
        self.bbox_head = nn.Conv2d(in_channels, 6, kernel_size=1) # x, y, z, w, l, h
        self.vel_head = nn.Conv2d(in_channels, 2, kernel_size=1) # vx, vy
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.heatmap_head(x), self.bbox_head(x), self.vel_head(x)

class CRAFX_Net(nn.Module):
    """
    CRAF-X End-to-End Network wrapper for testing.
    """
    def __init__(self, config: CRAFXConfig):
        super().__init__()
        self.config = config
        
        # Mocks for Encoders
        self.cam_enc = DummyEncoder(3, config.bev_channels) # raw image assumed 3 channels 
        self.lid_enc = DummyEncoder(4, config.bev_channels) # raw lidar assumed 4 channels
        
        # Novel Components
        self.ccp = CrossModalConsistencyProbe(
            in_channels=config.bev_channels, 
            alpha=config.alpha, 
            beta=config.beta
        )
        self.gafm = GatedAdaptiveFusionModule(
            in_channels=config.bev_channels, 
            tau=config.tau
        )
        
        # Mock DetHead
        self.head = DummyDetHead(in_channels=config.bev_channels)

    def forward(self, image: torch.Tensor, pointcloud: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            image: Dummy inputs for Camera
            pointcloud: Dummy inputs for LiDAR
        Returns:
            Dict containing outputs: H, B, V, S, A, F_fused, F_cam, F_lid
        """
        f_cam = self.cam_enc(image)
        f_lid = self.lid_enc(pointcloud)
        
        s, a = self.ccp(f_cam, f_lid)
        f_fused = self.gafm(f_cam, f_lid, a, s)
        
        h, b, v = self.head(f_fused)
        
        return {
            "H": h,
            "B": b,
            "V": v,
            "S": s,
            "A": a,
            "F_fused": f_fused,
            "F_cam": f_cam,
            "F_lid": f_lid
        }
