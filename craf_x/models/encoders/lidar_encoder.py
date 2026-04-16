import torch
import torch.nn as nn
import torch.nn.functional as F
from ...config import CRAFXConfig

class CRAFXLidarEncoder(nn.Module):
    """
    PointPillars-inspired LiDAR Voxel map encoder mapping scattered 
    points coordinates into uniform BEV representations natively.
    """
    def __init__(self, config: CRAFXConfig):
        super().__init__()
        self.config = config
        
        self.pillar_net = nn.Sequential(
            nn.Conv2d(4, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(128, config.bev_channels, kernel_size=1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.pillar_net(x)
        return F.interpolate(feat, size=(self.config.bev_h, self.config.bev_w), mode='bilinear', align_corners=False)
