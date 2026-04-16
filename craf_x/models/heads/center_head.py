import torch
import torch.nn as nn
from typing import Tuple
from ...config import CRAFXConfig

class CRAFXCenterHead(nn.Module):
    """
    Multi-task CenterPoint detection head processing the robust unified fused tensor.
    """
    def __init__(self, config: CRAFXConfig):
        super().__init__()
        in_channels = config.bev_channels
        
        # Shared Convolutional trunk for higher capacity processing post-fusion
        self.shared_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )
        
        # Task 1: Semantic Center Heatmap mapping
        self.heatmap_head = nn.Conv2d(in_channels, 10, kernel_size=1)
        
        # Task 2: 3D Regression Sub-head (X, Y, Z offsets + dimensions W, L, H)
        self.regression_head = nn.Conv2d(in_channels, 6, kernel_size=1)
        
        # Task 3: Velocity tracking representation (Vx, Vy)
        self.velocity_head = nn.Conv2d(in_channels, 2, kernel_size=1)
        
        # We explicitly initialize the heatmap bias based on Focal Loss standard anchors
        self.heatmap_head.bias.data.fill_(-2.19)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        features = self.shared_conv(x)
        
        # Heatmap operates under sigmoid constraints
        heatmap = torch.sigmoid(self.heatmap_head(features)) 
        bboxes = self.regression_head(features)
        velocity = self.velocity_head(features)
        
        return heatmap, bboxes, velocity
