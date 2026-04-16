import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from ...config import CRAFXConfig

class CRAFXCameraEncoder(nn.Module):
    """
    Simulates a Lift-Splat-Shoot style Camera Encoder mapping multi-view 
    perspective images into a continuous BEV grid tensor.
    """
    def __init__(self, config: CRAFXConfig, pretrained: bool = False):
        super().__init__()
        self.config = config
        
        # 1. 2D Image Backbone
        base_model = models.resnet18(pretrained=pretrained)
        # Drop classification layers to extract spatial feature maps
        self.backbone = nn.Sequential(*list(base_model.children())[:-2]) 
        
        # 2. View Transformation & Compression Block
        self.depth_projection = nn.Sequential(
            nn.Conv2d(512, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, config.bev_channels, kernel_size=1)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Camera images (B, C, H, W)
        Returns:
            bev_features: (B, bev_channels, H_bev, W_bev)
        """
        features_2d = self.backbone(x)
        bev_features = self.depth_projection(features_2d)
        
        # Force interpolation to the exact spatial grid configured
        return F.interpolate(bev_features, size=(self.config.bev_h, self.config.bev_w), mode='bilinear', align_corners=False)
