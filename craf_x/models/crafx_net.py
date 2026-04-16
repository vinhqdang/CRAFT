import torch
import torch.nn as nn
from typing import Tuple, Dict

from .ccp import CrossModalConsistencyProbe
from .gafm import GatedAdaptiveFusionModule
from .encoders.camera_encoder import CRAFXCameraEncoder
from .encoders.lidar_encoder import CRAFXLidarEncoder
from .heads.center_head import CRAFXCenterHead
from ..config import CRAFXConfig

class CRAFX_Net(nn.Module):
    """
    CRAF-X End-to-End Network wrapper natively instantiating structural backbones.
    """
    def __init__(self, config: CRAFXConfig):
        super().__init__()
        self.config = config
        
        # Production Encoders
        self.cam_enc = CRAFXCameraEncoder(config) 
        self.lid_enc = CRAFXLidarEncoder(config)
        
        # Novel Component 1: Cross-Modal Consistency Probe
        self.ccp = CrossModalConsistencyProbe(
            in_channels=config.bev_channels, 
            alpha=config.alpha, 
            beta=config.beta
        )
        
        # Novel Component 2: Gated Adaptive Fusion Module
        self.gafm = GatedAdaptiveFusionModule(
            in_channels=config.bev_channels, 
            tau=config.tau
        )
        
        # Production Detection Head
        self.head = CRAFXCenterHead(config)

    def forward(self, image: torch.Tensor, pointcloud: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            image: Inputs for Camera (B, 3, H_img, W_img)
            pointcloud: Inputs for LiDAR (B, 4, H_lid, W_lid)
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
