import torch
import torch.nn as nn
from typing import Tuple

class CrossModalConsistencyProbe(nn.Module):
    """
    Cross-modal Consistency Probe (CCP)
    Calculates per-cell consistency score S and modal attribution map A.
    """
    def __init__(self, in_channels: int, alpha: float = 2.0, beta: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        
        # S(i,j) = sigmoid( MLP_s( [F_cam(i,j) ⊕ F_lid(i,j) ⊕ |F_cam(i,j) - F_lid(i,j)|] ) )
        # The input to MLP_s is 3 * in_channels
        self.mlp_s = nn.Sequential(
            nn.Conv2d(3 * in_channels, in_channels, kernel_size=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, 1, kernel_size=1)
        )
        
        # Learned scalar gates initialized to 1.0
        self.gate_cam = nn.Parameter(torch.ones(1))
        self.gate_lid = nn.Parameter(torch.ones(1))
        
    def forward(self, f_cam: torch.Tensor, f_lid: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            f_cam: (B, C, H, W) Camera BEV features
            f_lid: (B, C, H, W) LiDAR BEV features
        Returns:
            s: (B, 1, H, W) Consistency score map
            a: (B, 2, H, W) Modal attribution map (channel 0: cam, channel 1: lid)
        """
        # Calculate absolute difference
        f_diff = torch.abs(f_cam - f_lid)
        
        # Concatenate features along channel dimension
        f_cat = torch.cat([f_cam, f_lid, f_diff], dim=1)
        
        # Compute consistency score S
        s = torch.sigmoid(self.mlp_s(f_cat))  # (B, 1, H, W)
        
        # Compute attribution maps A_cam and A_lid
        # A_cam = 1 - S^alpha * gate_cam
        # A_lid = S^beta * gate_lid
        a_cam = 1.0 - (s ** self.alpha) * self.gate_cam
        a_lid = (s ** self.beta) * self.gate_lid
        
        # Clip to ensure valid probabilities [0, 1]
        a_cam = torch.clamp(a_cam, 0.0, 1.0)
        a_lid = torch.clamp(a_lid, 0.0, 1.0)
        
        # Combine to (B, 2, H, W)
        a = torch.cat([a_cam, a_lid], dim=1)
        
        return s, a
