import torch
import torch.nn as nn

class GatedAdaptiveFusionModule(nn.Module):
    """
    Gated Adaptive Fusion Module (GAFM)
    Fuses F_cam and F_lid using the attribution map A and consistency score S.
    """
    def __init__(self, in_channels: int, tau: float = 0.5):
        super().__init__()
        self.tau = tau
        
        # W_cam and W_lid learned projections
        self.w_cam = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.w_lid = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        
        # Interaction MLP
        self.mlp_cross = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, kernel_size=1)
        )
        
    def forward(self, f_cam: torch.Tensor, f_lid: torch.Tensor, a: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        """
        Args:
            f_cam: (B, C, H, W)
            f_lid: (B, C, H, W)
            a: (B, 2, H, W) Attribution map
            s: (B, 1, H, W) Consistency score map
        Returns:
            f_fused: (B, C, H, W)
        """
        a_cam = a[:, 0:1, :, :]
        a_lid = a[:, 1:2, :, :]
        
        # Additive combination
        f_additive = a_cam * self.w_cam(f_cam) + a_lid * self.w_lid(f_lid)
        
        # Interaction term (Hadamard product)
        f_interact_base = f_cam * f_lid
        f_interact = self.mlp_cross(f_interact_base)
        
        # Mask interaction where S < tau
        # s >= tau -> 1.0, s < tau -> 0.0
        interaction_mask = (s >= self.tau).float()
        
        # F_fused
        f_fused = f_additive + f_interact * interaction_mask
        
        return f_fused
