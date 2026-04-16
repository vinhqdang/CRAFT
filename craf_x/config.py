import dataclasses

@dataclasses.dataclass
class CRAFXConfig:
    # CCP Parameters
    tau: float = 0.5  # threshold for gating interaction MLP
    alpha: float = 2.0 # temperature for A_cam calculation
    beta: float = 2.0  # temperature for A_lid calculation
    
    # Loss Weights
    lambda_1: float = 1.0  # CCP contrastive loss weight
    lambda_2: float = 1.0  # ACT loss weight
    lambda_3: float = 0.1  # MAR loss weight
    gamma: float = 1.0     # KL divergence weight in ACT
    mu: float = 0.01       # TV loss weight in MAR
    
    # PGD parameters
    epsilon_cam: float = 4 / 255.0
    epsilon_lid: float = 0.1
    pgd_k: int = 7
    
    # Network dims
    bev_channels: int = 256
    bev_h: int = 128
    bev_w: int = 128
