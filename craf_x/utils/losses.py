import torch
import torch.nn as nn
import torch.nn.functional as F

def compute_det_loss(preds: dict, targets: dict) -> torch.Tensor:
    """
    Computes a mock detection loss. 
    In reality, this uses Focal Loss for heatmaps and L1 for bboxes/velocities.
    """
    h_pred = preds['H']
    b_pred = preds['B']
    v_pred = preds['V']
    
    h_tgt = targets['H']
    b_tgt = targets['B']
    v_tgt = targets['V']
    
    # Mock losses
    l_h = F.mse_loss(h_pred, h_tgt)  # Should be Focal Loss
    l_b = F.l1_loss(b_pred, b_tgt)
    l_v = F.l1_loss(v_pred, v_tgt)
    
    return l_h + l_b + l_v

def compute_ccp_loss(s: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
    """
    Computes CCP contrastive consistency loss.
    L_ccp = - Σ_{(i,j)∈M} log S(i,j) - Σ_{(i,j)∉M} log(1 - S(i,j))
    
    Args:
        s: (B, 1, H, W) Consistency scores
        m: (B, 1, H, W) Binary mask of geometrically matched cells (1 if matched, 0 otherwise)
    """
    # Clamp s to avoid log(0)
    s_clamped = torch.clamp(s, 1e-7, 1.0 - 1e-7)
    
    loss_matched = -torch.sum(m * torch.log(s_clamped))
    loss_unmatched = -torch.sum((1 - m) * torch.log(1 - s_clamped))
    
    # Normalize by number of elements
    return (loss_matched + loss_unmatched) / m.numel()

def compute_mar_loss(a: torch.Tensor, mu: float = 0.01) -> torch.Tensor:
    """
    Computes Modal Attribution Regularization (MAR) loss.
    L_mar = - Σ_{(i,j)} [A(cam) log A(cam) + A(lid) log A(lid)] + μ ||A||_TV
    
    Args:
        a: (B, 2, H, W) Modal attribution map
        mu: Total Variation weight
    """
    a_clamped = torch.clamp(a, 1e-7, 1.0)
    
    # Entropy Loss
    entropy_loss = -torch.sum(a_clamped * torch.log(a_clamped))
    entropy_loss = entropy_loss / a.numel()
    
    # Total Variation Loss
    tv_h = torch.sum(torch.abs(a[:, :, 1:, :] - a[:, :, :-1, :]))
    tv_w = torch.sum(torch.abs(a[:, :, :, 1:] - a[:, :, :, :-1]))
    tv_loss = (tv_h + tv_w) / a.numel()
    
    return entropy_loss + mu * tv_loss
