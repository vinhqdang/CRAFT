import torch
import torch.nn as nn
import torch.nn.functional as F

from .targets import object_center_mask

def penalty_reduced_focal_loss(
    pred: torch.Tensor, target: torch.Tensor, alpha: float = 2.0, beta: float = 4.0
) -> torch.Tensor:
    """
    CornerNet/CenterNet penalty-reduced focal loss for Gaussian-splatted
    heatmap targets:

        L = -1/N * sum[ positives:  (1 - p)^alpha * log(p)
                        negatives:  (1 - y)^beta * p^alpha * log(1 - p) ]

    where "positives" are the exact object centers (y == 1) and N is the
    number of objects. The (1 - y)^beta factor down-weights cells near a
    center, which is what makes the soft Gaussian target sensible rather
    than contradictory.

    This replaces an MSE on the heatmap. Against a target that is >99.99%
    zero, MSE's optimum is a constant near the base rate, and that is
    exactly what the previous checkpoints learned -- a heatmap with std
    ~1e-3 that could not separate an object cell from an empty one.
    """
    pred = torch.clamp(pred, 1e-7, 1.0 - 1e-7)
    positives = target.eq(1.0).float()
    negatives = 1.0 - positives

    positive_loss = torch.log(pred) * torch.pow(1 - pred, alpha) * positives
    negative_loss = (
        torch.log(1 - pred)
        * torch.pow(pred, alpha)
        * torch.pow(1 - target, beta)
        * negatives
    )

    n_positive = positives.sum()
    total = -(positive_loss.sum() + negative_loss.sum())
    # With no objects in the batch only the negative term is defined.
    return total / n_positive if n_positive > 0 else -negative_loss.sum()


def masked_l1_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    L1 restricted to object cells, normalized by the number of supervised
    values rather than by the whole grid.

    An unmasked L1 over a grid whose target is almost entirely zero is
    minimized by predicting zero everywhere; that is why the previous box
    head collapsed to outputs of magnitude ~1e-3. `mask` is (B, 1, H, W)
    and broadcasts across the prediction's channels.
    """
    expanded = mask.expand_as(pred)
    n_supervised = expanded.sum()
    if n_supervised == 0:
        return pred.sum() * 0.0
    return (torch.abs(pred - target) * expanded).sum() / n_supervised


def compute_det_loss(preds: dict, targets: dict, object_mask: torch.Tensor = None) -> torch.Tensor:
    """
    Detection loss: penalty-reduced focal loss on the Gaussian-splatted
    heatmap, plus L1 on box regression and velocity masked to object cells.

    Args:
        preds: model outputs with 'H', 'B', 'V'.
        targets: matching ground-truth tensors.
        object_mask: optional (B, 1, H, W) mask of supervised cells. When
            omitted it is derived from the target heatmap, so existing
            callers need no change.
    """
    if object_mask is None:
        object_mask = object_center_mask(targets['H'])

    l_h = penalty_reduced_focal_loss(preds['H'], targets['H'])
    l_b = masked_l1_loss(preds['B'], targets['B'], object_mask)
    l_v = masked_l1_loss(preds['V'], targets['V'], object_mask)

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
