import torch
import torch.nn as nn
from typing import Dict, Tuple

from ..config import CRAFXConfig
from ..utils.losses import compute_det_loss, compute_ccp_loss, compute_mar_loss

def pgd_attack(
    model: nn.Module, 
    f_cam: torch.Tensor, 
    f_lid: torch.Tensor, 
    targets: dict,
    attack_modality: str = 'cam',
    epsilon: float = 4/255.0,
    k: int = 7
) -> torch.Tensor:
    """
    Performs PGD attack on extracted features.
    attack_modality: 'cam' or 'lid'
    """
    f_target = f_cam if attack_modality == 'cam' else f_lid
    f_other = f_lid if attack_modality == 'cam' else f_cam
    
    f_target = f_target.detach()
    f_other = f_other.detach()
    
    delta = torch.zeros_like(f_target, requires_grad=True)
    
    alpha = epsilon / k
    
    # To attack, we only need CCP -> GAFM -> Head
    ccp = model.ccp
    gafm = model.gafm
    head = model.head
    
    for _ in range(k):
        delta.requires_grad_()
        
        if attack_modality == 'cam':
            f_cam_adv = f_target + delta
            f_lid_adv = f_other
        else:
            f_cam_adv = f_other
            f_lid_adv = f_target + delta
            
        s, a = ccp(f_cam_adv, f_lid_adv)
        f_fused = gafm(f_cam_adv, f_lid_adv, a, s)
        h, b, v = head(f_fused)
        
        preds = {'H': h, 'B': b, 'V': v}
        loss = compute_det_loss(preds, targets)
        
        loss.backward()
        
        with torch.no_grad():
            delta_grad = delta.grad.sign()
            delta = delta + alpha * delta_grad
            delta = torch.clamp(delta, -epsilon, epsilon)
            delta.grad = None
            
    return delta.detach()

def compute_kl_divergence(a_clean: torch.Tensor, a_adv: torch.Tensor) -> torch.Tensor:
    """
    Computes KL divergence between clean attribution and perturbed attribution.
    To ensure safe KL divergence, we clamp values.
    """
    a_clean = torch.clamp(a_clean, 1e-7, 1.0)
    a_adv = torch.clamp(a_adv, 1e-7, 1.0)
    
    # KL(P || Q) = sum(P * log(P / Q))
    kl = torch.sum(a_clean * torch.log(a_clean / a_adv))
    return kl / a_clean.numel()

def act_training_step(
    model: nn.Module, 
    image: torch.Tensor, 
    pointcloud: torch.Tensor, 
    targets: dict, 
    m: torch.Tensor,
    config: CRAFXConfig
) -> Tuple[torch.Tensor, dict]:
    """
    Performs one step of Adversarial Consistency Training (ACT).
    """
    model.train()
    
    # ── CLEAN FORWARD PASS ──
    f_cam = model.cam_enc(image)
    f_lid = model.lid_enc(pointcloud)
    
    s_clean, a_clean = model.ccp(f_cam, f_lid)
    f_fused_clean = model.gafm(f_cam, f_lid, a_clean, s_clean)
    h, b, v = model.head(f_fused_clean)
    
    preds_clean = {'H': h, 'B': b, 'V': v}
    
    l_det_clean = compute_det_loss(preds_clean, targets)
    l_ccp = compute_ccp_loss(s_clean, m)
    
    # ── ADVERSARIAL AUGMENTATION — ATTACK CAMERA ──
    delta_cam = pgd_attack(model, f_cam, f_lid, targets, 'cam', config.epsilon_cam, config.pgd_k)
    f_cam_adv = f_cam + delta_cam
    s_adv_cam, a_adv_cam = model.ccp(f_cam_adv, f_lid)
    f_fused_adv_cam = model.gafm(f_cam_adv, f_lid, a_adv_cam, s_adv_cam)
    h_c, b_c, v_c = model.head(f_fused_adv_cam)
    l_det_adv_cam = compute_det_loss({'H': h_c, 'B': b_c, 'V': v_c}, targets)
    
    # ── ADVERSARIAL AUGMENTATION — ATTACK LIDAR ──
    delta_lid = pgd_attack(model, f_cam, f_lid, targets, 'lid', config.epsilon_lid, config.pgd_k)
    f_lid_adv = f_lid + delta_lid
    s_adv_lid, a_adv_lid = model.ccp(f_cam, f_lid_adv)
    f_fused_adv_lid = model.gafm(f_cam, f_lid_adv, a_adv_lid, s_adv_lid)
    h_l, b_l, v_l = model.head(f_fused_adv_lid)
    l_det_adv_lid = compute_det_loss({'H': h_l, 'B': b_l, 'V': v_l}, targets)
    
    # ── ACT LOSS ──
    kl_cam = compute_kl_divergence(a_clean, a_adv_cam)
    kl_lid = compute_kl_divergence(a_clean, a_adv_lid)
    l_act = l_det_adv_cam + l_det_adv_lid + config.gamma * (kl_cam + kl_lid)
    
    # ── ATTRIBUTION REGULARIZATION ──
    l_mar = compute_mar_loss(a_clean, config.mu)
    
    # ── TOTAL LOSS ──
    loss = l_det_clean + config.lambda_1 * l_ccp + config.lambda_2 * l_act + config.lambda_3 * l_mar
    
    metrics = {
        'loss': loss.item(),
        'l_det_clean': l_det_clean.item(),
        'l_ccp': l_ccp.item(),
        'l_act': l_act.item(),
        'l_mar': l_mar.item()
    }
    
    return loss, metrics
