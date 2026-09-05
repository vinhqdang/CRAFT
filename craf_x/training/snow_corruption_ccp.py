"""
Experimental CCP-mismatch supervision using synthetic weather corruption,
instead of (or alongside) adversarial perturbation, as the "mismatch"
target for compute_ccp_loss.

Motivation (see papers/vp-nav-imavis/manuscript/5discussion.tex, "The
CCP-Covariate Negative Result, Generalized"): the existing fix in
adversarial.py supervises CCP disagreement using PGD-adversarially-
perturbed features as the mismatch proxy. That plausibly failed to
transfer to real snow because an l-infinity-bounded adversarial
perturbation is a qualitatively different feature-space disturbance than
real snowfall produces. This module tests the natural next attempt:
supervise CCP disagreement using synthetic corruption that is actually
physically modeled on what snow does to each modality -- reduced camera
contrast/visibility with additive speckle noise, and LiDAR point/pillar
dropout with additive range noise simulating spurious near-range returns
-- rather than a gradient-based attack with no physical grounding.

This is a genuinely separate, additive training path, NOT a modification
of act_training_step in adversarial.py: that function and its exact
current behavior must stay untouched, since other in-progress training
runs (see papers, the CADC evaluation) depend on it unchanged. Everything
here is new: a new corruption function pair (ported from
conformal_monitor.corruption, which this shared library must not import,
since papers/ depends on craf_x/, not the reverse) and a new training-step
function that reuses adversarial.py's existing pieces (pgd_attack,
compute_kl_divergence) via import, without editing that file.
"""
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from ..config import CRAFXConfig
from ..utils.losses import compute_ccp_loss, compute_det_loss, compute_mar_loss
from .adversarial import compute_kl_divergence, pgd_attack


def apply_camera_snow_corruption(image: torch.Tensor, severity: float, seed: Optional[int] = None) -> torch.Tensor:
    """
    Simulates snowfall's effect on camera images: reduced contrast/visibility
    (attenuation toward a flat gray) plus additive bright-speckle "snowflake"
    noise, both scaled by `severity` in [0, 1]. Ported from
    conformal_monitor.corruption.apply_camera_snow_corruption (kept
    byte-identical in logic) so this shared-library module has no
    dependency on paper-specific code.
    """
    if severity <= 0.0:
        return image
    # torch.Generator() defaults to CPU; generate on CPU (device-agnostic,
    # matches the ported original) and move to the input's device.
    generator = torch.Generator().manual_seed(seed) if seed is not None else None

    attenuated = image * (1.0 - 0.5 * severity) + 0.5 * severity
    speckle = (torch.randn(image.shape, generator=generator) * (0.3 * severity)).to(image.device)
    bright_mask = (torch.rand(image.shape, generator=generator) < (0.05 * severity)).float().to(image.device)
    corrupted = attenuated + speckle + bright_mask * severity
    return corrupted


def apply_lidar_snow_corruption(pointcloud: torch.Tensor, severity: float, seed: Optional[int] = None) -> torch.Tensor:
    """
    Simulates snowfall's effect on LiDAR returns: random point/pillar
    dropout (snow-induced beam attenuation and false near-range returns)
    plus additive range noise, both scaled by `severity` in [0, 1]. Ported
    from conformal_monitor.corruption.apply_lidar_snow_corruption (see
    module docstring for why this is a port, not an import).
    """
    if severity <= 0.0:
        return pointcloud
    generator = torch.Generator().manual_seed(seed) if seed is not None else None

    keep_mask = (
        (torch.rand(pointcloud.shape[-2:], generator=generator) >= (0.4 * severity)).float().to(pointcloud.device)
    )
    range_noise = (torch.randn(pointcloud.shape, generator=generator) * (0.2 * severity)).to(pointcloud.device)
    corrupted = (pointcloud + range_noise) * keep_mask.unsqueeze(0)
    return corrupted


def act_training_step_snow_corrupted(
    model: nn.Module,
    image: torch.Tensor,
    pointcloud: torch.Tensor,
    targets: dict,
    m: torch.Tensor,
    config: CRAFXConfig,
    snow_severity: float = 0.6,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Same overall structure as adversarial.act_training_step (clean pass +
    PGD-adversarial branches for the ACT/attribution loss, unchanged), plus
    two ADDITIONAL branches: a camera-only and a LiDAR-only synthetic-
    snow-corrupted pass, each paired with the other modality's clean
    feature (mirroring exactly how the PGD branches pair one corrupted
    modality with the other's clean features). Both are supervised via
    compute_ccp_loss with a mismatched (m=0) target, added to l_ccp
    alongside (not instead of) the existing adversarial-branch supervision,
    so this is a genuinely new, additive experiment rather than a
    modification of the existing fix's behavior.
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
    l_ccp_clean = compute_ccp_loss(s_clean, m)

    # ── ADVERSARIAL AUGMENTATION (unchanged from adversarial.py's own logic) ──
    delta_cam = pgd_attack(model, f_cam, f_lid, targets, 'cam', config.epsilon_cam, config.pgd_k)
    f_cam_adv = f_cam + delta_cam
    s_adv_cam, a_adv_cam = model.ccp(f_cam_adv, f_lid)
    f_fused_adv_cam = model.gafm(f_cam_adv, f_lid, a_adv_cam, s_adv_cam)
    h_c, b_c, v_c = model.head(f_fused_adv_cam)
    l_det_adv_cam = compute_det_loss({'H': h_c, 'B': b_c, 'V': v_c}, targets)

    delta_lid = pgd_attack(model, f_cam, f_lid, targets, 'lid', config.epsilon_lid, config.pgd_k)
    f_lid_adv = f_lid + delta_lid
    s_adv_lid, a_adv_lid = model.ccp(f_cam, f_lid_adv)
    f_fused_adv_lid = model.gafm(f_cam, f_lid_adv, a_adv_lid, s_adv_lid)
    h_l, b_l, v_l = model.head(f_fused_adv_lid)
    l_det_adv_lid = compute_det_loss({'H': h_l, 'B': b_l, 'V': v_l}, targets)

    # ── SYNTHETIC-SNOW AUGMENTATION (new: physically-motivated corruption,
    # not a gradient-based attack, as the mismatch proxy) ──
    image_snow = apply_camera_snow_corruption(image, snow_severity)
    f_cam_snow = model.cam_enc(image_snow)
    s_snow_cam, a_snow_cam = model.ccp(f_cam_snow, f_lid)
    f_fused_snow_cam = model.gafm(f_cam_snow, f_lid, a_snow_cam, s_snow_cam)
    h_sc, b_sc, v_sc = model.head(f_fused_snow_cam)
    l_det_snow_cam = compute_det_loss({'H': h_sc, 'B': b_sc, 'V': v_sc}, targets)

    pointcloud_snow = apply_lidar_snow_corruption(pointcloud, snow_severity)
    f_lid_snow = model.lid_enc(pointcloud_snow)
    s_snow_lid, a_snow_lid = model.ccp(f_cam, f_lid_snow)
    f_fused_snow_lid = model.gafm(f_cam, f_lid_snow, a_snow_lid, s_snow_lid)
    h_sl, b_sl, v_sl = model.head(f_fused_snow_lid)
    l_det_snow_lid = compute_det_loss({'H': h_sl, 'B': b_sl, 'V': v_sl}, targets)

    # ── CCP MISMATCH SUPERVISION: adversarial branches (existing fix) +
    # synthetic-snow branches (this experiment), both against m=0 ──
    zero_mask = torch.zeros_like(m)
    l_ccp_adv_cam = compute_ccp_loss(s_adv_cam, zero_mask)
    l_ccp_adv_lid = compute_ccp_loss(s_adv_lid, zero_mask)
    l_ccp_snow_cam = compute_ccp_loss(s_snow_cam, zero_mask)
    l_ccp_snow_lid = compute_ccp_loss(s_snow_lid, zero_mask)
    l_ccp = l_ccp_clean + l_ccp_adv_cam + l_ccp_adv_lid + l_ccp_snow_cam + l_ccp_snow_lid

    # ── ACT LOSS (unchanged: still driven only by the adversarial branches,
    # since the KL/attribution-consistency term's own motivation -- keeping
    # attribution stable under an adversarial attack -- is orthogonal to
    # this experiment) ──
    kl_cam = compute_kl_divergence(a_clean, a_adv_cam)
    kl_lid = compute_kl_divergence(a_clean, a_adv_lid)
    l_act = l_det_adv_cam + l_det_adv_lid + config.gamma * (kl_cam + kl_lid)

    # ── ATTRIBUTION REGULARIZATION ──
    l_mar = compute_mar_loss(a_clean, config.mu)

    # ── TOTAL LOSS ──
    # The two new snow-corrupted branches' own detection losses
    # (l_det_snow_cam, l_det_snow_lid) are deliberately not added to the
    # total loss: their purpose here is solely to produce a CCP score to
    # supervise as a mismatch example, exactly how the adversarial
    # branches' det losses feed only l_act, not l_det_clean's role.
    # Folding them into l_act as well would conflate two different
    # robustness objectives (attack-robustness vs. weather-robustness) in
    # a single ablation; kept as reported-but-unused metrics for now so a
    # future run can isolate that choice if this experiment's core result
    # (does the CCP-informed bettor improve) is positive enough to warrant it.
    loss = l_det_clean + config.lambda_1 * l_ccp + config.lambda_2 * l_act + config.lambda_3 * l_mar

    metrics = {
        'loss': loss.item(),
        'l_det_clean': l_det_clean.item(),
        'l_ccp': l_ccp.item(),
        'l_ccp_clean': l_ccp_clean.item(),
        'l_ccp_adv': (l_ccp_adv_cam + l_ccp_adv_lid).item(),
        'l_ccp_snow': (l_ccp_snow_cam + l_ccp_snow_lid).item(),
        'l_act': l_act.item(),
        'l_mar': l_mar.item(),
        'l_det_snow_cam': l_det_snow_cam.item(),
        'l_det_snow_lid': l_det_snow_lid.item(),
    }

    return loss, metrics
