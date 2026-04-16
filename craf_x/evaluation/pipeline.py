import torch
from typing import Dict
from torch.utils.data import DataLoader

from craf_x.models.crafx_net import CRAFX_Net
from craf_x.config import CRAFXConfig
from craf_x.training.adversarial import pgd_attack
from craf_x.evaluation.metrics import (
    calculate_consistency_auc,
    calculate_attribution_fidelity,
    mock_map_nds_score,
    calculate_asr
)

def evaluate_crafx(model: CRAFX_Net, dataloader: DataLoader, config: CRAFXConfig) -> Dict[str, float]:
    """
    Runs the end-to-end evaluation pipeline for CRAF-X.
    Includes clean evaluation and adversarial evaluation.
    """
    model.eval()
    
    total_clean_map = 0.0
    total_adv_map = 0.0
    total_auc = 0.0
    total_af = 0.0
    
    num_batches = len(dataloader)
    
    for batch in dataloader:
        image = batch['image']
        pointcloud = batch['pointcloud']
        targets = batch['targets']
        
        # 1. Clean Pass
        with torch.no_grad():
            clean_out = model(image, pointcloud)
        
        clean_map, _ = mock_map_nds_score(
            {'H': clean_out['H'], 'B': clean_out['B'], 'V': clean_out['V']},
            targets
        )
        total_clean_map += clean_map
        
        # 2. Adversarial Pass (e.g., Attack Camera)
        f_cam = clean_out['F_cam']
        f_lid = clean_out['F_lid']
        
        # PGD attack turns on grad temporarily
        attack_mask = torch.ones_like(clean_out['S'])
        
        delta_cam = pgd_attack(model, f_cam, f_lid, targets, attack_modality='cam', epsilon=config.epsilon_cam, k=config.pgd_k)
        
        # Forward with perturbed features
        with torch.no_grad():
            f_cam_adv = f_cam + delta_cam
            s_adv, a_adv = model.ccp(f_cam_adv, f_lid)
            f_fused_adv = model.gafm(f_cam_adv, f_lid, a_adv, s_adv)
            h_adv, b_adv, v_adv = model.head(f_fused_adv)
            
        adv_map, _ = mock_map_nds_score(
            {'H': h_adv, 'B': b_adv, 'V': v_adv},
            targets
        )
        total_adv_map += adv_map
        
        # 3. Compute Probe metrics
        auc = calculate_consistency_auc(s_adv, attack_mask)
        total_auc += auc
        
        # Mock ablation diff for AF (normally requires full network rerun)
        ablation_diff = torch.zeros_like(a_adv[:, 0:1]) # mock
        af = calculate_attribution_fidelity(a_adv[:, 0:1], ablation_diff)
        total_af += af

    # Aggregate
    res = {
        'clean_mAP': total_clean_map / num_batches,
        'adv_mAP': total_adv_map / num_batches,
        'asr': calculate_asr(total_clean_map / num_batches, total_adv_map / num_batches),
        'consistency_auc': total_auc / num_batches,
        'attribution_fidelity': total_af / num_batches
    }
    
    return res
