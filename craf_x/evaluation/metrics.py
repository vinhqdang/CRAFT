import torch
import numpy as np
from sklearn.metrics import roc_auc_score
from scipy.stats import pearsonr

def calculate_consistency_auc(s_map: torch.Tensor, attack_mask: torch.Tensor) -> float:
    """
    AUC of S treating it as a binary classifier for 'cell is clean'.
    attack_mask=1 implies attacked, so ground_truth='clean' is 1 - attack_mask.
    """
    s_flat = s_map.detach().cpu().numpy().flatten()
    clean_target = 1.0 - attack_mask.detach().cpu().numpy().flatten()
    
    try:
        auc = roc_auc_score(clean_target, s_flat)
        return float(auc)
    except ValueError:
        return 0.5

def calculate_attribution_fidelity(a_map: torch.Tensor, ablation_diff_map: torch.Tensor) -> float:
    """
    Pearson correlation between A(cam) and ground truth camera diff.
    """
    a_flat = a_map.detach().cpu().numpy().flatten()
    diff_flat = ablation_diff_map.detach().cpu().numpy().flatten()
    
    # Check for constant arrays that would break pearsonr
    if np.all(a_flat == a_flat[0]) or np.all(diff_flat == diff_flat[0]):
        return 0.0
        
    r, _ = pearsonr(a_flat, diff_flat)
    return float(r)

def mock_map_nds_score(preds: dict, targets: dict):
    """
    Mock mAP and NDS by comparing H tensors with MSE and translating to [0,1] range.
    """
    diff_h = torch.mean(torch.abs(preds['H'] - targets['H'])).item()
    diff_b = torch.mean(torch.abs(preds['B'] - targets['B'])).item()
    
    map_score = max(0.0, 1.0 - diff_h)
    nds_score = max(0.0, 1.0 - (diff_h + diff_b) / 2.0)
    
    return map_score, nds_score

def calculate_asr(clean_map: float, adv_map: float) -> float:
    """
    Attack Success Rate representation purely based on mAP degradation.
    """
    return clean_map - adv_map
