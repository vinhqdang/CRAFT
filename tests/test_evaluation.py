import torch
import pytest
from craf_x.evaluation.metrics import (
    calculate_consistency_auc,
    calculate_attribution_fidelity,
    mock_map_nds_score,
    calculate_asr
)

def test_consistency_auc():
    s_map = torch.tensor([[[[0.1, 0.9], [0.8, 0.2]]]])
    attack_mask = torch.tensor([[[[1.0, 0.0], [0.0, 1.0]]]]) # attack means low S
    # attack=1 means attacked. We expect S to be low when attacked.
    
    auc = calculate_consistency_auc(s_map, attack_mask)
    assert 0.0 <= auc <= 1.0
    # Actually, a perfect attack detector would give AUC > 0.9
    assert auc > 0.9 # the values perfectly separate (clean: 0.9, 0.8 / attacked: 0.1, 0.2)

def test_attribution_fidelity():
    a_map = torch.tensor([[[[0.9, 0.1], [0.5, 0.5]]]]) # cam attribution
    ablation_diff = torch.tensor([[[[0.8, 0.2], [0.4, 0.4]]]]) # ground truth cam diff
    
    af = calculate_attribution_fidelity(a_map, ablation_diff)
    assert -1.0 <= af <= 1.0
    assert af > 0.8 # high correlation

def test_mock_map_nds_score():
    preds = {
        'H': torch.randn(2, 10, 32, 32),
        'B': torch.randn(2, 6, 32, 32),
        'V': torch.randn(2, 2, 32, 32)
    }
    targets = {
        'H': preds['H'],
        'B': preds['B'],
        'V': preds['V']
    }
    mAP, nds = mock_map_nds_score(preds, targets)
    assert mAP == 1.0 # perfect match
    assert nds == 1.0

def test_calculate_asr():
    # clean: 1.0, adv: 0.5
    asr = calculate_asr(1.0, 0.5)
    assert asr == 0.5 
