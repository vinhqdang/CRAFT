import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from torch.utils.data import DataLoader

from craf_x.config import CRAFXConfig
from craf_x.models.crafx_net import CRAFX_Net
from craf_x.datasets.nuscenes_mock import NuScenesMockDataset
from craf_x.evaluation.pipeline import evaluate_crafx

def main():
    print("Initializing CRAF-X Model...")
    config = CRAFXConfig(bev_h=32, bev_w=32)
    model = CRAFX_Net(config)
    
    print("Setting up Mock Dataset DataLoader...")
    dataset = NuScenesMockDataset(num_samples=2)
    loader = DataLoader(dataset, batch_size=2, shuffle=False)
    
    print("Running Evaluation Pipeline...")
    results = evaluate_crafx(model, loader, config)
    
    print("\n=== EVALUATION RESULTS ===")
    for k, v in results.items():
        print(f"{k}: {v:.4f}")

if __name__ == "__main__":
    main()
