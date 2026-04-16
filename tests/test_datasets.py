import torch
import pytest
from torch.utils.data import DataLoader
from craf_x.datasets.nuscenes_mock import NuScenesMockDataset

def test_nuscenes_mock_dataset():
    dataset = NuScenesMockDataset(num_samples=10)
    assert len(dataset) == 10
    
    sample = dataset[0]
    assert 'image' in sample
    assert 'pointcloud' in sample
    assert 'targets' in sample
    assert 'm' in sample
    
    # Check shapes
    assert sample['image'].shape == (3, 32, 32)
    assert sample['pointcloud'].shape == (4, 32, 32)
    assert sample['m'].shape == (1, 32, 32)
    assert 'H' in sample['targets']
    assert sample['targets']['H'].shape == (10, 32, 32)

def test_dataloader_batching():
    dataset = NuScenesMockDataset(num_samples=10)
    loader = DataLoader(dataset, batch_size=2, shuffle=False)
    
    batch = next(iter(loader))
    assert batch['image'].shape == (2, 3, 32, 32)
    assert batch['m'].shape == (2, 1, 32, 32)
    assert batch['targets']['H'].shape == (2, 10, 32, 32)
