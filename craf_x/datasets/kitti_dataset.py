import torch
import warnings
from torch.utils.data import Dataset

class CRAFXKittiDataset(Dataset):
    """
    Wrapper for the KITTI 3D Object Detection Dataset.
    Single camera and single LiDAR sweep.
    """
    def __init__(self, data_root: str, split: str = 'training'):
        self.data_root = data_root
        self.split = split
        self.sample_indices = ["000000", "000001", "000002"] # Mocked indices
        
    def __len__(self):
        return len(self.sample_indices)
        
    def __getitem__(self, idx):
        # image shapes
        image = torch.randn(3, 128, 128)
        pointcloud = torch.randn(4, 128, 128)
        m = torch.randint(0, 2, (1, 128, 128)).float()
        
        targets = {
            'H': torch.zeros(3, 128, 128), # KITTI usually 3 classes (Car, Pedestrian, Cyclist)
            'B': torch.zeros(6, 128, 128),
            'V': torch.zeros(2, 128, 128)
        }
        
        return {
            'image': image,
            'pointcloud': pointcloud,
            'm': m,
            'targets': targets,
            'idx': self.sample_indices[idx]
        }
