import torch
from torch.utils.data import Dataset

class NuScenesMockDataset(Dataset):
    """
    Mock Dataset returning tensors shaped like nuScenes data
    to validate the evaluation loops and model.
    """
    def __init__(self, num_samples: int = 100):
        self.num_samples = num_samples
        
    def __len__(self):
        return self.num_samples
        
    def __getitem__(self, idx):
        # image: (C, H, W). Mocking a 3-channel image mapped to BEV space for testing.
        image = torch.randn(3, 32, 32)
        # pointcloud: (C, H, W).
        pointcloud = torch.randn(4, 32, 32)
        # m: binary mask of matched cells
        m = torch.randint(0, 2, (1, 32, 32)).float()
        
        # mock targets
        targets = {
            'H': torch.zeros(10, 32, 32),
            'B': torch.zeros(6, 32, 32),
            'V': torch.zeros(2, 32, 32)
        }
        
        # Add a mock object (class 0, center at 16,16)
        targets['H'][0, 16, 16] = 1.0
        
        return {
            'image': image,
            'pointcloud': pointcloud,
            'm': m,
            'targets': targets
        }
