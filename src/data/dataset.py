import os
import numpy as np
import torch
from torch.utils.data import Dataset

class S3DISDataset(Dataset):
    """S3DIS dataset for semantic segmentation"""
    def __init__(self, data_dir, num_points=4096, split='train', test_area=5):
        self.data_dir = data_dir
        self.num_points = num_points
        self.split = split
        self.test_area = test_area

        # Get all processed files
        self.data_list = []
        point_files = [f for f in os.listdir(data_dir) if f.endswith('_point.npy')]

        for file in point_files:
            area_num = int(file.split('_')[1])  # Extract area number
            if split == 'train' and area_num != test_area:
                self.data_list.append(file.replace('_point.npy', ''))
            elif split == 'test' and area_num == test_area:
                self.data_list.append(file.replace('_point.npy', ''))

        print(f"{split.upper()} set: {len(self.data_list)} samples")

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        base_name = self.data_list[idx]

        # Load data
        points = np.load(os.path.join(self.data_dir, f'{base_name}_point.npy'))
        labels = np.load(os.path.join(self.data_dir, f'{base_name}_sem_label.npy'))

        # Extract coordinates only (xyz)
        coords = points[:, :3].astype(np.float32)

        # Randomly sample points
        if len(coords) >= self.num_points:
            choice = np.random.choice(len(coords), self.num_points, replace=False)
        else:
            choice = np.random.choice(len(coords), self.num_points, replace=True)

        coords = coords[choice]
        labels = labels[choice]

        # Normalize coordinates
        coords = self.normalize_coords(coords)

        return torch.FloatTensor(coords), torch.LongTensor(labels)

    def normalize_coords(self, coords):
        """Normalize coordinates to unit sphere"""
        centroid = np.mean(coords, axis=0)
        coords = coords - centroid
        m = np.max(np.sqrt(np.sum(coords**2, axis=1)))
        coords = coords / m
        return coords