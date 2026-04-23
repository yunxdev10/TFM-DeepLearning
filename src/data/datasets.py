import torch
from torch.utils.data import Dataset, WeightedRandomSampler
from PIL import Image
import pandas as pd
from pathlib import Path
from typing import Callable, Optional, List, Tuple
import numpy as np

class CXRDataset(Dataset):
    """
    Dataset for Kaggle COVID-19 Radiography Database.
    Assumes a dataframe with 'image_path' and 'label' columns.
    """
    def __init__(self, df: pd.DataFrame, transform: Optional[Callable] = None):
        self.df = df
        self.transform = transform
        
        # Mapping labels to indices (example depending on actual data)
        self.label_map = {name: idx for idx, name in enumerate(sorted(self.df['label'].unique()))}
        
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = row['image_path']
        label_str = row['label']
        
        # Convert grayscale or RGBA to RGB
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        label = torch.tensor(self.label_map[label_str], dtype=torch.long)
        return image, label

class CTDataset(Dataset):
    """
    Dataset for MosMedData CT Scans.
    Assumes a dataframe with 'image_path' and 'label' columns.
    """
    def __init__(self, df: pd.DataFrame, transform: Optional[Callable] = None):
        self.df = df
        self.transform = transform
        
        self.label_map = {name: idx for idx, name in enumerate(sorted(self.df['label'].unique()))}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = row['image_path']
        label_str = row['label']
        
        # Keep as Grayscale (1 channel) for CT
        image = Image.open(img_path).convert('L')
        
        if self.transform:
            image = self.transform(image)
            
        label = torch.tensor(self.label_map[label_str], dtype=torch.long)
        return image, label

def get_balanced_sampler(labels: List[int]) -> WeightedRandomSampler:
    """
    Creates a WeightedRandomSampler to handle class imbalance.
    """
    class_counts = np.bincount(labels)
    class_weights = 1. / class_counts
    sample_weights = np.array([class_weights[t] for t in labels])
    
    sampler = WeightedRandomSampler(
        weights=torch.from_numpy(sample_weights).double(),
        num_samples=len(sample_weights),
        replacement=True
    )
    return sampler
