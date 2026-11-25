"""
ZitounEye — PyTorch Dataset
=============================
Memory-efficient dataset for multi-temporal Sentinel-2 patches
with class-balanced sampling and heavy augmentation.
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, WeightedRandomSampler
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import albumentations as A


class OliveGroveDataset(Dataset):
    """
    PyTorch Dataset for olive grove segmentation.
    
    Loads pre-computed .npy patches (image + mask pairs) from disk.
    Applies augmentation pipeline via albumentations.
    
    Directory structure expected:
        patches_dir/
            train/
                tile_0000_image.npy  (C, H, W) float32
                tile_0000_mask.npy   (H, W) uint8
                ...
            val/
                ...
            test/
                ...
    """
    
    def __init__(
        self,
        patches_dir: str,
        split: str = "train",
        transform: Optional[A.Compose] = None,
        normalize_stats: Optional[Dict[str, np.ndarray]] = None,
    ):
        """
        Args:
            patches_dir: root directory containing train/val/test subdirs
            split: one of "train", "val", "test"
            transform: albumentations augmentation pipeline
            normalize_stats: dict with "mean" and "std" arrays, shape (C,)
        """
        self.split_dir = Path(patches_dir) / split
        self.transform = transform
        self.normalize_stats = normalize_stats
        
        # Discover all image/mask pairs
        self.image_files = sorted(self.split_dir.glob("*_image.npy"))
        self.mask_files = [
            str(f).replace("_image.npy", "_mask.npy") for f in self.image_files
        ]
        
        if len(self.image_files) == 0:
            raise ValueError(f"No patches found in {self.split_dir}")
        
        print(f"  [{split}] Found {len(self.image_files)} patches in {self.split_dir}")
    
    def __len__(self) -> int:
        return len(self.image_files)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # Load from disk
        image = np.load(self.image_files[idx]).astype(np.float32)  # (C, H, W)
        mask = np.load(self.mask_files[idx]).astype(np.int64)      # (H, W)
        
        # Normalize per channel
        if self.normalize_stats is not None:
            mean = self.normalize_stats["mean"][:, None, None]  # (C, 1, 1)
            std = self.normalize_stats["std"][:, None, None]
            image = (image - mean) / (std + 1e-8)
        
        # Albumentations expects (H, W, C)
        if self.transform is not None:
            image_hwc = np.transpose(image, (1, 2, 0))  # (H, W, C)
            
            augmented = self.transform(image=image_hwc, mask=mask)
            image_hwc = augmented["image"]
            mask = augmented["mask"]
            
            image = np.transpose(image_hwc, (2, 0, 1))  # back to (C, H, W)
        
        return {
            "image": torch.from_numpy(image.copy()),
            "mask": torch.from_numpy(mask.copy()),
        }
    
    def get_class_distribution(self) -> np.ndarray:
        """
        Compute class pixel distribution across all patches.
        Returns array of shape (num_classes,) with pixel counts.
        """
        # Sample a subset for efficiency
        n_sample = min(100, len(self))
        indices = np.random.choice(len(self), n_sample, replace=False)
        
        all_counts = []
        for idx in indices:
            mask = np.load(self.mask_files[idx])
            counts = np.bincount(mask.flatten(), minlength=7)
            all_counts.append(counts)
        
        total = np.sum(all_counts, axis=0)
        return total
    
    def get_balanced_sampler(self, num_classes: int = 7) -> WeightedRandomSampler:
        """
        Create a WeightedRandomSampler that oversamples patches 
        containing minority classes (olive, building, road).
        
        Each patch gets a weight proportional to the rarest class it contains.
        """
        weights = []
        
        # Class importance weights (olive = highest priority)
        class_importance = np.array([0.1, 5.0, 2.0, 2.0, 0.3, 3.0, 3.0])
        
        for mask_file in self.mask_files:
            mask = np.load(mask_file)
            classes_present = np.unique(mask)
            
            # Weight = max importance of any class present
            weight = max(class_importance[c] for c in classes_present if c < len(class_importance))
            weights.append(weight)
        
        weights = torch.DoubleTensor(weights)
        sampler = WeightedRandomSampler(weights, len(weights), replacement=True)
        
        return sampler


def get_train_augmentation(patch_size: int = 256) -> A.Compose:
    """
    Heavy augmentation pipeline for training.
    
    Designed for satellite imagery:
    - No color jittering that would break spectral signatures
    - Geometric transforms are fine (rotation, flip)
    - Elastic/grid distortion simulates terrain warping
    """
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.Transpose(p=0.3),
        
        # Brightness/contrast — mild, preserves spectral ratios
        A.RandomBrightnessContrast(
            brightness_limit=0.1,
            contrast_limit=0.1,
            p=0.3,
        ),
        
        # Geometric distortions
        A.ElasticTransform(
            alpha=50,
            sigma=10,
            p=0.2,
        ),
        A.GridDistortion(
            num_steps=5,
            distort_limit=0.1,
            p=0.2,
        ),
        
        # Random crop (if patches are larger than needed)
        A.RandomCrop(height=patch_size, width=patch_size, p=1.0)
        if patch_size < 256 else A.NoOp(),
        
        # Gaussian noise — simulates sensor noise
        A.GaussNoise(
            var_limit=(0.0, 0.001),
            p=0.2,
        ),
    ])


def get_val_augmentation(patch_size: int = 256) -> A.Compose:
    """No augmentation for validation — just ensure correct size."""
    return A.Compose([
        A.CenterCrop(height=patch_size, width=patch_size, p=1.0)
        if patch_size < 256 else A.NoOp(),
    ])
