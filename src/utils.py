"""
ZitounEye — Shared Utilities
============================
Common functions used across all pipeline stages.
"""

import os
import yaml
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def load_config(config_path: str = "configs/default.yaml") -> dict:
    """Load YAML configuration file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def ensure_dirs(config: dict) -> None:
    """Create all output directories specified in config."""
    for key, path in config["paths"].items():
        if key != "project_root" and key != "raw_shapefiles":
            os.makedirs(path, exist_ok=True)


def get_class_colors(config: dict) -> np.ndarray:
    """Return (num_classes, 3) array of RGB colors for visualization."""
    return np.array(config["classes"]["colors"], dtype=np.uint8)


def get_class_names(config: dict) -> List[str]:
    """Return list of class names."""
    return config["classes"]["names"]


def build_crop_code_mapping(config: dict) -> Dict[int, int]:
    """
    Build mapping from Spanish EuroCrop original_code → class_id.
    
    Returns dict like {101: 1, 349: 1, 102: 2, ...}
    """
    mapping = {}
    
    # Olive codes → class 1
    for code in config["classes"]["olive_codes"]:
        mapping[code] = 1
    
    # Vineyard codes → class 2
    for code in config["classes"]["vineyard_codes"]:
        mapping[code] = 2
    
    # Other orchard codes → class 3
    for code in config["classes"]["other_orchard_codes"]:
        mapping[code] = 3
    
    # Everything else will be handled by HCAT3 prefix logic in the label script
    return mapping


def colorize_mask(mask: np.ndarray, config: dict) -> np.ndarray:
    """
    Convert class-id mask (H, W) to RGB image (H, W, 3) for visualization.
    """
    colors = get_class_colors(config)
    h, w = mask.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    
    for class_id in range(len(colors)):
        rgb[mask == class_id] = colors[class_id]
    
    return rgb


def plot_prediction(
    image: np.ndarray,
    mask_true: np.ndarray,
    mask_pred: np.ndarray,
    config: dict,
    save_path: Optional[str] = None,
    title: str = ""
) -> None:
    """
    Plot side-by-side: RGB composite | ground truth | prediction.
    
    Args:
        image: (C, H, W) or (H, W, C) multi-band image — uses bands for RGB
        mask_true: (H, W) ground truth class IDs
        mask_pred: (H, W) predicted class IDs
        config: configuration dict
        save_path: optional path to save figure
        title: figure title
    """
    if image.ndim == 3 and image.shape[0] < image.shape[-1]:
        # (C, H, W) → (H, W, C)
        image = np.transpose(image, (1, 2, 0))
    
    # Create RGB composite from first 3 bands (B4, B3, B2 = Red, Green, Blue)
    # Assuming band order: B2, B3, B4, ...
    if image.shape[-1] >= 3:
        rgb = image[:, :, [2, 1, 0]]  # B4, B3, B2
    else:
        rgb = image[:, :, :3]
    
    # Normalize to [0, 1] for display
    p2, p98 = np.percentile(rgb[rgb > 0], [2, 98]) if np.any(rgb > 0) else (0, 1)
    rgb = np.clip((rgb - p2) / (p98 - p2 + 1e-8), 0, 1)
    
    gt_rgb = colorize_mask(mask_true, config)
    pred_rgb = colorize_mask(mask_pred, config)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    axes[0].imshow(rgb)
    axes[0].set_title("Sentinel-2 RGB")
    axes[0].axis("off")
    
    axes[1].imshow(gt_rgb)
    axes[1].set_title("Ground Truth")
    axes[1].axis("off")
    
    axes[2].imshow(pred_rgb)
    axes[2].set_title("Prediction")
    axes[2].axis("off")
    
    # Add legend
    class_names = get_class_names(config)
    class_colors = get_class_colors(config)
    patches = [
        mpatches.Patch(color=class_colors[i] / 255.0, label=class_names[i])
        for i in range(len(class_names))
    ]
    fig.legend(handles=patches, loc="lower center", ncol=len(class_names), fontsize=9)
    
    if title:
        fig.suptitle(title, fontsize=14, fontweight="bold")
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    
    plt.close()


def compute_channel_stats(patch_dir: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute per-channel mean and std across all training patches.
    Uses Welford's online algorithm for memory efficiency.
    
    Returns:
        (mean, std) each of shape (num_channels,)
    """
    from pathlib import Path
    
    patch_files = sorted(Path(patch_dir).glob("*_image.npy"))
    
    if len(patch_files) == 0:
        raise ValueError(f"No image patches found in {patch_dir}")
    
    # Load first to get number of channels
    sample = np.load(patch_files[0])
    n_channels = sample.shape[0]
    
    # Welford's online algorithm
    count = 0
    mean = np.zeros(n_channels, dtype=np.float64)
    m2 = np.zeros(n_channels, dtype=np.float64)
    
    for pf in patch_files:
        data = np.load(pf).astype(np.float64)  # (C, H, W)
        
        # Flatten spatial dims
        pixels = data.reshape(n_channels, -1)  # (C, H*W)
        
        for i in range(pixels.shape[1]):
            count += 1
            delta = pixels[:, i] - mean
            mean += delta / count
            delta2 = pixels[:, i] - mean
            m2 += delta * delta2
    
    variance = m2 / count
    std = np.sqrt(variance)
    
    return mean.astype(np.float32), std.astype(np.float32)
