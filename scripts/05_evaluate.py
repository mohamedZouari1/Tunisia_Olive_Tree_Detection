"""
ZitounEye — Step 5: Evaluate Model
====================================
Detailed evaluation with confusion matrices, per-class metrics,
hard-negative analysis, and visual prediction samples.

Usage:
    python scripts/05_evaluate.py --checkpoint checkpoints/tier1_best.pth
    python scripts/05_evaluate.py --checkpoint checkpoints/tier1_best.pth --split test
"""

import os
import sys
import argparse
import warnings
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

import segmentation_models_pytorch as smp

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.utils import load_config, ensure_dirs, plot_prediction, get_class_names
from src.dataset import OliveGroveDataset, get_val_augmentation
from src.metrics import SegmentationMetrics
from src.losses import DiceFocalLoss

warnings.filterwarnings("ignore")


def plot_confusion_matrix(cm: np.ndarray, class_names: list, save_path: str, title: str = ""):
    """Plot detailed confusion matrix with annotations."""
    # Normalize by row (true class)
    cm_normalized = cm.astype(np.float64)
    row_sums = cm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1  # avoid division by zero
    cm_normalized = cm_normalized / row_sums
    
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    
    # Raw counts
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=class_names, yticklabels=class_names,
        ax=axes[0], cbar_kws={"label": "Pixel Count"}
    )
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("True")
    axes[0].set_title("Confusion Matrix (Raw Counts)")
    
    # Normalized (percentage)
    sns.heatmap(
        cm_normalized, annot=True, fmt=".2%", cmap="YlOrRd",
        xticklabels=class_names, yticklabels=class_names,
        ax=axes[1], cbar_kws={"label": "Proportion"}
    )
    axes[1].set_xlabel("Predicted")
    axes[1].set_ylabel("True")
    axes[1].set_title("Confusion Matrix (Normalized by True Class)")
    
    if title:
        fig.suptitle(title, fontsize=14, fontweight="bold")
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


def plot_hard_negative_analysis(cm: np.ndarray, class_names: list, save_path: str):
    """
    Focus on olive ↔ vineyard ↔ other_orchard confusion.
    This is where real-world errors concentrate.
    """
    focus_classes = [1, 2, 3]  # olive, vineyard, other_orchard
    focus_names = [class_names[i] for i in focus_classes]
    
    # Extract sub-matrix
    sub_cm = cm[np.ix_(focus_classes, focus_classes)]
    
    # Normalize
    row_sums = sub_cm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    sub_cm_norm = sub_cm.astype(np.float64) / row_sums
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    sns.heatmap(
        sub_cm_norm, annot=True, fmt=".2%", cmap="Reds",
        xticklabels=focus_names, yticklabels=focus_names,
        ax=ax, vmin=0, vmax=1,
        annot_kws={"size": 14},
    )
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("True", fontsize=12)
    ax.set_title("🔍 Hard Negative Analysis\n(Olive ↔ Vineyard ↔ Other Orchard Confusion)", fontsize=13)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


def plot_per_class_iou(results: dict, class_names: list, save_path: str):
    """Bar chart of per-class IoU."""
    ious = [results["per_class_iou"][name] for name in class_names]
    
    fig, ax = plt.subplots(figsize=(10, 5))
    
    colors = ['#333333', '#228B22', '#800080', '#FFA500', '#FFD700', '#FF0000', '#808080']
    bars = ax.bar(class_names, ious, color=colors, edgecolor='black', linewidth=0.5)
    
    # Highlight olive bar
    bars[1].set_edgecolor('gold')
    bars[1].set_linewidth(3)
    
    ax.axhline(y=0.70, color='red', linestyle='--', alpha=0.7, label='Target (0.70)')
    ax.set_ylabel("IoU", fontsize=12)
    ax.set_title("Per-Class IoU", fontsize=14, fontweight="bold")
    ax.set_ylim(0, 1.0)
    ax.legend()
    
    # Add value labels
    for bar, iou in zip(bars, ious):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f"{iou:.3f}", ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.xticks(rotation=30, ha='right')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate olive grove segmentation model")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--split", default="val", choices=["val", "test"])
    parser.add_argument("--num-samples", type=int, default=20, help="Number of visual samples to save")
    args = parser.parse_args()
    
    config = load_config(args.config)
    ensure_dirs(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("=" * 60)
    print(f"ZitounEye — Model Evaluation ({args.split} set)")
    print("=" * 60)
    
    # Load checkpoint
    print(f"\nLoading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    
    # Determine model architecture from checkpoint config
    ckpt_config = checkpoint.get("config", config)
    
    # Try to detect which tier
    in_channels = None
    for tier in [1, 2]:
        tier_key = f"tier{tier}"
        if tier_key in ckpt_config.get("model", {}):
            model_cfg = ckpt_config["model"][tier_key]
            arch = model_cfg["architecture"]
            in_channels = model_cfg["in_channels"]
            
            # Try to build and load
            try:
                if arch == "Unet":
                    model = smp.Unet(
                        encoder_name=model_cfg["encoder"],
                        encoder_weights=None,  # loading from checkpoint
                        in_channels=in_channels,
                        classes=model_cfg["classes"],
                    )
                else:
                    model = smp.DeepLabV3Plus(
                        encoder_name=model_cfg["encoder"],
                        encoder_weights=None,
                        in_channels=in_channels,
                        classes=model_cfg["classes"],
                    )
                
                model.load_state_dict(checkpoint["model_state_dict"])
                print(f"  Model: {arch} (Tier {tier})")
                break
            except Exception:
                continue
    
    model = model.to(device)
    model.eval()
    
    # Load dataset
    patches_dir = config["paths"]["patches_dir"]
    stats_path = os.path.join(patches_dir, "normalize_stats.npz")
    normalize_stats = None
    if os.path.exists(stats_path):
        stats = np.load(stats_path)
        normalize_stats = {"mean": stats["mean"], "std": stats["std"]}
    
    dataset = OliveGroveDataset(
        patches_dir=patches_dir,
        split=args.split,
        transform=get_val_augmentation(config["dataset"]["patch_size"]),
        normalize_stats=normalize_stats,
    )
    
    dataloader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=4)
    
    # Evaluate
    class_names = get_class_names(config)
    metrics = SegmentationMetrics(num_classes=7, class_names=class_names)
    
    print(f"\nEvaluating on {len(dataset)} patches...")
    
    with torch.no_grad():
        for batch in dataloader:
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)
            
            outputs = model(images)
            preds = outputs.argmax(dim=1)
            
            metrics.update(preds, masks)
    
    results = metrics.compute()
    
    # Print detailed report
    metrics.print_report(results)
    
    # Save plots
    eval_dir = os.path.join(config["paths"]["predictions_dir"], f"eval_{args.split}")
    os.makedirs(eval_dir, exist_ok=True)
    
    # 1. Full confusion matrix
    plot_confusion_matrix(
        results["confusion_matrix"], class_names,
        os.path.join(eval_dir, "confusion_matrix.png"),
        title=f"ZitounEye — {args.split.upper()} Set Evaluation"
    )
    
    # 2. Hard negative analysis
    plot_hard_negative_analysis(
        results["confusion_matrix"], class_names,
        os.path.join(eval_dir, "hard_negative_analysis.png")
    )
    
    # 3. Per-class IoU bar chart
    plot_per_class_iou(
        results, class_names,
        os.path.join(eval_dir, "per_class_iou.png")
    )
    
    # 4. Visual prediction samples
    print(f"\nSaving {args.num_samples} visual samples...")
    samples_dir = os.path.join(eval_dir, "samples")
    os.makedirs(samples_dir, exist_ok=True)
    
    with torch.no_grad():
        sample_count = 0
        for batch in dataloader:
            if sample_count >= args.num_samples:
                break
            
            images = batch["image"].to(device)
            masks = batch["mask"]
            outputs = model(images)
            preds = outputs.argmax(dim=1).cpu()
            
            for j in range(min(images.shape[0], args.num_samples - sample_count)):
                save_path = os.path.join(samples_dir, f"sample_{sample_count:04d}.png")
                plot_prediction(
                    image=images[j].cpu().numpy(),
                    mask_true=masks[j].numpy(),
                    mask_pred=preds[j].numpy(),
                    config=config,
                    save_path=save_path,
                )
                sample_count += 1
    
    print(f"\n{'='*60}")
    print(f"✅ Evaluation complete!")
    print(f"   Results saved to: {eval_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
