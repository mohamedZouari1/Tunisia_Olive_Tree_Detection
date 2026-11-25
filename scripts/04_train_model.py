"""
ZitounEye — Step 4: Train Segmentation Model
==============================================
Trains a U-Net or DeepLabV3+ for multi-class olive grove segmentation.
Uses segmentation_models_pytorch, mixed precision, and class-balanced loss.

Usage:
    python scripts/04_train_model.py
    python scripts/04_train_model.py --tier 1                    # U-Net (default)
    python scripts/04_train_model.py --tier 2                    # DeepLabV3+
    python scripts/04_train_model.py --epochs 2 --debug          # Quick smoke test
    python scripts/04_train_model.py --resume checkpoints/best.pth
"""

import os
import sys
import argparse
import time
import json
import warnings
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from pathlib import Path
from datetime import datetime

import segmentation_models_pytorch as smp

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.utils import load_config, ensure_dirs, plot_prediction
from src.dataset import OliveGroveDataset, get_train_augmentation, get_val_augmentation
from src.losses import DiceFocalLoss, UltimateLoss
from src.metrics import SegmentationMetrics

warnings.filterwarnings("ignore")


def build_model(config: dict, tier: int = 1) -> nn.Module:
    """
    Build segmentation model using segmentation_models_pytorch.
    
    Tier 1: U-Net with EfficientNet-B4 encoder
        - Strong baseline, fast training
        - Skip connections preserve spatial detail
        
    Tier 2: DeepLabV3+ with EfficientNet-B4 encoder
        - Atrous Spatial Pyramid Pooling captures multi-scale context
        - Better for mixed landscapes with varied parcel sizes
    """
    tier_key = f"tier{tier}"
    model_config = config["model"][tier_key]
    
    arch = model_config["architecture"]
    
    if arch == "Unet":
        model = smp.Unet(
            encoder_name=model_config["encoder"],
            encoder_weights=model_config["encoder_weights"],
            in_channels=model_config["in_channels"],
            classes=model_config["classes"],
            activation=model_config["activation"],
        )
    elif arch == "DeepLabV3Plus":
        model = smp.DeepLabV3Plus(
            encoder_name=model_config["encoder"],
            encoder_weights=model_config["encoder_weights"],
            in_channels=model_config["in_channels"],
            classes=model_config["classes"],
            activation=model_config["activation"],
        )
    else:
        raise ValueError(f"Unknown architecture: {arch}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\n  Model: {arch} + {model_config['encoder']}")
    print(f"  Input channels: {model_config['in_channels']}")
    print(f"  Output classes: {model_config['classes']}")
    print(f"  Total params: {total_params:,}")
    print(f"  Trainable params: {trainable_params:,}")
    
    return model


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    device: torch.device,
    use_amp: bool = True,
) -> dict:
    """
    Train for one epoch with mixed precision support.
    
    Returns dict with loss and metrics.
    """
    model.train()
    
    total_loss = 0.0
    num_batches = 0
    metrics = SegmentationMetrics(num_classes=7)
    
    for batch in dataloader:
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)
        
        optimizer.zero_grad()
        
        if use_amp:
            with autocast():
                outputs = model(images)
                loss = criterion(outputs, masks)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
        
        # Update metrics
        preds = outputs.argmax(dim=1)
        metrics.update(preds, masks)
    
    results = metrics.compute()
    results["loss"] = total_loss / max(num_batches, 1)
    
    return results


@torch.no_grad()
def validate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    use_amp: bool = True,
) -> dict:
    """
    Validate model without gradient computation.
    """
    model.eval()
    
    total_loss = 0.0
    num_batches = 0
    metrics = SegmentationMetrics(num_classes=7)
    
    for batch in dataloader:
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)
        
        if use_amp:
            with autocast():
                outputs = model(images)
                loss = criterion(outputs, masks)
        else:
            outputs = model(images)
            loss = criterion(outputs, masks)
        
        total_loss += loss.item()
        num_batches += 1
        
        preds = outputs.argmax(dim=1)
        metrics.update(preds, masks)
    
    results = metrics.compute()
    results["loss"] = total_loss / max(num_batches, 1)
    
    return results


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    epoch: int,
    results: dict,
    path: str,
    config: dict,
):
    """Save model checkpoint with training state."""
    # Remove non-serializable items
    save_results = {k: v for k, v in results.items() if k != "confusion_matrix"}
    
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "results": save_results,
        "config": config,
    }
    
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(checkpoint, path)


def main():
    parser = argparse.ArgumentParser(
        description="Train olive grove segmentation model"
    )
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--tier", type=int, default=1, choices=[1, 2],
                        help="Model tier: 1=U-Net, 2=DeepLabV3+")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override number of epochs")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Override batch size")
    parser.add_argument("--lr", type=float, default=None,
                        help="Override learning rate")
    parser.add_argument("--loss", choices=["dice_focal", "ultimate"], default=None,
                        help="Loss function: dice_focal (default) or ultimate (best quality)")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from")
    parser.add_argument("--debug", action="store_true",
                        help="Debug mode: reduced dataset, verbose output")
    args = parser.parse_args()
    
    config = load_config(args.config)
    ensure_dirs(config)
    
    # Override config with CLI args
    train_config = config["training"]
    epochs = args.epochs or train_config["epochs"]
    batch_size = args.batch_size or train_config["batch_size"]
    lr = args.lr or train_config["learning_rate"]
    use_amp = train_config["amp"]
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("=" * 60)
    print(f"ZitounEye — Model Training (Tier {args.tier})")
    print("=" * 60)
    print(f"  Device: {device}")
    print(f"  Epochs: {epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Learning rate: {lr}")
    print(f"  Mixed precision: {use_amp}")
    
    # ─── Dataset ─────────────────────────────────────────────────────
    patches_dir = config["paths"]["patches_dir"]
    
    # Load normalization stats
    stats_path = os.path.join(patches_dir, "normalize_stats.npz")
    normalize_stats = None
    if os.path.exists(stats_path):
        stats = np.load(stats_path)
        normalize_stats = {"mean": stats["mean"], "std": stats["std"]}
        print(f"\n  Normalization: {len(stats['mean'])} channels")
    else:
        print(f"\n  ⚠️  No normalization stats found at {stats_path}")
        print(f"      Training without normalization")
    
    # Adjust in_channels based on actual data
    sample_files = list(Path(os.path.join(patches_dir, "train")).glob("*_image.npy"))
    if sample_files:
        sample = np.load(sample_files[0])
        actual_channels = sample.shape[0]
        print(f"  Detected {actual_channels} input channels from data")
        
        # Update config
        tier_key = f"tier{args.tier}"
        config["model"][tier_key]["in_channels"] = actual_channels
    
    # Create datasets
    print(f"\n  Loading datasets...")
    patch_size = config["dataset"]["patch_size"]
    
    train_dataset = OliveGroveDataset(
        patches_dir=patches_dir,
        split="train",
        transform=get_train_augmentation(patch_size),
        normalize_stats=normalize_stats,
    )
    
    val_dataset = OliveGroveDataset(
        patches_dir=patches_dir,
        split="val",
        transform=get_val_augmentation(patch_size),
        normalize_stats=normalize_stats,
    )
    
    # Balanced sampler — oversamples olive-containing patches
    train_sampler = train_dataset.get_balanced_sampler(
        num_classes=config["classes"]["num_classes"]
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=train_config["num_workers"],
        pin_memory=True,
        drop_last=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=train_config["num_workers"],
        pin_memory=True,
    )
    
    # ─── Model ─────────────────────────────────────────────────────
    model = build_model(config, tier=args.tier)
    model = model.to(device)
    
    # ─── Loss ──────────────────────────────────────────────────────
    loss_config = train_config["loss"]
    loss_type = args.loss or loss_config.get("type", "dice_focal")
    
    if loss_type == "ultimate":
        print(f"  Loss: 🏆 UltimateLoss (Dice + Focal + Lovász + Boundary)")
        criterion = UltimateLoss(
            num_classes=config["classes"]["num_classes"],
            class_weights=loss_config["class_weights"],
            focal_gamma=loss_config["focal_gamma"],
        )
    else:
        print(f"  Loss: DiceFocalLoss")
        criterion = DiceFocalLoss(
            num_classes=config["classes"]["num_classes"],
            dice_weight=loss_config["dice_weight"],
            focal_weight=loss_config["focal_weight"],
            focal_gamma=loss_config["focal_gamma"],
            class_weights=loss_config["class_weights"],
        )
    
    # ─── Optimizer ─────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=train_config["weight_decay"],
    )
    
    # ─── Scheduler ─────────────────────────────────────────────────
    if train_config["scheduler"] == "onecycle":
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=lr,
            epochs=epochs,
            steps_per_epoch=len(train_loader),
            pct_start=0.1,
        )
    elif train_config["scheduler"] == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs
        )
    else:
        scheduler = None
    
    # ─── AMP Scaler ────────────────────────────────────────────────
    scaler = GradScaler() if use_amp else None
    
    # ─── Resume ────────────────────────────────────────────────────
    start_epoch = 0
    best_miou = 0.0
    
    if args.resume:
        print(f"\n  Resuming from: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if checkpoint["scheduler_state_dict"] and scheduler:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        best_miou = checkpoint["results"].get("mean_iou", 0.0)
        print(f"  Resumed from epoch {start_epoch}, best mIoU: {best_miou:.4f}")
    
    # ─── Training loop ─────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("TRAINING")
    print(f"{'='*60}\n")
    
    checkpoint_dir = config["paths"]["checkpoints_dir"]
    log_dir = config["paths"]["logs_dir"]
    
    tier_name = f"tier{args.tier}"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{tier_name}_{timestamp}"
    
    history = []
    patience_counter = 0
    patience = train_config["early_stopping"]["patience"]
    
    for epoch in range(start_epoch, epochs):
        epoch_start = time.time()
        
        # Train
        train_results = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device, use_amp
        )
        
        # Validate
        val_results = validate(
            model, val_loader, criterion, device, use_amp
        )
        
        # Update scheduler (epoch-level schedulers)
        if scheduler and train_config["scheduler"] != "onecycle":
            scheduler.step()
        
        epoch_time = time.time() - epoch_start
        
        # Get current LR
        current_lr = optimizer.param_groups[0]["lr"]
        
        # Log
        olive_iou = val_results["per_class_iou"].get("olive_grove", 0.0)
        vineyard_iou = val_results["per_class_iou"].get("vineyard", 0.0)
        
        print(
            f"Epoch {epoch+1:>3}/{epochs} | "
            f"Train Loss: {train_results['loss']:.4f} | "
            f"Val Loss: {val_results['loss']:.4f} | "
            f"Val mIoU: {val_results['mean_iou']:.4f} | "
            f"🫒 Olive IoU: {olive_iou:.4f} | "
            f"🍇 Vine IoU: {vineyard_iou:.4f} | "
            f"LR: {current_lr:.2e} | "
            f"Time: {epoch_time:.0f}s"
        )
        
        # History
        history.append({
            "epoch": epoch + 1,
            "train_loss": train_results["loss"],
            "val_loss": val_results["loss"],
            "val_miou": val_results["mean_iou"],
            "olive_iou": olive_iou,
            "vineyard_iou": vineyard_iou,
            "lr": current_lr,
            "time": epoch_time,
        })
        
        # Save best model
        if val_results["mean_iou"] > best_miou:
            best_miou = val_results["mean_iou"]
            patience_counter = 0
            
            best_path = os.path.join(checkpoint_dir, f"{run_name}_best.pth")
            save_checkpoint(
                model, optimizer, scheduler, epoch, val_results, best_path, config
            )
            print(f"  ★ New best mIoU: {best_miou:.4f} → saved to {best_path}")
        else:
            patience_counter += 1
        
        # Save periodic checkpoint
        if (epoch + 1) % 10 == 0:
            periodic_path = os.path.join(checkpoint_dir, f"{run_name}_epoch{epoch+1}.pth")
            save_checkpoint(
                model, optimizer, scheduler, epoch, val_results, periodic_path, config
            )
        
        # Early stopping
        if patience_counter >= patience:
            print(f"\n  ⏹️  Early stopping after {patience} epochs without improvement")
            break
    
    # ─── Save training history ─────────────────────────────────────
    history_path = os.path.join(log_dir, f"{run_name}_history.json")
    os.makedirs(log_dir, exist_ok=True)
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"\nTraining history saved: {history_path}")
    
    # ─── Final evaluation ──────────────────────────────────────────
    print(f"\n{'='*60}")
    print("FINAL EVALUATION ON VALIDATION SET")
    print(f"{'='*60}")
    
    # Load best model
    best_path = os.path.join(checkpoint_dir, f"{run_name}_best.pth")
    if os.path.exists(best_path):
        checkpoint = torch.load(best_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
    
    final_results = validate(model, val_loader, criterion, device, use_amp)
    
    metrics_obj = SegmentationMetrics(
        num_classes=config["classes"]["num_classes"],
        class_names=config["classes"]["names"],
    )
    # Re-evaluate for detailed report
    model.eval()
    with torch.no_grad():
        for batch in val_loader:
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1)
            metrics_obj.update(preds, masks)
    
    results = metrics_obj.compute()
    metrics_obj.print_report(results)
    
    # Save sample predictions
    print(f"\nSaving sample predictions...")
    predictions_dir = config["paths"]["predictions_dir"]
    os.makedirs(predictions_dir, exist_ok=True)
    
    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            if i >= 5:  # Save 5 samples
                break
            
            images = batch["image"].to(device)
            masks = batch["mask"]
            
            outputs = model(images)
            preds = outputs.argmax(dim=1).cpu()
            
            for j in range(min(2, images.shape[0])):
                save_path = os.path.join(predictions_dir, f"sample_{i}_{j}.png")
                plot_prediction(
                    image=images[j].cpu().numpy(),
                    mask_true=masks[j].numpy(),
                    mask_pred=preds[j].numpy(),
                    config=config,
                    save_path=save_path,
                    title=f"Validation Sample {i}_{j}",
                )
    
    print(f"\n{'='*60}")
    print(f"✅ Training complete!")
    print(f"   Best mIoU: {best_miou:.4f}")
    print(f"   Best model: {best_path}")
    print(f"   History: {history_path}")
    print(f"{'='*60}")
    
    print(f"\nNext steps:")
    print(f"  1. python scripts/05_evaluate.py --checkpoint {best_path}")
    print(f"  2. python scripts/06_inference.py --checkpoint {best_path} --input <sentinel2.tif>")


if __name__ == "__main__":
    main()
