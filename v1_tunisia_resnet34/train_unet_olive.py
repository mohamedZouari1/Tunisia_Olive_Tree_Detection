#!/usr/bin/env python3
"""
=============================================================================
U-Net Semantic Segmentation for Olive Grove Detection — Sentinel-2 Tunisia
=============================================================================
Architecture:  U-Net with optional pretrained ResNet34 encoder
               (via segmentation_models_pytorch)
Input:         Sentinel-2 chips (128×128 px, 10m res → 1.28 km²/chip)
               14 bands: B2,B3,B4,B5,B6,B7,B8,B8A,B11,B12,NDVI,NDWI,EVI,SAVI
Output:        3-class mask (0 = background, 1 = extensif, 2 = intensif)

Usage:
  # Chip the GeoTIFFs:
  python train_unet_olive.py --mode chip \\
      --img_tif S2_extensif_mosaic.tif --mask_tif labels_extensif.tif --chip_out ./chips

  # Train with transfer learning:
  python train_unet_olive.py --mode train --data_dir ./chips

  # Inference:
  python train_unet_olive.py --mode predict --image scene.tif

Changes from original:
  - Fixed augmentation leak (separate datasets for train vs val/test)
  - Region-stratified train/val/test split (by parcel prefix)
  - Class weights for imbalanced data (extensif >> intensif)
  - Early stopping (patience=10)
  - Transfer learning via segmentation_models_pytorch (optional)
  - Computes band stats from actual data
  - Windows-safe num_workers
=============================================================================
"""

import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import rasterio
from rasterio.windows import Window
from pathlib import Path
from typing import List, Tuple, Optional
import json
import logging
import re
from collections import defaultdict

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s | %(levelname)s | %(message)s')
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
CHIP_SIZE    = 128       # pixels (128 × 10m = 1.28 km)
OVERLAP      = 32        # chip overlap for inference sliding window
N_BANDS      = 14        # B2…B12 + NDVI + NDWI + EVI + SAVI
N_CLASSES    = 3         # 0=background, 1=extensif, 2=intensif
BATCH_SIZE   = 16        # RTX 4090 can handle 16 easily
EPOCHS       = 80        # More epochs with early stopping
LR           = 3e-4      # Slightly higher LR for transfer learning
WEIGHT_DECAY = 1e-4
DEVICE       = 'cuda' if torch.cuda.is_available() else 'cpu'
SEED         = 42
PATIENCE     = 12        # Early stopping patience

# Band statistics — will be computed from actual data if stats file exists
# Fallback approximate values for Tunisia:
BAND_MEAN = np.array([
    0.053, 0.081, 0.069, 0.100, 0.175, 0.195,
    0.212, 0.228, 0.176, 0.089, 0.310, -0.125, 0.155, 0.215
], dtype=np.float32)

BAND_STD = np.array([
    0.031, 0.038, 0.042, 0.048, 0.065, 0.068,
    0.082, 0.088, 0.072, 0.052, 0.180,  0.098, 0.130, 0.160
], dtype=np.float32)


def load_band_stats(stats_file='band_stats.json'):
    """Load precomputed band statistics if available."""
    global BAND_MEAN, BAND_STD
    if os.path.exists(stats_file):
        with open(stats_file) as f:
            stats = json.load(f)
        BAND_MEAN = np.array(stats['mean'], dtype=np.float32)
        BAND_STD = np.array(stats['std'], dtype=np.float32)
        log.info(f'Loaded band stats from {stats_file}')
    else:
        log.warning(f'No {stats_file} found, using approximate values')


# ---------------------------------------------------------------------------
# U-Net Building Blocks (custom, used when smp is not available)
# ---------------------------------------------------------------------------
class ConvBlock(nn.Module):
    """Double convolution block with BatchNorm + ReLU."""
    def __init__(self, in_ch: int, out_ch: int, dropout: float = 0.1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class Encoder(nn.Module):
    def __init__(self, in_ch: int, features: List[int]):
        super().__init__()
        self.blocks = nn.ModuleList()
        self.pools  = nn.ModuleList()
        ch = in_ch
        for f in features:
            self.blocks.append(ConvBlock(ch, f))
            self.pools.append(nn.MaxPool2d(2))
            ch = f

    def forward(self, x):
        skips = []
        for block, pool in zip(self.blocks, self.pools):
            x = block(x)
            skips.append(x)
            x = pool(x)
        return x, skips


class Bottleneck(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = ConvBlock(in_ch, out_ch, dropout=0.2)

    def forward(self, x):
        return self.block(x)


class Decoder(nn.Module):
    def __init__(self, features: List[int]):
        super().__init__()
        self.up_convs  = nn.ModuleList()
        self.dec_blocks = nn.ModuleList()
        rev = list(reversed(features))
        in_ch = rev[0] * 2
        for f in rev:
            self.up_convs.append(
                nn.ConvTranspose2d(in_ch, f, kernel_size=2, stride=2))
            self.dec_blocks.append(ConvBlock(f * 2, f))
            in_ch = f

    def forward(self, x, skips):
        skips = list(reversed(skips))
        for up, block, skip in zip(self.up_convs, self.dec_blocks, skips):
            x = up(x)
            if x.shape != skip.shape:
                x = F.interpolate(x, size=skip.shape[2:])
            x = torch.cat([skip, x], dim=1)
            x = block(x)
        return x


class UNet(nn.Module):
    """Custom U-Net for Sentinel-2 olive grove segmentation."""
    def __init__(
        self,
        in_channels:  int = N_BANDS,
        out_channels: int = N_CLASSES,
        features:     List[int] = None,
    ):
        super().__init__()
        if features is None:
            features = [64, 128, 256, 512]
        self.encoder    = Encoder(in_channels, features)
        self.bottleneck = Bottleneck(features[-1], features[-1] * 2)
        self.decoder    = Decoder(features)
        self.head       = nn.Conv2d(features[0], out_channels, 1)

    def forward(self, x):
        x, skips = self.encoder(x)
        x = self.bottleneck(x)
        x = self.decoder(x, skips)
        return self.head(x)


def build_model(use_smp=True):
    """
    Build the segmentation model.
    Tries segmentation_models_pytorch (ResNet34 encoder) first,
    falls back to custom U-Net.
    """
    if use_smp:
        try:
            import segmentation_models_pytorch as smp
            model = smp.Unet(
                encoder_name="resnet34",
                encoder_weights="imagenet",
                in_channels=N_BANDS,
                classes=N_CLASSES,
                activation=None,  # raw logits
            )
            log.info('Using smp.Unet with pretrained ResNet34 encoder (transfer learning)')
            return model
        except ImportError:
            log.warning('segmentation_models_pytorch not found, using custom U-Net')

    model = UNet(in_channels=N_BANDS, out_channels=N_CLASSES)
    log.info('Using custom U-Net (no transfer learning)')
    return model


# ---------------------------------------------------------------------------
# Dataset — FIXED: separate instances for train vs val/test
# ---------------------------------------------------------------------------
class OliveChipDataset(Dataset):
    """
    Loads 128×128 chips from chips/images/ and chips/masks/.
    Augmentation is controlled per-instance (no shared state leak).
    """
    def __init__(
        self,
        pairs: List[Tuple[Path, Path]],
        augment: bool = False,
        mean: np.ndarray = None,
        std: np.ndarray = None,
    ):
        self.pairs = pairs
        self.augment = augment
        mean = mean if mean is not None else BAND_MEAN
        std = std if std is not None else BAND_STD
        self.mean = mean[:, None, None]
        self.std = std[:, None, None]
        log.info(f'Dataset: {len(self.pairs)} chips (augment={augment})')

    def __len__(self):
        return len(self.pairs)

    def _load(self, path: Path, is_mask: bool = False):
        with rasterio.open(path) as src:
            arr = src.read()
        if is_mask:
            return arr[0].astype(np.int64)
        return arr.astype(np.float32)

    def _normalize(self, img: np.ndarray) -> np.ndarray:
        return (img - self.mean) / (self.std + 1e-8)

    def _augment(self, img, mask):
        # Random horizontal flip
        if np.random.rand() > 0.5:
            img  = np.flip(img,  axis=2).copy()
            mask = np.flip(mask, axis=1).copy()
        # Random vertical flip
        if np.random.rand() > 0.5:
            img  = np.flip(img,  axis=1).copy()
            mask = np.flip(mask, axis=0).copy()
        # Random 90° rotation
        k = np.random.randint(0, 4)
        img  = np.rot90(img,  k, axes=(1, 2)).copy()
        mask = np.rot90(mask, k, axes=(0, 1)).copy()
        # Random brightness/contrast jitter (subtle, per-band)
        if np.random.rand() > 0.5:
            scale = np.random.uniform(0.9, 1.1, size=(N_BANDS, 1, 1)).astype(np.float32)
            img = img * scale
        return img, mask

    def __getitem__(self, idx):
        img_path, mask_path = self.pairs[idx]
        img  = self._normalize(self._load(img_path))
        mask = self._load(mask_path, is_mask=True)

        if self.augment:
            img, mask = self._augment(img, mask)

        return (
            torch.from_numpy(img.copy()),
            torch.from_numpy(mask.copy()),
        )


# ---------------------------------------------------------------------------
# Data Splitting — Region-stratified by source mosaic
# ---------------------------------------------------------------------------
def discover_chip_pairs(data_dir: str) -> List[Tuple[Path, Path]]:
    """Find all matching image/mask chip pairs."""
    img_dir  = Path(data_dir) / 'images'
    mask_dir = Path(data_dir) / 'masks'
    pairs = []
    for img_path in sorted(img_dir.glob('*.tif')):
        mask_path = mask_dir / img_path.name
        if mask_path.exists():
            pairs.append((img_path, mask_path))
    log.info(f'Found {len(pairs)} chip pairs in {data_dir}')
    return pairs


def stratified_split(pairs, val_ratio=0.15, test_ratio=0.10, seed=SEED):
    """
    Split chips into train/val/test by source region.
    Chips from the same source mosaic (extensif_* vs intensif_*) are grouped,
    and we ensure both sources appear in all splits.
    """
    rng = np.random.RandomState(seed)

    # Group by source prefix
    groups = defaultdict(list)
    for pair in pairs:
        name = pair[0].stem  # e.g., chip_ext_00042 or chip_int_00007
        if '_ext_' in name:
            groups['extensif'].append(pair)
        elif '_int_' in name:
            groups['intensif'].append(pair)
        else:
            groups['unknown'].append(pair)

    # If no grouping was found (all same prefix), fall back to simple split
    if len(groups) <= 1:
        groups = {'all': pairs}

    train_pairs, val_pairs, test_pairs = [], [], []

    for source, source_pairs in groups.items():
        n = len(source_pairs)
        indices = rng.permutation(n)
        n_test = max(1, int(test_ratio * n))
        n_val  = max(1, int(val_ratio * n))

        test_idx = indices[:n_test]
        val_idx  = indices[n_test:n_test + n_val]
        train_idx = indices[n_test + n_val:]

        for i in test_idx:
            test_pairs.append(source_pairs[i])
        for i in val_idx:
            val_pairs.append(source_pairs[i])
        for i in train_idx:
            train_pairs.append(source_pairs[i])

        log.info(f'  Source "{source}": {len(train_idx)} train, '
                 f'{len(val_idx)} val, {len(test_idx)} test')

    return train_pairs, val_pairs, test_pairs


# ---------------------------------------------------------------------------
# Loss — Dice + Weighted CrossEntropy
# ---------------------------------------------------------------------------
class DiceCELoss(nn.Module):
    def __init__(self, n_classes: int = N_CLASSES, smooth: float = 1.0,
                 class_weights: torch.Tensor = None):
        super().__init__()
        self.n_classes = n_classes
        self.smooth = smooth
        # Class weights: background=0.1, extensif=0.5, intensif=1.5
        if class_weights is None:
            class_weights = torch.tensor([0.1, 0.5, 1.5], dtype=torch.float32)
        self.ce = nn.CrossEntropyLoss(weight=class_weights)

    def dice_loss(self, logits, targets):
        probs = F.softmax(logits, dim=1)
        total_dice = 0.0
        for c in range(self.n_classes):
            pred = probs[:, c]
            tgt  = (targets == c).float()
            inter = (pred * tgt).sum()
            total_dice += 1 - (2 * inter + self.smooth) / \
                          (pred.sum() + tgt.sum() + self.smooth)
        return total_dice / self.n_classes

    def forward(self, logits, targets):
        return self.ce(logits, targets) + self.dice_loss(logits, targets)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def compute_iou(preds: torch.Tensor,
                targets: torch.Tensor,
                n_classes: int = N_CLASSES) -> np.ndarray:
    """Return per-class IoU as numpy array."""
    ious = []
    preds   = preds.view(-1).cpu().numpy()
    targets = targets.view(-1).cpu().numpy()
    for c in range(n_classes):
        pred_c = preds   == c
        tgt_c  = targets == c
        inter  = (pred_c & tgt_c).sum()
        union  = (pred_c | tgt_c).sum()
        ious.append(inter / (union + 1e-8) if union > 0 else float('nan'))
    return np.array(ious)


# ---------------------------------------------------------------------------
# Training Loop
# ---------------------------------------------------------------------------
def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for imgs, masks in loader:
        imgs, masks = imgs.to(device), masks.to(device)
        optimizer.zero_grad()
        logits = model(imgs)
        loss   = criterion(logits, masks)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_preds, all_targets = [], []
    for imgs, masks in loader:
        imgs, masks = imgs.to(device), masks.to(device)
        logits = model(imgs)
        total_loss += criterion(logits, masks).item()
        preds = logits.argmax(dim=1)
        all_preds.append(preds.cpu())
        all_targets.append(masks.cpu())
    preds   = torch.cat(all_preds)
    targets = torch.cat(all_targets)
    ious    = compute_iou(preds, targets)
    return total_loss / len(loader), ious


def train(args):
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    # Load band stats
    load_band_stats(os.path.join(args.data_dir, '..', 'band_stats.json'))
    if not os.path.exists('band_stats.json'):
        load_band_stats('band_stats.json')

    # Discover chips and split
    all_pairs = discover_chip_pairs(args.data_dir)
    if len(all_pairs) == 0:
        log.error('No chip pairs found! Run --mode chip first.')
        sys.exit(1)

    train_pairs, val_pairs, test_pairs = stratified_split(all_pairs)

    # Create SEPARATE dataset instances (fixes augmentation leak)
    train_ds = OliveChipDataset(train_pairs, augment=True,
                                 mean=BAND_MEAN, std=BAND_STD)
    val_ds   = OliveChipDataset(val_pairs, augment=False,
                                 mean=BAND_MEAN, std=BAND_STD)
    test_ds  = OliveChipDataset(test_pairs, augment=False,
                                 mean=BAND_MEAN, std=BAND_STD)

    # Windows-safe: use num_workers=0 on Windows, 4 on Linux
    n_workers = 0 if sys.platform == 'win32' else 4
    train_loader = DataLoader(train_ds, BATCH_SIZE, shuffle=True,
                              num_workers=n_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds, BATCH_SIZE, shuffle=False,
                              num_workers=n_workers, pin_memory=True)
    test_loader  = DataLoader(test_ds, 1, shuffle=False)

    # Model
    model = build_model(use_smp=True).to(DEVICE)

    # Class weights on device
    class_weights = torch.tensor([0.1, 0.5, 1.5], dtype=torch.float32).to(DEVICE)

    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=LR * 0.01)
    criterion = DiceCELoss(n_classes=N_CLASSES, class_weights=class_weights)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(f'Model: {n_params/1e6:.2f}M parameters | Device: {DEVICE}')
    log.info(f'Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}')

    best_miou = 0.0
    best_ckpt = args.checkpoint
    history   = []
    patience_counter = 0

    for epoch in range(1, EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, DEVICE)
        val_loss, val_ious = evaluate(model, val_loader, criterion, DEVICE)
        scheduler.step()

        miou = float(np.nanmean(val_ious))
        history.append({
            'epoch': epoch,
            'train_loss': round(train_loss, 4),
            'val_loss':   round(val_loss, 4),
            'mIoU':       round(miou, 4),
            'IoU_bg':     round(float(val_ious[0]), 4),
            'IoU_extensif': round(float(val_ious[1]), 4) if N_CLASSES > 2 else None,
            'IoU_intensif': round(float(val_ious[2]), 4) if N_CLASSES > 2 else None,
            'lr': round(optimizer.param_groups[0]['lr'], 8),
        })

        log.info(f'Epoch {epoch:03d}/{EPOCHS} | '
                 f'Train {train_loss:.4f} | Val {val_loss:.4f} | '
                 f'mIoU {miou:.4f} | '
                 f'IoU/class: {[f"{v:.3f}" for v in val_ious]}')

        if miou > best_miou:
            best_miou = miou
            patience_counter = 0
            torch.save({
                'epoch':       epoch,
                'model_state': model.state_dict(),
                'opt_state':   optimizer.state_dict(),
                'miou':        best_miou,
                'n_classes':   N_CLASSES,
                'n_bands':     N_BANDS,
                'band_mean':   BAND_MEAN.tolist(),
                'band_std':    BAND_STD.tolist(),
            }, best_ckpt)
            log.info(f'  ✓ Saved best checkpoint (mIoU={best_miou:.4f})')
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                log.info(f'  Early stopping at epoch {epoch} '
                         f'(no improvement for {PATIENCE} epochs)')
                break

    # Final test evaluation
    ckpt = torch.load(best_ckpt, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt['model_state'])
    test_loss, test_ious = evaluate(model, test_loader, criterion, DEVICE)
    log.info(f'\n{"="*60}')
    log.info(f'TEST RESULTS (best model from epoch {ckpt["epoch"]})')
    log.info(f'  Test mIoU: {np.nanmean(test_ious):.4f}')
    log.info(f'  IoU background: {test_ious[0]:.4f}')
    log.info(f'  IoU extensif:   {test_ious[1]:.4f}')
    log.info(f'  IoU intensif:   {test_ious[2]:.4f}')
    log.info(f'{"="*60}')

    with open('training_history.json', 'w') as f:
        json.dump(history, f, indent=2)
    log.info('Training complete. History saved to training_history.json')


# ---------------------------------------------------------------------------
# Inference — Sliding Window Prediction on a Full S2 Scene
# ---------------------------------------------------------------------------
@torch.no_grad()
def predict(args):
    ckpt  = torch.load(args.checkpoint, map_location=DEVICE, weights_only=False)

    # Try smp first, fall back to custom
    n_bands = ckpt.get('n_bands', N_BANDS)
    n_classes = ckpt.get('n_classes', N_CLASSES)
    try:
        import segmentation_models_pytorch as smp
        model = smp.Unet(
            encoder_name="resnet34",
            encoder_weights=None,
            in_channels=n_bands,
            classes=n_classes,
            activation=None,
        ).to(DEVICE)
    except ImportError:
        model = UNet(in_channels=n_bands, out_channels=n_classes).to(DEVICE)

    model.load_state_dict(ckpt['model_state'])
    model.eval()

    mean = np.array(ckpt.get('band_mean', BAND_MEAN.tolist()),
                    dtype=np.float32)[:, None, None]
    std  = np.array(ckpt.get('band_std',  BAND_STD.tolist()),
                    dtype=np.float32)[:, None, None]

    with rasterio.open(args.image) as src:
        H, W    = src.height, src.width
        profile = src.profile
        full    = src.read().astype(np.float32)

    full = (full - mean) / (std + 1e-8)

    step      = CHIP_SIZE - OVERLAP
    prob_acc  = np.zeros((n_classes, H, W), np.float32)
    count_acc = np.zeros((H, W), np.float32)

    ys = list(range(0, H - CHIP_SIZE + 1, step)) + [H - CHIP_SIZE]
    xs = list(range(0, W - CHIP_SIZE + 1, step)) + [W - CHIP_SIZE]

    total = len(set(ys)) * len(set(xs))
    done = 0
    for y in set(ys):
        for x in set(xs):
            chip = full[:, y:y+CHIP_SIZE, x:x+CHIP_SIZE]
            t    = torch.from_numpy(chip).unsqueeze(0).to(DEVICE)
            prob = F.softmax(model(t), dim=1).squeeze(0).cpu().numpy()
            prob_acc[:, y:y+CHIP_SIZE, x:x+CHIP_SIZE] += prob
            count_acc[y:y+CHIP_SIZE, x:x+CHIP_SIZE]   += 1
            done += 1
            if done % 100 == 0:
                log.info(f'  Progress: {done}/{total} chips ({done/total*100:.0f}%)')

    pred_mask = prob_acc.argmax(axis=0).astype(np.uint8)

    out_path = Path(args.image).stem + '_olive_mask.tif'
    profile.update(count=1, dtype='uint8', compress='lzw', nodata=255)
    with rasterio.open(out_path, 'w', **profile) as dst:
        dst.write(pred_mask[np.newaxis])

    log.info(f'Prediction saved: {out_path}')
    classes, counts = np.unique(pred_mask, return_counts=True)
    for c, cnt in zip(classes, counts):
        labels = {0:'background', 1:'olive_extensif', 2:'olive_intensif'}
        pct = cnt / pred_mask.size * 100
        log.info(f'  Class {c} ({labels.get(c,"?")}): {cnt:,} px ({pct:.1f}%)')


# ---------------------------------------------------------------------------
# Data Preparation — Chip GeoTIFFs with 50% overlap
# ---------------------------------------------------------------------------
def chip_image(image_path: str, mask_path: str, out_dir: str,
               chip_size: int = CHIP_SIZE, stride: int = None,
               source_tag: str = None):
    """
    Slice a large Sentinel-2 GeoTIFF + matching mask into chips.
    Uses 50% overlap by default for more training data.
    source_tag: prefix for chip names (e.g., 'ext' or 'int')
    """
    if stride is None:
        stride = chip_size // 2  # 50% overlap

    if source_tag is None:
        source_tag = Path(image_path).stem.split('_')[1][:3]  # ext or int

    img_out  = Path(out_dir) / 'images'
    mask_out = Path(out_dir) / 'masks'
    img_out.mkdir(parents=True, exist_ok=True)
    mask_out.mkdir(parents=True, exist_ok=True)

    # Count existing chips to continue numbering
    existing = list(img_out.glob(f'chip_{source_tag}_*.tif'))
    start_idx = len(existing)

    with rasterio.open(image_path) as img_src, \
         rasterio.open(mask_path)  as msk_src:

        H, W    = img_src.height, img_src.width
        n_chips = 0
        n_skipped = 0

        log.info(f'Chipping {image_path} ({W}x{H}) with stride={stride}...')

        for y in range(0, H - chip_size + 1, stride):
            for x in range(0, W - chip_size + 1, stride):
                win  = Window(x, y, chip_size, chip_size)
                mask = msk_src.read(window=win)

                # Skip chips that are >95% background (class 0)
                olive_ratio = (mask[0] > 0).mean()
                if olive_ratio < 0.05:
                    n_skipped += 1
                    continue

                img = img_src.read(window=win)

                # Skip chips with NaN/nodata
                if np.isnan(img).any() or (img == 0).all():
                    n_skipped += 1
                    continue

                name = f'chip_{source_tag}_{start_idx + n_chips:05d}.tif'

                prof_i = img_src.profile.copy()
                prof_i.update(width=chip_size, height=chip_size,
                              transform=rasterio.windows.transform(win, img_src.transform))
                with rasterio.open(img_out / name, 'w', **prof_i) as dst:
                    dst.write(img)

                prof_m = msk_src.profile.copy()
                prof_m.update(width=chip_size, height=chip_size,
                              transform=rasterio.windows.transform(win, msk_src.transform))
                with rasterio.open(mask_out / name, 'w', **prof_m) as dst:
                    dst.write(mask)

                n_chips += 1

    log.info(f'Chipping done: {n_chips} chips saved, {n_skipped} skipped '
             f'(tag={source_tag})')
    return n_chips


# ---------------------------------------------------------------------------
# Compute Band Statistics from Chips
# ---------------------------------------------------------------------------
def compute_band_stats(data_dir: str, output_file: str = 'band_stats.json'):
    """Compute per-band mean and std from all training chips."""
    img_dir = Path(data_dir) / 'images'
    chips = sorted(img_dir.glob('*.tif'))
    log.info(f'Computing band statistics from {len(chips)} chips...')

    # Welford's online algorithm for numerical stability
    n = 0
    mean = np.zeros(N_BANDS, dtype=np.float64)
    M2   = np.zeros(N_BANDS, dtype=np.float64)

    for chip_path in chips:
        with rasterio.open(chip_path) as src:
            data = src.read().astype(np.float64)  # (14, H, W)
        pixels = data.reshape(N_BANDS, -1)  # (14, H*W)

        for j in range(pixels.shape[1]):
            n += 1
            delta = pixels[:, j] - mean
            mean += delta / n
            delta2 = pixels[:, j] - mean
            M2 += delta * delta2

    std = np.sqrt(M2 / n)

    stats = {
        'mean': mean.tolist(),
        'std': std.tolist(),
        'n_pixels': n,
        'n_chips': len(chips),
    }
    with open(output_file, 'w') as f:
        json.dump(stats, f, indent=2)

    log.info(f'Band statistics saved to {output_file}')
    log.info(f'  Mean: {[f"{v:.4f}" for v in mean]}')
    log.info(f'  Std:  {[f"{v:.4f}" for v in std]}')
    return stats


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='U-Net olive grove segmentation — Sentinel-2 Tunisia')
    parser.add_argument('--mode', choices=['train', 'predict', 'chip', 'stats'],
                        default='train')
    parser.add_argument('--data_dir', default='./chips',
                        help='chips/images + chips/masks directories')
    parser.add_argument('--checkpoint', default='best_unet_olive.pth',
                        help='path to save/load model checkpoint')
    parser.add_argument('--image', default=None,
                        help='[predict] path to input GeoTIFF')
    # Chip helper args
    parser.add_argument('--img_tif', default=None,
                        help='[chip] large image GeoTIFF to chip')
    parser.add_argument('--mask_tif', default=None,
                        help='[chip] matching label mask GeoTIFF')
    parser.add_argument('--chip_out', default='./chips',
                        help='[chip] output directory for chips')
    parser.add_argument('--source_tag', default=None,
                        help='[chip] source tag for chip names (ext/int)')
    parser.add_argument('--no_smp', action='store_true',
                        help='[train] disable segmentation_models_pytorch')

    args = parser.parse_args()

    if args.mode == 'train':
        train(args)
    elif args.mode == 'predict':
        if not args.image:
            parser.error('--image required for predict mode')
        predict(args)
    elif args.mode == 'chip':
        if not args.img_tif or not args.mask_tif:
            parser.error('--img_tif and --mask_tif required for chip mode')
        chip_image(args.img_tif, args.mask_tif, args.chip_out,
                   source_tag=args.source_tag)
    elif args.mode == 'stats':
        compute_band_stats(args.data_dir)
