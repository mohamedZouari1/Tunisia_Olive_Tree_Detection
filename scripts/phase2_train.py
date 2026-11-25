"""
ZitounEye - Phase 2 Training: Jaen + Cordoba + Sevilla
=======================================================
Builds on Phase 1 by adding 2 new provinces, then fine-tunes 
the Phase 1 checkpoint for 50 epochs with more diverse data.

Usage:
    python scripts/phase2_train.py
    python scripts/phase2_train.py --debug     (2 epochs, fast test)
    python scripts/phase2_train.py --skip-patching  (reuse patches)

What's new vs Phase 1:
    - 3 provinces instead of 1 (Jaen + Cordoba + Sevilla)
    - Sevilla has palms, rice, vineyards -> model learns diversity
    - Fine-tuned from Phase 1 checkpoint (transfer learning)
    - 50 epochs with cosine annealing LR
    - Stricter patch filtering
"""

import sys, os, glob, argparse, warnings, json
from datetime import datetime
from collections import Counter

import io
try:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
except Exception:
    pass

warnings.filterwarnings("ignore")

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

# ---------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------
PROJECT_ROOT   = r"c:\Users\AORUS\Documents\zitoun_eye"
PHASE1_PATCHES = os.path.join(PROJECT_ROOT, "data", "patches")           # existing Jaen patches
PHASE2_IMAGERY = os.path.join(PROJECT_ROOT, "spain_phase2_images")       # new imagery to be downloaded
PHASE2_PATCHES = os.path.join(PROJECT_ROOT, "data", "patches_phase2")    # combined patches output
CHECKPOINTS    = os.path.join(PROJECT_ROOT, "checkpoints")
LOGS           = os.path.join(PROJECT_ROOT, "logs")

SHAPEFILES = {
    "es_j_2023":  os.path.join(PROJECT_ROOT, "ES_2023", "ES_2023", "es_j_2023",  "es_j_2023.shp"),
    "es_co_2023": os.path.join(PROJECT_ROOT, "ES_2023", "ES_2023", "es_co_2023", "es_co_2023.shp"),
    "es_se_2023": os.path.join(PROJECT_ROOT, "ES_2023", "ES_2023", "es_se_2023", "es_se_2023.shp"),
}

NUM_CLASSES  = 7
MODEL_BANDS  = 56
PATCH_SIZE   = 256
OVERLAP      = 64
BATCH_SIZE   = 16
EPOCHS       = 50
LR           = 5e-4
EARLY_STOP   = 15

CLASS_NAMES = [
    "background", "olive_grove", "vineyard",
    "other_orchard", "arable_crop", "building", "road"
]

# Weights: emphasize olive more, slightly downweight background
CLASS_WEIGHTS = torch.tensor([0.1, 4.0, 2.0, 1.5, 1.0, 0.5, 0.5], dtype=torch.float32)

OLIVE_KEYWORDS = ["olive", "olea", "olivo"]
VINE_KEYWORDS  = ["vineyard", "vine", "grape", "vitis"]
ORCHARD_KEYWORDS = [
    "almond", "citrus", "orange", "lemon", "fig", "carob", "cherry",
    "peach", "apple", "pear", "plum", "walnut", "pistachio", "avocado",
    "mango", "tropical", "fruit", "orchard", "nut_tree", "pomegranate",
    "plantation", "permanent_crop",
]
ARABLE_KEYWORDS = [
    "wheat", "barley", "maize", "corn", "rice", "sunflower", "cotton",
    "sugar", "potato", "tomato", "legume", "bean", "pea", "soybean",
    "rape", "canola", "oat", "rye", "cereal", "grain", "arable", "annual",
    "vegetable", "forage", "hay", "clover", "alfalfa", "sorghum", "tobacco",
    "lavender", "artichoke", "pepper", "melon", "onion", "fallow",
    "horticultur", "flower", "nursery",
]


def classify_by_name(name_str):
    if not name_str or str(name_str) == "nan":
        return 0
    name = str(name_str).lower().strip()
    if any(kw in name for kw in OLIVE_KEYWORDS):
        return 1
    if any(kw in name for kw in VINE_KEYWORDS):
        return 2
    if any(kw in name for kw in ORCHARD_KEYWORDS):
        if "olive" not in name:
            return 3
    if any(kw in name for kw in ARABLE_KEYWORDS):
        return 4
    return 0


# ---------------------------------------------------------------
# DATASET
# ---------------------------------------------------------------
class PatchDataset(Dataset):
    def __init__(self, patch_dir, norm_mean, norm_std, augment=False):
        self.files  = sorted(glob.glob(os.path.join(patch_dir, "*_image.npy")))
        self.mean   = norm_mean[:, np.newaxis, np.newaxis]
        self.std    = norm_std[:, np.newaxis, np.newaxis]
        self.augment = augment

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img_path = self.files[idx]
        msk_path = img_path.replace("_image.npy", "_mask.npy")
        img = np.load(img_path).astype(np.float32)
        msk = np.load(msk_path).astype(np.int64)

        img = (img - self.mean) / self.std

        if self.augment:
            # Random horizontal flip
            if np.random.rand() > 0.5:
                img = np.flip(img, axis=2).copy()
                msk = np.flip(msk, axis=1).copy()
            # Random vertical flip
            if np.random.rand() > 0.5:
                img = np.flip(img, axis=1).copy()
                msk = np.flip(msk, axis=0).copy()
            # Random 90-degree rotation
            k = np.random.randint(0, 4)
            if k > 0:
                img = np.rot90(img, k, axes=(1, 2)).copy()
                msk = np.rot90(msk, k, axes=(0, 1)).copy()

        return torch.tensor(img), torch.tensor(msk)


def get_sample_weights(dataset):
    """Weight patches so olive-containing ones are sampled more."""
    weights = []
    for img_path in dataset.files:
        msk_path = img_path.replace("_image.npy", "_mask.npy")
        msk = np.load(msk_path)
        olive_frac = (msk == 1).sum() / msk.size
        w = 1.0 + olive_frac * 8.0   # olive patches get up to 9x more weight
        weights.append(w)
    return torch.tensor(weights, dtype=torch.float32)


# ---------------------------------------------------------------
# RASTERIZATION (same logic as Phase 1)
# ---------------------------------------------------------------
def rasterize_tile(shapefile_path, reference_tif_path):
    import rasterio
    from rasterio import features as rio_features
    import geopandas as gpd
    from pyproj import Transformer

    with rasterio.open(reference_tif_path) as src:
        transform = src.transform
        H, W = src.height, src.width
        crs   = src.crs
        bounds = src.bounds

    try:
        xformer = Transformer.from_crs(str(crs), "EPSG:4258", always_xy=True)
        minx, miny = xformer.transform(bounds.left, bounds.bottom)
        maxx, maxy = xformer.transform(bounds.right, bounds.top)
        bbox = (minx, miny, maxx, maxy)
    except Exception:
        bbox = (bounds.left, bounds.bottom, bounds.right, bounds.top)

    gdf = gpd.read_file(shapefile_path, bbox=bbox)
    if str(gdf.crs) != str(crs):
        gdf = gdf.to_crs(crs)
    gdf = gdf[gdf.geometry.is_valid & ~gdf.geometry.is_empty]
    print(f"      {len(gdf)} parcels in tile")

    if len(gdf) == 0:
        return np.zeros((H, W), dtype=np.uint8)

    name_col = next((c for c in ["EC_hcat_n","ec_hcat_n","EC_trans_n"] if c in gdf.columns), None)
    if name_col is None:
        print(f"      No name column found! Columns: {list(gdf.columns)}")
        return np.zeros((H, W), dtype=np.uint8)

    shapes = []
    counts = Counter()
    for _, row in gdf.iterrows():
        cls = classify_by_name(row[name_col])
        counts[cls] += 1
        if cls > 0:
            shapes.append((row.geometry, cls))

    for c in sorted(counts):
        print(f"      Class {c} ({CLASS_NAMES[c]}): {counts[c]} parcels")

    label = np.zeros((H, W), dtype=np.uint8)
    if shapes:
        rio_features.rasterize(shapes, out=label, transform=transform, all_touched=True)
    return label


def cut_patches(tile_paths, label_raster_func, tile_idx, out_dir,
                patch_size=PATCH_SIZE, overlap=OVERLAP,
                min_valid=0.5, max_bg=0.97):
    import rasterio
    from rasterio.windows import Window

    season_order = ["spring", "summer", "fall", "winter"]
    ref_path = tile_paths[season_order[0]]

    with rasterio.open(ref_path) as src:
        H, W = src.height, src.width
        n_bands = src.count

    print(f"      Tile: {H}x{W}, {n_bands*4} total bands -> strip processing")

    label = label_raster_func(ref_path)
    label = label[:H, :W]

    srcs = [rasterio.open(tile_paths[s]) for s in season_order]
    stride  = patch_size - overlap
    n_total = 0
    n_olive = 0

    try:
        for y in range(0, H - patch_size + 1, stride):
            window = Window(0, y, W, patch_size)
            strip_bands = []
            for sf in srcs:
                bd = sf.read(window=window).astype(np.float32)
                np.nan_to_num(bd, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
                strip_bands.append(bd)
            strip = np.concatenate(strip_bands, axis=0)  # (56, patch_size, W)
            lbl_strip = label[y:y+patch_size, :W]

            for x in range(0, W - patch_size + 1, stride):
                img_p = strip[:, :, x:x+patch_size]
                lbl_p = lbl_strip[:, x:x+patch_size]

                valid = np.any(img_p != 0, axis=0).sum() / (patch_size**2)
                if valid < min_valid:
                    continue
                u, c = np.unique(lbl_p, return_counts=True)
                if c.max() / (patch_size**2) > max_bg:
                    continue

                name = f"p2_t{tile_idx}_{n_total:06d}"
                np.save(os.path.join(out_dir, f"{name}_image.npy"), img_p)
                np.save(os.path.join(out_dir, f"{name}_mask.npy"), lbl_p)
                n_total += 1
                if 1 in u:
                    n_olive += 1

            if (y // stride) % 10 == 0:
                print(f"      Strip y={y}/{H}: {n_total} patches ({n_olive} olive)")
    finally:
        for sf in srcs:
            sf.close()

    print(f"      Done: {n_total} patches ({n_olive} olive)")
    return n_total, n_olive


# ---------------------------------------------------------------
# TILE MATCHING (same logic as Phase 1)
# ---------------------------------------------------------------
def match_tiles(imagery_dir, province_key):
    tifs = glob.glob(os.path.join(imagery_dir, f"{province_key}*.tif"))
    if not tifs:
        return {}

    season_map = {}
    for path in tifs:
        fname = os.path.basename(path).lower()
        for season in ["spring", "summer", "fall", "winter"]:
            if f"_{season}_" in fname:
                # Group by spatial tile (by pixel offset in filename)
                parts = fname.replace(".tif", "").split("-")
                # tile key = offset suffix
                if len(parts) >= 3:
                    tile_key = f"{parts[1]}_{parts[2].split('.')[0]}"
                else:
                    tile_key = "0_0"
                key = (tile_key, season)
                if key not in season_map or \
                   os.path.getsize(path) > os.path.getsize(season_map[key]):
                    season_map[key] = path

    # Group by tile
    tiles = {}
    for (tile_key, season), path in season_map.items():
        tiles.setdefault(tile_key, {})[season] = path

    complete = {k: v for k, v in tiles.items() if len(v) == 4}
    return complete


# ---------------------------------------------------------------
# MODEL
# ---------------------------------------------------------------
def build_model():
    try:
        import segmentation_models_pytorch as smp
    except ImportError:
        raise ImportError("Run: pip install segmentation-models-pytorch")
    return smp.Unet(
        encoder_name="efficientnet-b4",
        encoder_weights=None,
        in_channels=MODEL_BANDS,
        classes=NUM_CLASSES,
    )


def load_phase1_checkpoint(model, device):
    """Load the best Phase 1 checkpoint to fine-tune from."""
    pattern = os.path.join(CHECKPOINTS, "phase1_*_best.pth")
    files = sorted(glob.glob(pattern))
    if not files:
        print("  No Phase 1 checkpoint found - training from scratch")
        return model
    ckpt_path = files[-1]
    print(f"  Fine-tuning from: {os.path.basename(ckpt_path)}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state, strict=True)
    return model


# ---------------------------------------------------------------
# LOSS
# ---------------------------------------------------------------
class DiceFocalLoss(nn.Module):
    def __init__(self, class_weights, gamma=2.0, dice_weight=0.5):
        super().__init__()
        self.gamma       = gamma
        self.dice_weight = dice_weight
        self.ce_weight   = 1.0 - dice_weight
        self.class_weights = class_weights

    def forward(self, logits, targets):
        # Focal loss
        ce = nn.functional.cross_entropy(logits, targets,
                                         weight=self.class_weights.to(logits.device))
        pt = torch.exp(-ce)
        focal = ((1 - pt) ** self.gamma) * ce

        # Dice loss (per class, macro average)
        probs = torch.softmax(logits, dim=1)
        dice = 0.0
        for c in range(NUM_CLASSES):
            p = probs[:, c]
            t = (targets == c).float()
            intersection = (p * t).sum()
            dice += 1 - (2 * intersection + 1) / (p.sum() + t.sum() + 1)
        dice /= NUM_CLASSES

        return self.focal_weight * focal + self.dice_weight * dice

    @property
    def focal_weight(self):
        return self.ce_weight


# ---------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug",          action="store_true", help="2 epochs only")
    parser.add_argument("--skip-patching",  action="store_true", help="Skip to training")
    parser.add_argument("--epochs",         type=int, default=EPOCHS)
    parser.add_argument("--batch-size",     type=int, default=BATCH_SIZE)
    args = parser.parse_args()

    if args.debug:
        args.epochs = 2

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("=" * 60)
    print("  ZitounEye - Phase 2: Jaen + Cordoba + Sevilla")
    print("=" * 60)
    print(f"  Device:  {device}")
    print(f"  Epochs:  {args.epochs}")
    print()

    # Create output dirs
    os.makedirs(os.path.join(PHASE2_PATCHES, "train"), exist_ok=True)
    os.makedirs(os.path.join(PHASE2_PATCHES, "val"),   exist_ok=True)

    # ---------------------------------------------------------------
    # STEP 1: REUSE JAEN PATCHES (already cut in Phase 1)
    # ---------------------------------------------------------------
    if not args.skip_patching:
        print("=" * 60)
        print("STEP 1: Copying Phase 1 Jaen patches")
        print("=" * 60)
        import shutil
        jaen_train = os.path.join(PHASE1_PATCHES, "train")
        jaen_val   = os.path.join(PHASE1_PATCHES, "val")
        phase2_train = os.path.join(PHASE2_PATCHES, "train")
        phase2_val   = os.path.join(PHASE2_PATCHES, "val")

        n_copied = 0
        for split_src, split_dst in [(jaen_train, phase2_train), (jaen_val, phase2_val)]:
            for f in glob.glob(os.path.join(split_src, "*.npy")):
                dst = os.path.join(split_dst, os.path.basename(f))
                if not os.path.exists(dst):
                    shutil.copy2(f, dst)
                    n_copied += 1
        print(f"  Copied {n_copied} Jaen patch files")

        # ---------------------------------------------------------------
        # STEP 2: NEW PROVINCES (Cordoba + Sevilla)
        # ---------------------------------------------------------------
        print()
        print("=" * 60)
        print("STEP 2: Rasterize + patch new provinces")
        print("=" * 60)

        new_provinces = ["es_co_2023", "es_se_2023"]
        tile_counter  = 100  # start from 100 so no clash with Jaen tiles (0-3)

        for prov_key in new_provinces:
            print(f"\n  Province: {prov_key}")
            shp_path = SHAPEFILES[prov_key]

            if not os.path.exists(shp_path):
                print(f"  Shapefile not found: {shp_path}")
                continue

            # Find imagery
            img_dir = PHASE2_IMAGERY
            tiles = match_tiles(img_dir, prov_key)
            print(f"  Found {len(tiles)} complete tiles (all 4 seasons)")

            if not tiles:
                print(f"  No imagery found! Download it from GEE first.")
                print(f"  Run: gee_phase2_cordoba_sevilla.js in Earth Engine")
                continue

            tile_list = sorted(tiles.items())
            n_train = max(1, int(len(tile_list) * 0.75))

            for i, (tile_key, tile_paths) in enumerate(tile_list):
                split = "train" if i < n_train else "val"
                split_dir = os.path.join(PHASE2_PATCHES, split)
                print(f"\n  Tile {i+1}/{len(tile_list)} ({tile_key}) -> {split}")

                def label_fn(ref_path, shp=shp_path):
                    return rasterize_tile(shp, ref_path)

                n_p, n_o = cut_patches(tile_paths, label_fn, tile_counter, split_dir)
                tile_counter += 1

        # ---------------------------------------------------------------
        # STEP 3: NORMALIZATION STATS
        # ---------------------------------------------------------------
        print()
        print("=" * 60)
        print("STEP 3: Computing normalization stats")
        print("=" * 60)
        train_dir = os.path.join(PHASE2_PATCHES, "train")
        files = glob.glob(os.path.join(train_dir, "*_image.npy"))[:500]  # subsample
        running_sum  = np.zeros(MODEL_BANDS, dtype=np.float64)
        running_sum2 = np.zeros(MODEL_BANDS, dtype=np.float64)
        n_pixels = 0
        for f in files:
            img = np.load(f).astype(np.float64)  # (56, H, W)
            running_sum  += img.sum(axis=(1, 2))
            running_sum2 += (img**2).sum(axis=(1, 2))
            n_pixels += img.shape[1] * img.shape[2]
        mean = running_sum  / n_pixels
        std  = np.sqrt(running_sum2 / n_pixels - mean**2)
        stats_path = os.path.join(PHASE2_PATCHES, "normalize_stats.npz")
        np.savez(stats_path, mean=mean.astype(np.float32), std=std.astype(np.float32))
        print(f"  Saved: {stats_path}")

    else:
        print("Skipping patching (--skip-patching)")
        stats_path = os.path.join(PHASE2_PATCHES, "normalize_stats.npz")
        if not os.path.exists(stats_path):
            # Fall back to Phase 1 stats
            stats_path = os.path.join(PHASE1_PATCHES, "normalize_stats.npz")
            print(f"  Using Phase 1 stats: {stats_path}")

    # ---------------------------------------------------------------
    # STEP 4: LOAD DATASETS
    # ---------------------------------------------------------------
    print()
    print("=" * 60)
    print("STEP 4: Loading datasets")
    print("=" * 60)

    norm = np.load(stats_path)
    mean_arr = norm["mean"]
    std_arr  = norm["std"]
    std_arr  = np.where(std_arr < 1e-6, 1.0, std_arr)

    train_dir = os.path.join(PHASE2_PATCHES, "train")
    val_dir   = os.path.join(PHASE2_PATCHES, "val")

    if not os.path.exists(train_dir) or len(glob.glob(os.path.join(train_dir, "*.npy"))) == 0:
        print("No training patches found! Run without --skip-patching first.")
        return

    train_ds = PatchDataset(train_dir, mean_arr, std_arr, augment=True)
    val_ds   = PatchDataset(val_dir,   mean_arr, std_arr, augment=False)
    print(f"  Train: {len(train_ds)} patches")
    print(f"  Val:   {len(val_ds)} patches")

    # Weighted sampler for balanced olive training
    sample_weights = get_sample_weights(train_ds)
    sampler = WeightedRandomSampler(sample_weights, len(train_ds), replacement=True)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler,
                              num_workers=0, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                              num_workers=0, pin_memory=True)

    # ---------------------------------------------------------------
    # STEP 5: BUILD & LOAD MODEL
    # ---------------------------------------------------------------
    print()
    print("=" * 60)
    print("STEP 5: Building model")
    print("=" * 60)
    model = build_model()
    model = load_phase1_checkpoint(model, device)
    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters: {n_params:,}")

    # ---------------------------------------------------------------
    # STEP 6: TRAINING
    # ---------------------------------------------------------------
    print()
    print("=" * 60)
    print("STEP 6: Training")
    print("=" * 60)

    criterion = DiceFocalLoss(CLASS_WEIGHTS)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    scaler     = torch.amp.GradScaler()
    best_iou   = 0.0
    best_olive = 0.0
    no_improve = 0
    history    = []
    ckpt_name  = f"phase2_{timestamp}_best.pth"

    for epoch in range(1, args.epochs + 1):
        import time
        t0 = time.time()

        # Train
        model.train()
        train_loss = 0.0
        n_batches  = 0
        for imgs, masks in train_loader:
            imgs, masks = imgs.to(device), masks.to(device)
            with torch.amp.autocast(device_type="cuda"):
                out  = model(imgs)
                loss = criterion(out, masks)
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item()
            n_batches  += 1
        train_loss /= max(n_batches, 1)

        # Validate
        model.eval()
        val_loss   = 0.0
        n_val      = 0
        confusion  = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs, masks = imgs.to(device), masks.to(device)
                with torch.amp.autocast(device_type="cuda"):
                    out  = model(imgs)
                    loss = criterion(out, masks)
                val_loss += loss.item()
                n_val    += 1
                preds   = out.argmax(dim=1).cpu().numpy().ravel()
                targets = masks.cpu().numpy().ravel()
                valid   = (targets >= 0) & (targets < NUM_CLASSES) & (preds >= 0) & (preds < NUM_CLASSES)
                np.add.at(confusion, (targets[valid], preds[valid]), 1)
        val_loss /= max(n_val, 1)

        # Compute per-class IoU
        ious = []
        for c in range(NUM_CLASSES):
            tp = confusion[c, c]
            fp = confusion[:, c].sum() - tp
            fn = confusion[c, :].sum() - tp
            denom = tp + fp + fn
            ious.append(tp / denom if denom > 0 else 0.0)
        miou       = np.mean(ious)
        olive_iou  = ious[1]
        elapsed    = time.time() - t0

        print(f"Epoch {epoch:>3}/{args.epochs} | "
              f"Loss: {train_loss:.4f}/{val_loss:.4f} | "
              f"mIoU: {miou:.4f} | olive: {olive_iou:.4f} | "
              f"LR: {scheduler.get_last_lr()[0]:.2e} | {elapsed:.0f}s")

        scheduler.step()

        history.append({
            "epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
            "miou": miou, "olive_iou": olive_iou
        })

        if miou > best_iou:
            best_iou   = miou
            best_olive = olive_iou
            no_improve = 0
            ckpt_path  = os.path.join(CHECKPOINTS, ckpt_name)
            torch.save({
                "epoch":           epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "miou":            miou,
                "olive_iou":       olive_iou,
            }, ckpt_path)
            print(f"  * Best mIoU: {best_iou:.4f} (olive: {best_olive:.4f}) -> {ckpt_name}")
        else:
            no_improve += 1
            if no_improve >= EARLY_STOP and not args.debug:
                print(f"\n  Early stopping (no improvement for {EARLY_STOP} epochs)")
                break

    # Save history
    hist_path = os.path.join(LOGS, f"phase2_{timestamp}_history.json")
    with open(hist_path, "w") as f:
        json.dump(history, f, indent=2)

    # Final report
    print()
    print("=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"  Best mIoU:       {best_iou:.4f}")
    print(f"  Best Olive IoU:  {best_olive:.4f}")
    print(f"  Checkpoint:      {os.path.join(CHECKPOINTS, ckpt_name)}")
    print()
    print("  Per-class IoU (best epoch):")
    for c, name in enumerate(CLASS_NAMES):
        marker = " <- TARGET" if c == 1 else ""
        print(f"    {name:<18}: {ious[c]:.4f}{marker}")
    print()
    print("  Phase 2 complete!")
    print("  Next: Download imagery for more provinces and run Phase 3.")


if __name__ == "__main__":
    main()
