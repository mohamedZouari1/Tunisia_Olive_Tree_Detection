"""
ZitounEye - Phase 1: End-to-End Training on Jaen
==================================================
Single script that does EVERYTHING for Phase 1:
  1. Reads EuroCrop shapefiles -> reclassifies to 7 classes
  2. Rasterizes labels onto each Sentinel-2 tile
  3. Stacks 4 seasonal tiles -> 56-band images
  4. Cuts into 256256 training patches
  5. Computes normalization stats
  6. Trains U-Net with EfficientNet-B4
  7. Evaluates and saves results

Usage:
    python scripts/phase1_train_jaen.py
    python scripts/phase1_train_jaen.py --epochs 5 --debug     # Quick test
    python scripts/phase1_train_jaen.py --epochs 100 --loss ultimate  # Best quality
"""

import os
import sys
import argparse
import warnings
import time
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torch.cuda.amp import autocast, GradScaler
from pathlib import Path
from datetime import datetime
from collections import Counter

warnings.filterwarnings("ignore")

# Fix Windows console encoding
import io
try:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
except Exception:
    pass

# ===============================================================
# CONFIGURATION
# ===============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
IMAGERY_DIR = PROJECT_ROOT / "spain_j_images"
SHAPEFILES_DIR = PROJECT_ROOT / "ES_2023" / "ES_2023" / "es_j_2023"
CROP_MAPPING_CSV = PROJECT_ROOT / "es_2023.csv"
HCAT_CSV = PROJECT_ROOT / "HCAT3.csv"
OUTPUT_DIR = PROJECT_ROOT / "data"
PATCHES_DIR = OUTPUT_DIR / "patches"
CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"
PREDICTIONS_DIR = PROJECT_ROOT / "predictions"

# 7-class mapping
CLASS_NAMES = ["background", "olive_grove", "vineyard", "other_orchard", "arable_crop", "building", "road"]
NUM_CLASSES = 7

# Olive-related Spanish crop codes from es_2023.csv
# These map to HCAT3 codes which we'll classify
OLIVE_CODES = set()  # Will be populated from CSV

# Class weights for loss function (olive upweighted 3x)
CLASS_WEIGHTS = [0.1, 3.0, 1.5, 1.5, 0.5, 2.0, 2.0]

PATCH_SIZE = 256
PATCH_OVERLAP = 64


# ===============================================================
# STEP 1: BUILD CROP CODE -> CLASS ID MAPPING
# ===============================================================

def build_crop_mapping():
    """
    Read es_2023.csv and HCAT3.csv to map Spanish crop codes to our 7 classes.
    
    Flow: Spanish code (EC_hcat_c) -> HCAT3 code -> HCAT3 name -> our class ID
    """
    import csv
    
    print("\n" + "=" * 60)
    print("STEP 1: Building crop code mapping")
    print("=" * 60)
    
    # Read HCAT3 taxonomy
    hcat_names = {}
    with open(HCAT_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            code = row.get("HCAT3_code", "").strip()
            name = row.get("HCAT3_name", "").strip().lower()
            if code:
                hcat_names[code] = name
    
    print(f"  Loaded {len(hcat_names)} HCAT3 categories")
    
    # Read Spanish mapping
    es_to_hcat = {}
    with open(CROP_MAPPING_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            es_code = row.get("EC_hcat_c", "").strip()
            # Try multiple possible column names for the Spanish code
            local_code = (
                row.get("EC_trans_n", "").strip() or
                row.get("EC_hcat_n", "").strip()
            )
            if es_code:
                es_to_hcat[es_code] = local_code
    
    print(f"  Loaded {len(es_to_hcat)} Spanish crop code mappings")
    
    # Build final mapping: HCAT3 code -> class ID
    code_to_class = {}
    
    for hcat_code, hcat_name in hcat_names.items():
        name = hcat_name.lower()
        
        # CLASS 1: Olive grove
        if any(kw in name for kw in ["olive", "olea", "olivo", "aceituna"]):
            code_to_class[hcat_code] = 1
        # CLASS 2: Vineyard
        elif any(kw in name for kw in ["vineyard", "vine", "grape", "vitis", "vinedo", "vid"]):
            code_to_class[hcat_code] = 2
        # CLASS 3: Other orchards
        elif any(kw in name for kw in [
            "almond", "citrus", "orange", "lemon", "fig", "carob", "cherry",
            "peach", "apple", "pear", "plum", "apricot", "walnut", "pistachio",
            "pomegranate", "avocado", "mango", "chestnut", "hazelnut",
            "fruit", "orchard", "nut", "tree", "berries"
        ]):
            code_to_class[hcat_code] = 3
        # CLASS 4: Arable crops
        elif any(kw in name for kw in [
            "wheat", "barley", "maize", "corn", "rice", "sunflower", "cotton",
            "sugar", "potato", "tomato", "legume", "bean", "pea", "lentil",
            "soybean", "rape", "canola", "oat", "rye", "triticale",
            "cereal", "grain", "arable", "annual", "crop", "vegetable",
            "forage", "hay", "clover", "alfalfa", "lucerne", "sorghum",
            "tobacco", "flax", "hemp", "herbs", "spice", "lavender",
            "artichoke", "pepper", "melon", "lettuce", "onion", "garlic"
        ]):
            code_to_class[hcat_code] = 4
        # CLASS 0: Everything else (forest, fallow, grassland, etc.)
        else:
            code_to_class[hcat_code] = 0
    
    # Count per class
    class_counts = Counter(code_to_class.values())
    for cls_id, name in enumerate(CLASS_NAMES[:5]):
        print(f"  Class {cls_id} ({name}): {class_counts.get(cls_id, 0)} HCAT3 codes")
    
    olive_codes = [k for k, v in code_to_class.items() if v == 1]
    print(f"\n   Olive HCAT3 codes: {olive_codes}")
    
    return code_to_class


def classify_by_name(name_str):
    """
    Classify a parcel by its EC_hcat_n name.
    
    The shapefile has columns:
      EC_hcat_n = 'olive_plantations', 'pasture_meadow_grassland_grass', etc.
      EC_trans_n = 'OLIVE', 'PERMANENT PASTURES OF 5 OR MORE YEARS', etc.
    """
    if not name_str or str(name_str) == "nan":
        return 0
    
    name = str(name_str).lower().strip()
    
    # CLASS 1: Olive grove
    if any(kw in name for kw in ["olive", "olea", "olivo"]):
        return 1
    # CLASS 2: Vineyard
    if any(kw in name for kw in ["vineyard", "vine", "grape", "vitis", "vinedo"]):
        return 2
    # CLASS 3: Other orchards / permanent crops (trees)
    if any(kw in name for kw in [
        "almond", "citrus", "orange", "lemon", "fig", "carob", "cherry",
        "peach", "apple", "pear", "plum", "apricot", "walnut", "pistachio",
        "pomegranate", "avocado", "mango", "chestnut", "hazelnut",
        "fruit", "orchard", "nut_tree", "berries", "tropical",
        "plantation", "permanent_crop",
    ]):
        # But NOT olive plantations (already caught above)
        if "olive" not in name:
            return 3
    # CLASS 4: Arable crops
    if any(kw in name for kw in [
        "wheat", "barley", "maize", "corn", "rice", "sunflower", "cotton",
        "sugar", "potato", "tomato", "legume", "bean", "pea", "lentil",
        "soybean", "rape", "canola", "oat", "rye", "triticale",
        "cereal", "grain", "arable", "annual", "vegetable",
        "forage", "hay", "clover", "alfalfa", "lucerne", "sorghum",
        "tobacco", "flax", "hemp", "herb", "spice", "lavender",
        "artichoke", "pepper", "melon", "lettuce", "onion", "garlic",
        "horticultur", "flower", "nursery", "fallow",
        "temporary_grass",
    ]):
        return 4
    
    # Everything else (forest, pasture, grassland, water, etc.) -> background
    return 0


def rasterize_labels_for_tile(shapefile_path, reference_tif_path, code_to_class=None):
    """
    Rasterize vector crop parcels onto a Sentinel-2 tile grid.
    Uses EC_hcat_n (name) column for direct classification.
    Returns (H, W) numpy array of class IDs.
    """
    import rasterio
    from rasterio import features as rio_features
    import geopandas as gpd
    
    # Get reference grid
    with rasterio.open(reference_tif_path) as src:
        transform = src.transform
        height = src.height
        width = src.width
        crs = src.crs
        bounds = src.bounds
    
    # Convert UTM bounds to EPSG:4258 for bbox filter (shapefile is in EPSG:4258)
    from pyproj import Transformer
    try:
        transformer = Transformer.from_crs(str(crs), "EPSG:4258", always_xy=True)
        minx, miny = transformer.transform(bounds.left, bounds.bottom)
        maxx, maxy = transformer.transform(bounds.right, bounds.top)
        filter_bbox = (minx, miny, maxx, maxy)
    except Exception:
        filter_bbox = (bounds.left, bounds.bottom, bounds.right, bounds.top)
    
    # Read shapefile filtered to tile extent
    print(f"    Reading shapefile (bbox filter)...")
    gdf = gpd.read_file(shapefile_path, bbox=filter_bbox)
    
    # Reproject to match imagery CRS
    if str(gdf.crs) != str(crs):
        print(f"    Reprojecting {gdf.crs} -> {crs}")
        gdf = gdf.to_crs(crs)
    
    gdf = gdf[gdf.geometry.is_valid & ~gdf.geometry.is_empty]
    print(f"    {len(gdf)} parcels in tile extent")
    
    if len(gdf) == 0:
        return np.zeros((height, width), dtype=np.uint8)
    
    # Classify each parcel using EC_hcat_n name
    # Try EC_hcat_n first, then EC_trans_n
    name_col = None
    for col in ["EC_hcat_n", "ec_hcat_n", "EC_trans_n", "ec_trans_n"]:
        if col in gdf.columns:
            name_col = col
            break
    
    if name_col is None:
        # Fallback: try EC_hcat_c with code_to_class mapping
        for col in ["EC_hcat_c", "ec_hcat_c"]:
            if col in gdf.columns:
                print(f"    Using code column '{col}' with mapping")
                shapes = []
                for _, row in gdf.iterrows():
                    code = str(row[col]).strip()
                    cls_id = code_to_class.get(code, 0) if code_to_class else 0
                    if cls_id > 0:
                        shapes.append((row.geometry, cls_id))
                break
        else:
            print(f"     No crop name/code column found! Columns: {list(gdf.columns)}")
            return np.zeros((height, width), dtype=np.uint8)
    else:
        print(f"    Using name column '{name_col}' for classification")
        
        # Classify and count
        class_counts = Counter()
        shapes = []
        
        for _, row in gdf.iterrows():
            cls_id = classify_by_name(row[name_col])
            class_counts[cls_id] += 1
            if cls_id > 0:  # Skip background in rasterization (it's the default)
                shapes.append((row.geometry, cls_id))
        
        print(f"    Classification breakdown:")
        for cls_id in sorted(class_counts.keys()):
            name = CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else f"class_{cls_id}"
            marker = " " if cls_id == 1 else ""
            print(f"      {name:<16}: {class_counts[cls_id]:>6} parcels{marker}")
    
    print(f"    Rasterizing {len(shapes)} non-background parcels...")
    
    label_raster = np.zeros((height, width), dtype=np.uint8)
    
    if shapes:
        rio_features.rasterize(
            shapes,
            out=label_raster,
            transform=transform,
            all_touched=True,
        )
    
    # Pixel stats
    unique, counts = np.unique(label_raster, return_counts=True)
    for u, c in zip(unique, counts):
        pct = c / label_raster.size * 100
        print(f"      Class {u} ({CLASS_NAMES[u]}): {c:,} px ({pct:.1f}%)")
    
    return label_raster


# ===============================================================
# STEP 3: STACK SEASONS + CUT PATCHES
# ===============================================================

def find_matching_tiles(imagery_dir):
    """
    GEE split each season into 4 tiles. Match them by pixel offset.
    Returns dict: tile_key -> {season: filepath}
    """
    files = sorted(Path(imagery_dir).glob("*.tif"))
    
    tiles = {}
    for f in files:
        name = f.stem
        # Extract season and tile offset
        # Format: es_j_2023_<season>_2023-<row>-<col>[-suffix]
        parts = name.split("-")
        if len(parts) >= 3:
            # tile_key = row-col part
            tile_key = f"{parts[-3] if len(parts) >= 4 else parts[-2]}-{parts[-2] if len(parts) >= 4 else parts[-1]}"
            # Actually let's use the pixel offsets
            # The files are named like: es_j_2023_fall_2023-0000000000-0000000000-003.tif
            row_offset = parts[1] if len(parts) >= 3 else "0"
            col_offset = parts[2] if len(parts) >= 4 else parts[1]
            # Clean col_offset (remove suffix like -003)
            if not col_offset.isdigit():
                col_offset = col_offset.split("-")[0] if "-" in col_offset else col_offset
            
            tile_key = f"{row_offset}_{col_offset}"
        else:
            tile_key = name
        
        # Detect season
        for season in ["spring", "summer", "fall", "winter"]:
            if season in name:
                if tile_key not in tiles:
                    tiles[tile_key] = {}
                tiles[tile_key][season] = str(f)
                break
    
    # Group by actual pixel offsets by reading file bounds
    import rasterio
    
    # Re-group by geographic bounds (more reliable than filename parsing)
    bound_groups = {}
    for f in files:
        with rasterio.open(str(f)) as src:
            b = src.bounds
            key = f"{int(b.left)}_{int(b.bottom)}"
        
        for season in ["spring", "summer", "fall", "winter"]:
            if season in f.name:
                if key not in bound_groups:
                    bound_groups[key] = {}
                bound_groups[key][season] = str(f)
                break
    
    # Keep only tiles with all 4 seasons
    complete = {k: v for k, v in bound_groups.items() if len(v) == 4}
    
    print(f"  Found {len(files)} GeoTIFFs")
    print(f"  Grouped into {len(bound_groups)} spatial tiles")
    print(f"  {len(complete)} tiles have all 4 seasons")
    
    return complete


def stack_and_patch_tile(tile_paths, label_raster_func, tile_idx, split_dir, 
                          patch_size=256, overlap=64, min_valid=0.5, max_single_class=0.98):
    """
    Stack 4 seasons -> 56 bands, overlay labels, cut patches.
    
    MEMORY-EFFICIENT: processes in horizontal strips instead of loading
    the entire tile (~17 GB) into memory at once.
    Each strip is ~500 MB = very manageable.
    """
    import rasterio
    from rasterio.windows import Window
    
    season_order = ["spring", "summer", "fall", "winter"]
    
    # Get tile dimensions from the first season file
    reference_path = tile_paths[season_order[0]]
    with rasterio.open(reference_path) as src:
        H = src.height
        W = src.width
        n_bands_per_season = src.count
    
    total_bands = n_bands_per_season * len(season_order)
    print(f"    Tile size: {H} x {W}, {total_bands} total bands (will process in strips)")
    
    # Get labels for this tile (label raster is uint8, so only ~38 MB for 8960x8960)
    label = label_raster_func(reference_path)
    if label.shape[0] > H:
        label = label[:H, :]
    if label.shape[1] > W:
        label = label[:, :W]
    
    # Open all season files for windowed reading
    season_files = []
    for season in season_order:
        season_files.append(rasterio.open(tile_paths[season]))
    
    # Process in horizontal strips
    stride = patch_size - overlap
    patch_count = 0
    olive_patch_count = 0
    
    try:
        for y in range(0, H - patch_size + 1, stride):
            # Read a strip of height=patch_size from each season
            window = Window(0, y, W, patch_size)
            
            strip_bands = []
            for sf in season_files:
                band_data = sf.read(window=window).astype(np.float32)  # (14, patch_size, W)
                np.nan_to_num(band_data, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
                strip_bands.append(band_data)
            
            # Stack: (56, patch_size, W)
            strip = np.concatenate(strip_bands, axis=0)
            
            # Corresponding label strip
            lbl_strip = label[y:y+patch_size, :W]
            
            # Cut patches along width
            for x in range(0, W - patch_size + 1, stride):
                img_patch = strip[:, :, x:x+patch_size]
                lbl_patch = lbl_strip[:, x:x+patch_size]
                
                # Quality filter 1: enough valid pixels
                valid = np.any(img_patch != 0, axis=0)
                if valid.sum() / (patch_size * patch_size) < min_valid:
                    continue
                
                # Quality filter 2: not too homogeneous (skip all-background)
                unique, counts = np.unique(lbl_patch, return_counts=True)
                if counts.max() / (patch_size * patch_size) > max_single_class:
                    continue
                
                # Save patch
                patch_name = f"jaen_t{tile_idx}_{patch_count:06d}"
                np.save(os.path.join(split_dir, f"{patch_name}_image.npy"), img_patch)
                np.save(os.path.join(split_dir, f"{patch_name}_mask.npy"), lbl_patch)
                
                patch_count += 1
                if 1 in unique:
                    olive_patch_count += 1
            
            # Progress every ~10 strips
            if (y // stride) % 10 == 0:
                print(f"      Strip y={y}/{H}: {patch_count} patches so far ({olive_patch_count} olive)")
    
    finally:
        for sf in season_files:
            sf.close()
    
    print(f"    -> {patch_count} patches ({olive_patch_count} contain olive groves)")
    return patch_count, olive_patch_count


# ===============================================================
# STEP 4: PYTORCH DATASET
# ===============================================================

class PatchDataset(Dataset):
    """Simple dataset loading .npy patch files."""
    
    def __init__(self, patches_dir, normalize_stats=None, augment=False):
        self.patches_dir = patches_dir
        self.normalize_stats = normalize_stats
        self.augment = augment
        
        # Find all patch files
        self.image_files = sorted(Path(patches_dir).glob("*_image.npy"))
        self.mask_files = [str(f).replace("_image.npy", "_mask.npy") for f in self.image_files]
        
        print(f"    Loaded {len(self.image_files)} patches from {patches_dir}")
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        image = np.load(self.image_files[idx]).astype(np.float32)  # (C, H, W)
        mask = np.load(self.mask_files[idx]).astype(np.int64)      # (H, W)
        
        # Normalize
        if self.normalize_stats is not None:
            mean = self.normalize_stats["mean"][:, None, None]
            std = self.normalize_stats["std"][:, None, None]
            image = (image - mean) / (std + 1e-8)
        
        # Simple augmentation (no external dependencies needed)
        if self.augment:
            # Random horizontal flip
            if np.random.random() > 0.5:
                image = np.flip(image, axis=2).copy()
                mask = np.flip(mask, axis=1).copy()
            # Random vertical flip
            if np.random.random() > 0.5:
                image = np.flip(image, axis=1).copy()
                mask = np.flip(mask, axis=0).copy()
            # Random 90 rotation
            k = np.random.randint(0, 4)
            if k > 0:
                image = np.rot90(image, k, axes=(1, 2)).copy()
                mask = np.rot90(mask, k, axes=(0, 1)).copy()
        
        return {
            "image": torch.from_numpy(image),
            "mask": torch.from_numpy(mask),
        }
    
    def get_balanced_sampler(self):
        """Oversample patches containing olive groves."""
        weights = []
        for mask_file in self.mask_files:
            mask = np.load(mask_file)
            if 1 in mask:  # Contains olive
                weights.append(5.0)
            elif np.any(mask > 0):  # Contains some crop
                weights.append(2.0)
            else:
                weights.append(1.0)
        
        return WeightedRandomSampler(weights, len(weights), replacement=True)


# ===============================================================
# STEP 5: LOSS FUNCTIONS
# ===============================================================

class DiceFocalLoss(nn.Module):
    """Combined Dice + Focal loss with class weighting."""
    
    def __init__(self, num_classes=7, class_weights=None, gamma=2.0):
        super().__init__()
        self.num_classes = num_classes
        self.gamma = gamma
        self.weights = torch.tensor(class_weights or [1.0]*num_classes, dtype=torch.float32)
    
    def forward(self, logits, targets):
        # --- Focal Loss ---
        ce = F.cross_entropy(logits, targets, weight=self.weights.to(logits.device), reduction="none")
        probs = F.softmax(logits, dim=1)
        targets_clean = targets.clone().clamp(0, self.num_classes - 1)
        pt = probs.gather(1, targets_clean.unsqueeze(1)).squeeze(1)
        focal = ((1 - pt) ** self.gamma) * ce
        focal_loss = focal.mean()
        
        # --- Dice Loss ---
        one_hot = F.one_hot(targets_clean, self.num_classes).permute(0, 3, 1, 2).float()
        dims = (0, 2, 3)
        intersection = (probs * one_hot).sum(dim=dims)
        cardinality = (probs + one_hot).sum(dim=dims)
        dice_per_class = (2 * intersection + 1) / (cardinality + 1)
        dice_loss = 1 - (dice_per_class * self.weights.to(logits.device)).sum() / self.weights.sum()
        
        return 0.5 * dice_loss + 0.5 * focal_loss


# ===============================================================
# MAIN PIPELINE
# ===============================================================

def main():
    parser = argparse.ArgumentParser(description="ZitounEye Phase 1: Train on Jaen")
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--loss", choices=["dice_focal", "ultimate"], default="dice_focal")
    parser.add_argument("--debug", action="store_true", help="Quick 2-epoch test")
    parser.add_argument("--skip-patching", action="store_true", help="Skip to training if patches exist")
    args = parser.parse_args()
    
    if args.debug:
        args.epochs = 2
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("=" * 60)
    print("  ZitounEye - Phase 1: Train on Jaen")
    print("=" * 60)
    print(f"  Device:     {device}")
    print(f"  Epochs:     {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Loss:       {args.loss}")
    
    # --- STEP 1: Crop mapping --------------------------------
    code_to_class = build_crop_mapping()
    
    if not args.skip_patching:
        # --- STEP 2: Find and match tiles --------------------
        print("\n" + "=" * 60)
        print("STEP 2: Matching seasonal tiles")
        print("=" * 60)
        
        tile_dict = find_matching_tiles(IMAGERY_DIR)
        
        if not tile_dict:
            print("   No complete tiles found! Check your imagery folder.")
            return
        
        # --- STEP 3: Find shapefile --------------------------
        print("\n" + "=" * 60)
        print("STEP 3: Loading shapefile")
        print("=" * 60)
        
        # Find the .shp file
        shp_files = list(SHAPEFILES_DIR.glob("*.shp"))
        if not shp_files:
            # Try one level up or down
            shp_files = list(SHAPEFILES_DIR.parent.glob("es_j_2023/*.shp"))
        if not shp_files:
            shp_files = list(SHAPEFILES_DIR.glob("**/*.shp"))
        
        if not shp_files:
            print(f"   No shapefile found in {SHAPEFILES_DIR}")
            print(f"  Available dirs: {[d.name for d in SHAPEFILES_DIR.parent.iterdir() if d.is_dir()]}")
            return
        
        shp_path = str(shp_files[0])
        print(f"  Shapefile: {shp_path}")
        
        # --- STEP 4: Rasterize + Patch ----------------------
        print("\n" + "=" * 60)
        print("STEP 4: Rasterizing labels & cutting patches")
        print("=" * 60)
        
        # 80/20 train/val split by tile
        tile_keys = list(tile_dict.keys())
        n_train = max(1, int(len(tile_keys) * 0.75))
        train_tiles = tile_keys[:n_train]
        val_tiles = tile_keys[n_train:]
        
        # If we only have 4 tiles, use 3 for train, 1 for val
        if len(tile_keys) == 4:
            train_tiles = tile_keys[:3]
            val_tiles = tile_keys[3:]
        
        print(f"  Train tiles: {len(train_tiles)}, Val tiles: {len(val_tiles)}")
        
        # Create output dirs
        train_dir = PATCHES_DIR / "train"
        val_dir = PATCHES_DIR / "val"
        train_dir.mkdir(parents=True, exist_ok=True)
        val_dir.mkdir(parents=True, exist_ok=True)
        
        total_train = 0
        total_val = 0
        total_olive = 0
        
        def make_label_func(shapefile, code_map):
            """Create a label rasterization function for a reference tif."""
            def rasterize(ref_tif_path):
                return rasterize_labels_for_tile(shapefile, ref_tif_path, code_map)
            return rasterize
        
        label_func = make_label_func(shp_path, code_to_class)
        
        for i, tile_key in enumerate(tile_keys):
            split = "train" if tile_key in train_tiles else "val"
            out_dir = train_dir if split == "train" else val_dir
            
            print(f"\n  Tile {i+1}/{len(tile_keys)} ({tile_key}) -> {split}")
            
            n_patches, n_olive = stack_and_patch_tile(
                tile_paths=tile_dict[tile_key],
                label_raster_func=label_func,
                tile_idx=i,
                split_dir=str(out_dir),
                patch_size=PATCH_SIZE,
                overlap=PATCH_OVERLAP,
            )
            
            if split == "train":
                total_train += n_patches
            else:
                total_val += n_patches
            total_olive += n_olive
        
        print(f"\n   Patching complete!")
        print(f"     Train: {total_train} patches")
        print(f"     Val:   {total_val} patches")
        print(f"     Olive: {total_olive} patches contain olive groves")
        
        # --- Compute normalization stats ---------------------
        print("\n  Computing normalization statistics...")
        
        train_files = sorted((PATCHES_DIR / "train").glob("*_image.npy"))
        if train_files:
            # Sample up to 500 patches for speed
            sample_files = train_files[::max(1, len(train_files) // 500)]
            
            running_sum = None
            running_sq_sum = None
            count = 0
            
            for f in sample_files:
                data = np.load(f).astype(np.float64)
                if running_sum is None:
                    C = data.shape[0]
                    running_sum = np.zeros(C)
                    running_sq_sum = np.zeros(C)
                
                # Per-channel stats
                for c in range(C):
                    valid = data[c][data[c] != 0]
                    if len(valid) > 0:
                        running_sum[c] += valid.sum()
                        running_sq_sum[c] += (valid ** 2).sum()
                        count += len(valid)
            
            count_per_channel = count / C
            mean = running_sum / max(count_per_channel, 1)
            std = np.sqrt(running_sq_sum / max(count_per_channel, 1) - mean ** 2)
            std = np.maximum(std, 1e-6)
            
            stats_path = PATCHES_DIR / "normalize_stats.npz"
            np.savez(stats_path, mean=mean.astype(np.float32), std=std.astype(np.float32))
            print(f"  Saved: {stats_path} ({len(mean)} channels)")
    
    # --- STEP 5: Load datasets -------------------------------
    print("\n" + "=" * 60)
    print("STEP 5: Loading datasets")
    print("=" * 60)
    
    # Load normalization
    stats_path = PATCHES_DIR / "normalize_stats.npz"
    normalize_stats = None
    if stats_path.exists():
        stats = np.load(stats_path)
        normalize_stats = {"mean": stats["mean"], "std": stats["std"]}
        print(f"  Normalization: {len(stats['mean'])} channels")
    
    train_dataset = PatchDataset(str(PATCHES_DIR / "train"), normalize_stats, augment=True)
    val_dataset = PatchDataset(str(PATCHES_DIR / "val"), normalize_stats, augment=False)
    
    if len(train_dataset) == 0:
        print("   No training patches found!")
        return
    
    train_sampler = train_dataset.get_balanced_sampler()
    
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, sampler=train_sampler,
        num_workers=4, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=4, pin_memory=True,
    )
    
    # Detect input channels
    sample = train_dataset[0]["image"]
    in_channels = sample.shape[0]
    print(f"  Input channels: {in_channels}")
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches:   {len(val_loader)}")
    
    # --- STEP 6: Build model ---------------------------------
    print("\n" + "=" * 60)
    print("STEP 6: Building model")
    print("=" * 60)
    
    import segmentation_models_pytorch as smp
    
    model = smp.Unet(
        encoder_name="efficientnet-b4",
        encoder_weights="imagenet",
        in_channels=in_channels,
        classes=NUM_CLASSES,
    )
    model = model.to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Architecture: U-Net + EfficientNet-B4")
    print(f"  Parameters: {total_params:,}")
    print(f"  Input: {in_channels} channels -> Output: {NUM_CLASSES} classes")
    
    # Loss
    criterion = DiceFocalLoss(NUM_CLASSES, CLASS_WEIGHTS, gamma=2.0)
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    
    # Scheduler
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.lr, epochs=args.epochs,
        steps_per_epoch=len(train_loader), pct_start=0.1,
    )
    
    scaler = GradScaler()
    
    # --- STEP 7: Training loop -------------------------------
    print("\n" + "=" * 60)
    print("STEP 7: Training")
    print("=" * 60 + "\n")
    
    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    
    best_miou = 0.0
    best_olive_iou = 0.0
    patience = 15
    patience_counter = 0
    history = []
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    for epoch in range(args.epochs):
        t0 = time.time()
        
        # -- Train --
        model.train()
        train_loss = 0.0
        n_batches = 0
        
        for batch in train_loader:
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)
            
            optimizer.zero_grad()
            
            with autocast():
                outputs = model(images)
                loss = criterion(outputs, masks)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            
            train_loss += loss.item()
            n_batches += 1
        
        train_loss /= max(n_batches, 1)
        
        # -- Validate --
        model.eval()
        val_loss = 0.0
        val_batches = 0
        confusion = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
        
        with torch.no_grad():
            for batch in val_loader:
                images = batch["image"].to(device)
                masks = batch["mask"].to(device)
                
                with autocast():
                    outputs = model(images)
                    loss = criterion(outputs, masks)
                
                val_loss += loss.item()
                val_batches += 1
                
                preds = outputs.argmax(dim=1).cpu().numpy().ravel()
                targets = masks.cpu().numpy().ravel()
                
                # Vectorized confusion matrix update
                valid_mask = (targets >= 0) & (targets < NUM_CLASSES) & (preds >= 0) & (preds < NUM_CLASSES)
                np.add.at(confusion, (targets[valid_mask], preds[valid_mask]), 1)
        
        val_loss /= max(val_batches, 1)
        
        # Compute per-class IoU
        ious = []
        per_class_iou = {}
        for c in range(NUM_CLASSES):
            tp = confusion[c, c]
            fp = confusion[:, c].sum() - tp
            fn = confusion[c, :].sum() - tp
            iou = tp / max(tp + fp + fn, 1)
            ious.append(iou)
            per_class_iou[CLASS_NAMES[c]] = iou
        
        miou = np.mean(ious)
        olive_iou = per_class_iou["olive_grove"]
        vineyard_iou = per_class_iou["vineyard"]
        
        elapsed = time.time() - t0
        lr = optimizer.param_groups[0]["lr"]
        
        print(
            f"Epoch {epoch+1:>3}/{args.epochs} | "
            f"Loss: {train_loss:.4f}/{val_loss:.4f} | "
            f"mIoU: {miou:.4f} | "
            f" {olive_iou:.4f} | "
            f" {vineyard_iou:.4f} | "
            f"LR: {lr:.2e} | "
            f"{elapsed:.0f}s"
        )
        
        history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "miou": miou,
            "olive_iou": olive_iou,
            "vineyard_iou": vineyard_iou,
        })
        
        # Save best
        if miou > best_miou:
            best_miou = miou
            best_olive_iou = olive_iou
            patience_counter = 0
            
            best_path = CHECKPOINTS_DIR / f"phase1_{timestamp}_best.pth"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "miou": miou,
                "olive_iou": olive_iou,
                "config": {
                    "model": {"tier1": {
                        "architecture": "Unet",
                        "encoder": "efficientnet-b4",
                        "in_channels": in_channels,
                        "classes": NUM_CLASSES,
                    }},
                    "classes": {"num_classes": NUM_CLASSES, "names": CLASS_NAMES},
                }
            }, str(best_path))
            print(f"  * Best mIoU: {best_miou:.4f} (olive: {best_olive_iou:.4f}) -> {best_path.name}")
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            print(f"\n   Early stopping (no improvement for {patience} epochs)")
            break
    
    # --- RESULTS ---------------------------------------------
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"  Best mIoU:       {best_miou:.4f}")
    print(f"  Best Olive IoU:  {best_olive_iou:.4f}")
    print(f"  Checkpoint:      {best_path}")
    
    # Save history
    history_path = PROJECT_ROOT / "logs" / f"phase1_{timestamp}_history.json"
    history_path.parent.mkdir(parents=True, exist_ok=True)
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"  History:         {history_path}")
    
    # Print final confusion for olive
    print(f"\n  Per-class IoU:")
    for c in range(NUM_CLASSES):
        marker = " <- TARGET" if c == 1 else ""
        print(f"    {CLASS_NAMES[c]:<16}: {ious[c]:.4f}{marker}")
    
    print(f"\n   Olive confusion breakdown:")
    olive_total = confusion[1, :].sum()
    if olive_total > 0:
        for c in range(NUM_CLASSES):
            if confusion[1, c] > 0:
                pct = confusion[1, c] / olive_total * 100
                print(f"    Olive predicted as {CLASS_NAMES[c]:<16}: {pct:.1f}%")
    
    print(f"\n{'='*60}")
    print(f" Phase 1 complete!")
    print(f"{'='*60}")
    print(f"\nNext: Add more provinces and run Phase 2:")
    print(f"  python scripts/04_train_model.py --tier 1 --loss ultimate")


if __name__ == "__main__":
    main()
