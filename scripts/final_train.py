"""
ZitounEye - FINAL MODEL: Jaen + Cordoba + Sevilla + OSM Buildings/Roads
========================================================================
The best olive grove detector, trained on 3 Andalusian provinces with
full 8-class labels including OSM-derived buildings and roads.

Classes:
    0 = background      (forest, bare soil, water, etc.)
    1 = olive_grove     *** PRIMARY TARGET ***
    2 = vineyard
    3 = other_orchard   (almond, citrus, etc.)
    4 = arable_crop     (wheat, sunflower, etc.)
    5 = building        (from OpenStreetMap)
    6 = road            (from OpenStreetMap)

Usage:
    python scripts/final_train.py               (full 100-epoch run)
    python scripts/final_train.py --debug       (2 epochs, quick test)
    python scripts/final_train.py --skip-patching  (reuse existing patches)
    python scripts/final_train.py --skip-patching --resume latest
    python scripts/final_train.py --skip-patching --resume checkpoints/final_XXXXX_best.pth
"""

import sys, os, glob, argparse, warnings, json, time
from datetime import datetime
from collections import Counter



warnings.filterwarnings("ignore")

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

# ================================================================
# CONFIGURATION
# ================================================================
PROJECT_ROOT = r"c:\Users\AORUS\Documents\zitoun_eye"

PROVINCE_CONFIGS = {
    "es_j_2023": {
        "imagery_dir": os.path.join(PROJECT_ROOT, "spain_j_images"),
        "shapefile":   os.path.join(PROJECT_ROOT, "ES_2023", "ES_2023", "es_j_2023", "es_j_2023.shp"),
        "osm_bbox":    (37.39, -4.12, 38.59, -2.50),   # (south, west, north, east)
        "split":       "train",
        "val_fraction": 0.25,
    },
    "es_co_2023": {
        "imagery_dir": os.path.join(PROJECT_ROOT, "spain_phase2_images"),
        "shapefile":   os.path.join(PROJECT_ROOT, "ES_2023", "ES_2023", "es_co_2023", "es_co_2023.shp"),
        "osm_bbox":    (37.24, -5.65, 38.73, -4.07),
        "split":       "train",
        "val_fraction": 0.25,
    },
    "es_se_2023": {
        "imagery_dir": os.path.join(PROJECT_ROOT, "spain_phase2_images"),
        "shapefile":   os.path.join(PROJECT_ROOT, "ES_2023", "ES_2023", "es_se_2023", "es_se_2023.shp"),
        "osm_bbox":    (36.84, -6.54, 38.02, -4.86),
        "split":       "val",     # Sevilla = validation province
        "val_fraction": 1.0,
    },
}

PATCHES_DIR   = os.path.join(PROJECT_ROOT, "data", "patches_final")
CHECKPOINTS   = os.path.join(PROJECT_ROOT, "checkpoints")
LOGS          = os.path.join(PROJECT_ROOT, "logs")

NUM_CLASSES   = 7
MODEL_BANDS   = 56      # 14 bands x 4 seasons
PATCH_SIZE    = 256
OVERLAP       = 64
BATCH_SIZE    = 16
EPOCHS        = 100
LR            = 3e-4
WEIGHT_DECAY  = 1e-4
EARLY_STOP    = 20

CLASS_NAMES = [
    "background", "olive_grove", "vineyard",
    "other_orchard", "arable_crop", "building", "road"
]

# Olive gets 5x weight, vineyard/orchard 2x, rest normal, background minimal
CLASS_WEIGHTS = torch.tensor([0.1, 5.0, 2.0, 2.0, 1.0, 1.5, 2.0], dtype=torch.float32)

SEASON_ORDER = ["spring", "summer", "fall", "winter"]

# ================================================================
# CLASSIFICATION HELPERS
# ================================================================
OLIVE_KW   = ["olive", "olea", "olivo"]
VINE_KW    = ["vineyard", "vine", "grape", "vitis"]
ORCHARD_KW = ["almond", "citrus", "orange", "lemon", "fig", "carob",
               "cherry", "peach", "apple", "pear", "plum", "walnut",
               "pistachio", "avocado", "mango", "tropical", "fruit",
               "orchard", "nut_tree", "pomegranate", "plantation", "permanent_crop"]
ARABLE_KW  = ["wheat", "barley", "maize", "corn", "rice", "sunflower",
               "cotton", "sugar", "potato", "tomato", "legume", "bean",
               "pea", "soybean", "rape", "canola", "oat", "rye",
               "cereal", "grain", "arable", "annual", "vegetable",
               "forage", "hay", "clover", "alfalfa", "sorghum",
               "tobacco", "lavender", "artichoke", "pepper", "melon",
               "onion", "fallow", "horticultur", "flower", "nursery"]


def classify_by_name(name_str):
    if not name_str or str(name_str) == "nan":
        return 0
    n = str(name_str).lower().strip()
    if any(kw in n for kw in OLIVE_KW):   return 1
    if any(kw in n for kw in VINE_KW):    return 2
    if any(kw in n for kw in ORCHARD_KW):
        if "olive" not in n:               return 3
    if any(kw in n for kw in ARABLE_KW):  return 4
    return 0


# ================================================================
# OSM BUILDINGS + ROADS
# ================================================================
_OSM_BUILDINGS = None
_OSM_ROADS = None
_OSM_LOADED = False

def get_osm_layers(bbox_wsne, crs_str):
    """
    Load buildings and roads from local Geofabrik shapefiles for the given bbox.
    Uses global in-memory caching for lightning-fast cx lookups.
    """
    global _OSM_BUILDINGS, _OSM_ROADS, _OSM_LOADED
    import geopandas as gpd
    import os
    import warnings
    warnings.filterwarnings("ignore", message="PROJ: proj_identify")
    
    if not _OSM_LOADED:
        print("        Loading Geofabrik shapefiles into memory (this will take ~10 seconds)...")
        bld_path = os.path.join(PROJECT_ROOT, "data", "osm", "gis_osm_buildings_a_free_1.shp")
        rds_path = os.path.join(PROJECT_ROOT, "data", "osm", "gis_osm_roads_free_1.shp")
        
        if os.path.exists(bld_path) and os.path.exists(rds_path):
            _OSM_BUILDINGS = gpd.read_file(bld_path)
            _OSM_ROADS = gpd.read_file(rds_path)
            
            # Filter roads globally to save time later
            valid_roads = ['motorway', 'trunk', 'primary', 'secondary', 'tertiary', 'unclassified', 'residential', 'motorway_link', 'trunk_link', 'primary_link', 'secondary_link', 'tertiary_link']
            _OSM_ROADS = _OSM_ROADS[_OSM_ROADS['fclass'].isin(valid_roads)]
            
            # Force spatial index generation
            _ = _OSM_BUILDINGS.sindex
            _ = _OSM_ROADS.sindex
        else:
            print("        ERROR: Local Geofabrik shapefiles not found in data/osm. Skipping.")
        _OSM_LOADED = True

    if _OSM_BUILDINGS is None or _OSM_ROADS is None:
        return None, None

    west, south, east, north = bbox_wsne
    
    # Use fast in-memory spatial index bounding box query
    try:
        buildings = _OSM_BUILDINGS.cx[west:east, south:north].copy()
        if len(buildings) > 0:
            buildings = buildings.to_crs(crs_str)
        else:
            buildings = None
    except Exception as e:
        buildings = None
        
    try:
        roads = _OSM_ROADS.cx[west:east, south:north].copy()
        if len(roads) > 0:
            roads = roads.to_crs(crs_str)
        else:
            roads = None
    except Exception as e:
        roads = None

    return buildings, roads


# ================================================================
# TILE MATCHING
# ================================================================
def match_tiles(imagery_dir, province_key):
    """Find all complete tile groups (all 4 seasons present)."""
    all_tifs = glob.glob(os.path.join(imagery_dir, f"{province_key}*.tif"))
    if not all_tifs:
        return {}

    # Group by spatial offset key
    tile_map = {}
    for path in all_tifs:
        fname = os.path.basename(path).lower()
        season = None
        for s in SEASON_ORDER:
            if f"_{s}_" in fname:
                season = s
                break
        if season is None:
            continue

        # Extract offset from filename e.g. ...-0000000000-0000008960-001.tif
        parts = fname.replace(".tif", "").split("-")
        if len(parts) >= 3:
            offset = f"{parts[-3]}_{parts[-2]}"
        else:
            offset = "0_0"

        tile_map.setdefault(offset, {})[season] = path

    complete = {k: v for k, v in tile_map.items() if len(v) == 4}
    incomplete = {k: v for k, v in tile_map.items() if len(v) < 4}

    for k, v in incomplete.items():
        missing = [s for s in SEASON_ORDER if s not in v]
        print(f"      Warning: tile {k} missing seasons {missing}")

    return complete


# ================================================================
# RASTERIZATION WITH OSM OVERLAY
# ================================================================
def rasterize_tile(shapefile_path, reference_tif_path, no_osm=False):
    """
    Rasterize EuroCrop parcels + OSM buildings + OSM roads.
    Priority: roads > buildings > crops > background
    """
    import rasterio
    from rasterio import features as rio_features
    import geopandas as gpd
    from pyproj import Transformer
    from shapely.geometry import box

    with rasterio.open(reference_tif_path) as src:
        transform = src.transform
        H, W = src.height, src.width
        crs    = src.crs
        bounds = src.bounds

    # Convert UTM bounds to WGS84 for shapefile bbox filter
    try:
        xf = Transformer.from_crs(str(crs), "EPSG:4258", always_xy=True)
        minx, miny = xf.transform(bounds.left, bounds.bottom)
        maxx, maxy = xf.transform(bounds.right, bounds.top)
        bbox4258 = (minx, miny, maxx, maxy)
    except Exception:
        bbox4258 = (bounds.left, bounds.bottom, bounds.right, bounds.top)

    # --- Rasterize EuroCrop crops ---
    gdf = gpd.read_file(shapefile_path, bbox=bbox4258)
    if str(gdf.crs) != str(crs):
        gdf = gdf.to_crs(crs)
    gdf = gdf[gdf.geometry.is_valid & ~gdf.geometry.is_empty]

    name_col = next((c for c in ["EC_hcat_n","ec_hcat_n","EC_trans_n"] if c in gdf.columns), None)
    label = np.zeros((H, W), dtype=np.uint8)

    if name_col and len(gdf) > 0:
        crop_shapes = []
        counts = Counter()
        for _, row in gdf.iterrows():
            cls = classify_by_name(row[name_col])
            counts[cls] += 1
            if cls > 0:
                crop_shapes.append((row.geometry, cls))

        for c in sorted(counts):
            if counts[c] > 0:
                print(f"        {CLASS_NAMES[c]}: {counts[c]} parcels")

        if crop_shapes:
            rio_features.rasterize(crop_shapes, out=label,
                                   transform=transform, all_touched=True)

    # --- Overlay OSM Buildings + Roads using a 4x4 Grid for safety & speed ---
    if not no_osm:
        try:
            grid_n = 4
            transformer = Transformer.from_crs(str(crs), "EPSG:4326", always_xy=True)

            for row in range(grid_n):
                for col in range(grid_n):
                    ymin = row * (H // grid_n)
                    ymax = (row + 1) * (H // grid_n) if row < grid_n - 1 else H
                    xmin = col * (W // grid_n)
                    xmax = (col + 1) * (W // grid_n) if col < grid_n - 1 else W

                    # Optimization: only download OSM if there are crop labels in this cell
                    cell_crop_mask = label[ymin:ymax, xmin:xmax]
                    if not np.any(cell_crop_mask > 0):
                        continue

                    # Calculate coordinate bounds for this cell in UTM
                    x_min_coord, y_max_coord = transform * (xmin, ymin)
                    x_max_coord, y_min_coord = transform * (xmax, ymax)

                    cell_left = x_min_coord
                    cell_right = x_max_coord
                    cell_bottom = y_min_coord
                    cell_top = y_max_coord

                    # Convert to WGS84
                    w_lon, s_lat = transformer.transform(cell_left, cell_bottom)
                    e_lon, n_lat = transformer.transform(cell_right, cell_top)

                    west = min(w_lon, e_lon)
                    east = max(w_lon, e_lon)
                    south = min(s_lat, n_lat)
                    north = max(s_lat, n_lat)

                    print(f"      Grid Cell [{row},{col}]: downloading OSM layers...")
                    osm_bld, osm_rds = get_osm_layers((west, south, east, north), str(crs))

                    # Define cell transform for local rasterization
                    cell_transform = rasterio.transform.from_origin(cell_left, cell_top, transform.a, -transform.e)

                    # Create overlay masks for this cell
                    cell_label = label[ymin:ymax, xmin:xmax]
                    cell_box = box(cell_left, cell_bottom, cell_right, cell_top)

                    # Overlay Buildings (class 5)
                    if osm_bld is not None and len(osm_bld) > 0:
                        blds = osm_bld[osm_bld.intersects(cell_box)]
                        if len(blds) > 0:
                            bld_shapes = [(geom, 5) for geom in blds.geometry if geom.is_valid]
                            if bld_shapes:
                                rio_features.rasterize(bld_shapes, out=cell_label,
                                                       transform=cell_transform, all_touched=True)

                    # Overlay Roads (class 6)
                    if osm_rds is not None and len(osm_rds) > 0:
                        rds = osm_rds[osm_rds.intersects(cell_box)]
                        if len(rds) > 0:
                            rds_buf = rds.copy()
                            rds_buf["geometry"] = rds_buf.geometry.buffer(10)
                            rd_shapes = [(geom, 6) for geom in rds_buf.geometry if geom.is_valid]
                            if rd_shapes:
                                rio_features.rasterize(rd_shapes, out=cell_label,
                                                       transform=cell_transform, all_touched=True)

                    label[ymin:ymax, xmin:xmax] = cell_label
        except Exception as e:
            print(f"      Failed to overlay OSM layers: {e}")

    # Stats
    u, c = np.unique(label, return_counts=True)
    for ui, ci in zip(u, c):
        print(f"        -> Class {ui} ({CLASS_NAMES[ui]}): {ci/label.size*100:.1f}% of pixels")

    return label


# ================================================================
# STRIP-BASED PATCH CUTTING
# ================================================================
def cut_patches_strip(tile_paths, label, tile_idx, out_dir,
                      patch_size=PATCH_SIZE, overlap=OVERLAP,
                      min_valid=0.5, max_bg=0.99):
    import rasterio
    from rasterio.windows import Window

    ref_path = tile_paths[SEASON_ORDER[0]]
    with rasterio.open(ref_path) as src:
        H, W = src.height, src.width

    label = label[:H, :W]
    srcs  = [rasterio.open(tile_paths[s]) for s in SEASON_ORDER]
    stride = patch_size - overlap
    n_total = 0
    n_olive = 0

    try:
        for y in range(0, H - patch_size + 1, stride):
            window = Window(0, y, W, patch_size)
            strip_list = []
            for sf in srcs:
                bd = sf.read(window=window).astype(np.float32)
                np.nan_to_num(bd, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
                strip_list.append(bd)
            strip     = np.concatenate(strip_list, axis=0)  # (56, ps, W)
            lbl_strip = label[y:y+patch_size, :W]

            for x in range(0, W - patch_size + 1, stride):
                img_p = strip[:, :, x:x+patch_size]
                lbl_p = lbl_strip[:, x:x+patch_size]

                # Filter: enough valid pixels
                if np.any(img_p != 0, axis=0).sum() / patch_size**2 < min_valid:
                    continue
                # Filter: not all background
                u, c = np.unique(lbl_p, return_counts=True)
                if c.max() / patch_size**2 > max_bg:
                    continue

                name = f"final_t{tile_idx}_{n_total:06d}"
                img_path = os.path.join(out_dir, f"{name}_image.npy")
                msk_path = os.path.join(out_dir, f"{name}_mask.npy")

                # Write to temp files first, then atomically rename into place.
                # This prevents a half-written .npy from ever being visible to
                # the dataloader if the process is killed/crashes mid-save, or
                # if a sync client (OneDrive) opens the file while it's being
                # written (the cause of the earlier truncated-read crash).
                tmp_img_base = os.path.join(out_dir, f".tmp_{name}_image")
                tmp_msk_base = os.path.join(out_dir, f".tmp_{name}_mask")
                np.save(tmp_img_base, img_p)   # numpy appends ".npy" automatically
                np.save(tmp_msk_base, lbl_p)
                os.replace(tmp_img_base + ".npy", img_path)
                os.replace(tmp_msk_base + ".npy", msk_path)

                n_total += 1
                if 1 in u:
                    n_olive += 1

            if (y // stride) % 10 == 0:
                print(f"      Strip y={y}/{H}: {n_total} patches ({n_olive} olive)")
    finally:
        for sf in srcs:
            sf.close()

    return n_total, n_olive


# ================================================================
# DATASET
# ================================================================
class PatchDataset(Dataset):
    def __init__(self, patch_dir, mean, std, augment=False):
        self.files   = sorted(glob.glob(os.path.join(patch_dir, "*_image.npy")))
        self.mean    = mean[:, np.newaxis, np.newaxis]
        self.std     = std[:, np.newaxis, np.newaxis]
        self.augment = augment

    def __len__(self): return len(self.files)

    def __getitem__(self, idx, _retries=3):
        try:
            img = np.load(self.files[idx]).astype(np.float32)
            msk = np.load(self.files[idx].replace("_image.npy", "_mask.npy")).astype(np.int64)
        except Exception as e:
            # Defensive fallback: a patch can occasionally fail to read due to
            # a transient filesystem/sync race (e.g. OneDrive touching the file
            # mid-read). Don't let one bad read kill a multi-hour training run -
            # log it and substitute a different random sample instead.
            print(f"  [WARN] Failed to load patch, skipping: {self.files[idx]} ({e})")
            if _retries <= 0:
                raise
            return self.__getitem__(np.random.randint(0, len(self.files)), _retries=_retries - 1)

        img = (img - self.mean) / self.std

        if self.augment:
            if np.random.rand() > 0.5:  # hflip
                img = np.flip(img, 2).copy(); msk = np.flip(msk, 1).copy()
            if np.random.rand() > 0.5:  # vflip
                img = np.flip(img, 1).copy(); msk = np.flip(msk, 0).copy()
            k = np.random.randint(4)    # rot90
            if k: img = np.rot90(img, k, (1,2)).copy(); msk = np.rot90(msk, k).copy()
            # Random brightness jitter (per-band scale)
            if np.random.rand() > 0.5:
                scale = np.random.uniform(0.85, 1.15, (MODEL_BANDS,1,1)).astype(np.float32)
                img = img * scale

        return torch.tensor(img), torch.tensor(msk)


def make_sampler(dataset):
    weights = []
    for f in dataset.files:
        msk = np.load(f.replace("_image.npy", "_mask.npy"))
        olive_frac = (msk == 1).mean()
        weights.append(1.0 + olive_frac * 9.0)
    return WeightedRandomSampler(torch.tensor(weights, dtype=torch.float32),
                                 len(dataset), replacement=True)


# ================================================================
# LOSS
# ================================================================
class OliveFocusedLoss(nn.Module):
    """Dice + Focal loss with strong olive weighting."""
    def __init__(self, class_weights, gamma=2.5, dice_w=0.6):
        super().__init__()
        self.cw     = class_weights
        self.gamma  = gamma
        self.dice_w = dice_w

    def forward(self, logits, targets):
        # Cross-entropy focal
        ce   = nn.functional.cross_entropy(logits, targets,
                                            weight=self.cw.to(logits.device))
        pt   = torch.exp(-ce)
        focal = (1 - pt) ** self.gamma * ce

        # Dice per class (macro)
        probs = torch.softmax(logits, dim=1)
        dice  = 0.0
        for c in range(NUM_CLASSES):
            p  = probs[:, c]
            t  = (targets == c).float()
            inter = (p * t).sum()
            dice += 1 - (2*inter + 1) / (p.sum() + t.sum() + 1)
        dice /= NUM_CLASSES

        return (1-self.dice_w) * focal + self.dice_w * dice


# ================================================================
# MAIN
# ================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug",          action="store_true")
    parser.add_argument("--skip-patching",  action="store_true")
    parser.add_argument("--epochs",         type=int, default=EPOCHS)
    parser.add_argument("--batch-size",     type=int, default=BATCH_SIZE)
    parser.add_argument("--no-osm",         action="store_true",
                        help="Skip OSM buildings/roads download")
    parser.add_argument("--resume",         type=str, default=None,
                        help="Path to a final_*_best.pth checkpoint to resume from, "
                             "or 'latest' to auto-pick the most recent one in checkpoints/")
    args = parser.parse_args()
    if args.debug: args.epochs = 2

    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("=" * 65)
    print("  ZitounEye - FINAL MODEL: Jaen + Cordoba + Sevilla + OSM")
    print("=" * 65)
    print(f"  Device:  {device}")
    print(f"  Epochs:  {args.epochs}")
    print(f"  OSM:     {'disabled' if args.no_osm else 'enabled (buildings + roads)'}")
    print()

    os.makedirs(os.path.join(PATCHES_DIR, "train"), exist_ok=True)
    os.makedirs(os.path.join(PATCHES_DIR, "val"),   exist_ok=True)
    os.makedirs(CHECKPOINTS, exist_ok=True)
    os.makedirs(LOGS,        exist_ok=True)

    # ============================================================
    # PATCHING
    # ============================================================
    if not args.skip_patching:
        print("=" * 65)
        print("STEP 1: Building patches from all 3 provinces")
        print("=" * 65)

        tile_counter = 0
        grand_train  = 0
        grand_val    = 0
        grand_olive  = 0

        for prov_key, cfg in PROVINCE_CONFIGS.items():
            print(f"\n>>> Province: {prov_key} (role={cfg['split']})")
            shp = cfg["shapefile"]
            img_dir = cfg["imagery_dir"]

            if not os.path.exists(shp):
                print(f"  Shapefile missing: {shp}"); continue

            # Match tiles
            tiles = match_tiles(img_dir, prov_key)
            print(f"  Matched {len(tiles)} complete tiles (4 seasons each)")
            if not tiles:
                print(f"  No tiles found in {img_dir}"); continue

            tile_list  = sorted(tiles.items())
            n_val_tiles = max(1, int(len(tile_list) * cfg["val_fraction"]))
            n_tr_tiles  = len(tile_list) - n_val_tiles

            # If province is forced to val, put all in val
            if cfg["split"] == "val":
                train_tiles = []
                val_tiles   = tile_list
            else:
                train_tiles = tile_list[:n_tr_tiles]
                val_tiles   = tile_list[n_tr_tiles:]

            for split_name, tile_subset in [("train", train_tiles), ("val", val_tiles)]:
                split_dir = os.path.join(PATCHES_DIR, split_name)
                for tile_key, tile_paths in tile_subset:
                    print(f"\n  Tile {tile_counter} ({tile_key}) -> {split_name}")
                    print(f"    Rasterizing labels + OSM overlay...")
                    label = rasterize_tile(shp, tile_paths[SEASON_ORDER[0]],
                                           no_osm=args.no_osm)
                    print(f"    Cutting patches...")
                    n_p, n_o = cut_patches_strip(tile_paths, label,
                                                  tile_counter, split_dir)
                    if split_name == "train":
                        grand_train += n_p
                    else:
                        grand_val   += n_p
                    grand_olive += n_o
                    tile_counter += 1

        print(f"\n  Patching complete!")
        print(f"    Train: {grand_train} patches")
        print(f"    Val:   {grand_val} patches")
        print(f"    Olive: {grand_olive} patches contain olive groves")

        # Compute normalization stats from training patches
        print("\n  Computing normalization stats...")
        train_dir   = os.path.join(PATCHES_DIR, "train")
        files_sample = glob.glob(os.path.join(train_dir, "*_image.npy"))
        np.random.shuffle(files_sample)
        files_sample = files_sample[:500]
        rsum  = np.zeros(MODEL_BANDS, dtype=np.float64)
        rsum2 = np.zeros(MODEL_BANDS, dtype=np.float64)
        npix  = 0
        for f in files_sample:
            img = np.load(f).astype(np.float64)
            rsum  += img.sum((1,2))
            rsum2 += (img**2).sum((1,2))
            npix  += img.shape[1] * img.shape[2]
        mean = (rsum / npix).astype(np.float32)
        std  = np.sqrt(np.maximum(rsum2/npix - (rsum/npix)**2, 1e-10)).astype(np.float32)
        stats_path = os.path.join(PATCHES_DIR, "normalize_stats.npz")
        np.savez(stats_path, mean=mean, std=std)
        print(f"  Saved: {stats_path}")

    else:
        print("Skipping patching (--skip-patching)")
        stats_path = os.path.join(PATCHES_DIR, "normalize_stats.npz")
        if not os.path.exists(stats_path):
            # Fall back to phase1 stats
            stats_path = os.path.join(PROJECT_ROOT, "data", "patches", "normalize_stats.npz")
            print(f"  Using Phase 1 stats as fallback: {stats_path}")

    # ============================================================
    # DATASET + DATALOADER
    # ============================================================
    print()
    print("=" * 65)
    print("STEP 2: Loading datasets")
    print("=" * 65)

    norm   = np.load(stats_path)
    mean_a = norm["mean"]
    std_a  = np.where(norm["std"] < 1e-6, 1.0, norm["std"])

    train_dir = os.path.join(PATCHES_DIR, "train")
    val_dir   = os.path.join(PATCHES_DIR, "val")

    if not os.path.exists(train_dir) or \
       not glob.glob(os.path.join(train_dir, "*.npy")):
        print("No training patches found. Run without --skip-patching.")
        return

    train_ds = PatchDataset(train_dir, mean_a, std_a, augment=True)
    val_ds   = PatchDataset(val_dir,   mean_a, std_a, augment=False)
    print(f"  Train: {len(train_ds)} patches")
    print(f"  Val:   {len(val_ds)} patches")

    sampler      = make_sampler(train_ds)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              sampler=sampler, num_workers=0, pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=args.batch_size,
                              shuffle=False, num_workers=0, pin_memory=True)

    # ============================================================
    # MODEL
    # ============================================================
    print()
    print("=" * 65)
    print("STEP 3: Building model")
    print("=" * 65)

    try:
        import segmentation_models_pytorch as smp
    except ImportError:
        raise ImportError("pip install segmentation-models-pytorch")

    model = smp.Unet(
        encoder_name="efficientnet-b4",
        encoder_weights=None,
        in_channels=MODEL_BANDS,
        classes=NUM_CLASSES,
    )

    # Fine-tune from best Phase 1 checkpoint (only used when NOT resuming a
    # final-stage run, since --resume loads full final-stage weights below)
    if not args.resume:
        phase1_ckpts = sorted(glob.glob(os.path.join(CHECKPOINTS, "phase1_*_best.pth")))
        if phase1_ckpts:
            ckpt_path = phase1_ckpts[-1]
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            state = ckpt.get("model_state_dict", ckpt)
            model.load_state_dict(state, strict=True)
            print(f"  Fine-tuning from: {os.path.basename(ckpt_path)}")
            print(f"  Phase 1 olive IoU was: {ckpt.get('olive_iou', '?'):.4f}")
        else:
            print("  No Phase 1 checkpoint found - training from scratch")

    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters: {n_params:,}")

    # ============================================================
    # RESUME (optional)
    # ============================================================
    start_epoch       = 1
    resume_best_olive = 0.0
    resume_best_miou  = 0.0
    optimizer_state   = None
    resume_ckpt_name  = None

    if args.resume:
        if args.resume == "latest":
            ckpts = sorted(glob.glob(os.path.join(CHECKPOINTS, "final_*_best.pth")))
            resume_path = ckpts[-1] if ckpts else None
            if resume_path is None:
                print("  No final_*_best.pth checkpoints found to resume from - "
                      "starting fresh instead.")
        else:
            # Accept either a bare filename or a full/relative path
            resume_path = args.resume
            if not os.path.isabs(resume_path) and not os.path.exists(resume_path):
                candidate = os.path.join(CHECKPOINTS, os.path.basename(resume_path))
                if os.path.exists(candidate):
                    resume_path = candidate

        if resume_path and os.path.exists(resume_path):
            print(f"  Resuming from: {resume_path}")
            rckpt = torch.load(resume_path, map_location="cpu", weights_only=False)
            model.load_state_dict(rckpt["model_state_dict"], strict=True)
            model = model.to(device)
            optimizer_state    = rckpt.get("optimizer_state_dict")
            start_epoch         = rckpt.get("epoch", 0) + 1
            resume_best_olive   = rckpt.get("olive_iou", 0.0)
            resume_best_miou    = rckpt.get("miou", 0.0)
            resume_ckpt_name    = os.path.basename(resume_path)
            print(f"  Resuming at epoch {start_epoch}, "
                  f"previous best olive IoU={resume_best_olive:.4f}, mIoU={resume_best_miou:.4f}")
        elif resume_path:
            print(f"  Resume checkpoint not found: {resume_path} - starting fresh instead.")

    # ============================================================
    # TRAINING
    # ============================================================
    print()
    print("=" * 65)
    print("STEP 4: Training")
    print("=" * 65)

    criterion = OliveFocusedLoss(CLASS_WEIGHTS)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    if optimizer_state is not None:
        try:
            optimizer.load_state_dict(optimizer_state)
            print("  Restored optimizer state from checkpoint.")
        except Exception as e:
            print(f"  Could not restore optimizer state ({e}) - using fresh optimizer.")

    remaining_epochs = max(1, args.epochs - start_epoch + 1)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=LR,
        steps_per_epoch=len(train_loader),
        epochs=remaining_epochs,   # schedule covers what's left when resuming
        pct_start=0.1,
    )
    scaler    = torch.amp.GradScaler()

    best_olive = resume_best_olive
    best_miou  = resume_best_miou
    no_improve = 0
    history    = []
    # Keep appending to the same checkpoint file when resuming, otherwise
    # start a new timestamped one.
    ckpt_name  = resume_ckpt_name if resume_ckpt_name else f"final_{timestamp}_best.pth"
    last_ious  = [0.0] * NUM_CLASSES

    print(f"{'Epoch':>6} | {'Train':>8} {'Val':>8} | {'mIoU':>6} {'Olive':>6} | {'LR':>10} | {'Time':>5}")
    print("-" * 70)

    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()

        # Train
        model.train()
        tr_loss = 0.0; nb = 0
        for imgs, masks in train_loader:
            imgs, masks = imgs.to(device), masks.to(device)
            with torch.amp.autocast(device_type="cuda"):
                out  = model(imgs)
                loss = criterion(out, masks)
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            tr_loss += loss.item(); nb += 1
        tr_loss /= max(nb, 1)

        # Validate
        model.eval()
        vl_loss = 0.0; nv = 0
        confusion = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs, masks = imgs.to(device), masks.to(device)
                with torch.amp.autocast(device_type="cuda"):
                    out  = model(imgs)
                    loss = criterion(out, masks)
                vl_loss += loss.item(); nv += 1
                p = out.argmax(1).cpu().numpy().ravel()
                t = masks.cpu().numpy().ravel()
                ok = (t >= 0) & (t < NUM_CLASSES) & (p >= 0) & (p < NUM_CLASSES)
                np.add.at(confusion, (t[ok], p[ok]), 1)
        vl_loss /= max(nv, 1)

        ious = []
        for c in range(NUM_CLASSES):
            tp = confusion[c, c]
            fp = confusion[:, c].sum() - tp
            fn = confusion[c, :].sum() - tp
            d  = tp + fp + fn
            ious.append(float(tp)/d if d > 0 else 0.0)
        miou      = float(np.mean(ious))
        olive_iou = ious[1]
        elapsed   = time.time() - t0
        lr_now    = scheduler.get_last_lr()[0]

        print(f"{epoch:>6} | {tr_loss:>8.4f} {vl_loss:>8.4f} | "
              f"{miou:>6.4f} {olive_iou:>6.4f} | "
              f"{lr_now:>10.2e} | {elapsed:>4.0f}s")

        history.append({"epoch": epoch, "train_loss": tr_loss, "val_loss": vl_loss,
                         "miou": miou, "olive_iou": olive_iou})

        if olive_iou > best_olive:
            best_olive = olive_iou
            best_miou  = miou
            no_improve = 0
            last_ious  = ious
            torch.save({
                "epoch":            epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "miou":             miou,
                "olive_iou":        olive_iou,
                "class_ious":       ious,
            }, os.path.join(CHECKPOINTS, ckpt_name))
            print(f"  *** NEW BEST: olive IoU={olive_iou:.4f}, mIoU={miou:.4f} -> {ckpt_name}")
        else:
            no_improve += 1
            
        # ALWAYS save the latest epoch so we never lose work if it doesn't beat the max!
        latest_ckpt = ckpt_name.replace("_best.pth", "_latest.pth")
        torch.save({
            "epoch":            epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "miou":             miou,
            "olive_iou":        olive_iou,
            "class_ious":       ious,
        }, os.path.join(CHECKPOINTS, latest_ckpt))

        if no_improve >= EARLY_STOP and not args.debug:
                print(f"\n  Early stopping ({EARLY_STOP} epochs no improvement)")
                break

    # Save history
    with open(os.path.join(LOGS, f"final_{timestamp}_history.json"), "w") as f:
        json.dump(history, f, indent=2)

    # ============================================================
    # FINAL REPORT
    # ============================================================
    print()
    print("=" * 65)
    print("FINAL RESULTS")
    print("=" * 65)
    print(f"  Best Olive IoU:  {best_olive:.4f}")
    print(f"  Best mIoU:       {best_miou:.4f}")
    print(f"  Checkpoint:      {os.path.join(CHECKPOINTS, ckpt_name)}")
    print()
    print("  Per-class IoU:")
    for c, name in enumerate(CLASS_NAMES):
        bar = "#" * int(last_ious[c] * 30)
        marker = " <-- TARGET" if c == 1 else ""
        print(f"    {name:<18} {last_ious[c]:.4f}  {bar}{marker}")
    print()
    print("  DONE! Use predict.py to run inference on any Sentinel-2 image.")
    print(f"  Model saved: checkpoints/{ckpt_name}")


if __name__ == "__main__":
    main()