"""
ZitounEye — Step 3: Create Training Dataset
=============================================
Rasterizes vector labels onto Sentinel-2 imagery grids,
cuts into 256×256 patches, computes normalization stats,
and creates train/val/test splits.

Usage:
    python scripts/03_create_dataset.py
    python scripts/03_create_dataset.py --config configs/default.yaml
"""

import os
import sys
import argparse
import warnings
import numpy as np
import rasterio
from rasterio import features as rio_features
from rasterio.transform import from_bounds
from rasterio.crs import CRS
import geopandas as gpd
from pathlib import Path
from tqdm import tqdm
from collections import Counter

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.utils import load_config, ensure_dirs, compute_channel_stats

warnings.filterwarnings("ignore")


def rasterize_labels(
    label_gpkg_path: str,
    reference_tif_path: str,
    output_path: str,
    osm_buildings_path: str = None,
    osm_roads_path: str = None,
) -> np.ndarray:
    """
    Rasterize vector labels to match a reference GeoTIFF's grid.
    
    Priority order (higher overwrites lower):
        1. Roads (class 6) — narrow, would be lost if overwritten
        2. Buildings (class 5) — small features
        3. Crop parcels (classes 1-4)
    
    Args:
        label_gpkg_path: path to crop label GeoPackage
        reference_tif_path: path to Sentinel-2 GeoTIFF (defines grid)
        output_path: where to save rasterized label GeoTIFF
        osm_buildings_path: optional path to OSM buildings GeoPackage
        osm_roads_path: optional path to OSM roads GeoPackage
    
    Returns:
        np.ndarray of shape (H, W) with class IDs
    """
    # Get reference grid info
    with rasterio.open(reference_tif_path) as src:
        transform = src.transform
        height = src.height
        width = src.width
        crs = src.crs
    
    # Initialize empty raster
    label_raster = np.zeros((height, width), dtype=np.uint8)
    
    # 1. Rasterize crop parcels
    print(f"    Rasterizing crop labels...")
    crop_gdf = gpd.read_file(label_gpkg_path)
    
    # Reproject to match imagery CRS if needed
    if crop_gdf.crs != crs:
        crop_gdf = crop_gdf.to_crs(crs)
    
    # Filter to valid geometries
    crop_gdf = crop_gdf[crop_gdf.geometry.is_valid & ~crop_gdf.geometry.is_empty]
    
    if len(crop_gdf) > 0:
        shapes = [(geom, int(class_id)) for geom, class_id 
                  in zip(crop_gdf.geometry, crop_gdf["class_id"])
                  if class_id > 0]  # Skip background
        
        if shapes:
            rio_features.rasterize(
                shapes,
                out=label_raster,
                transform=transform,
                all_touched=True,
            )
    
    # 2. Overlay buildings (class 5)
    if osm_buildings_path and os.path.exists(osm_buildings_path):
        print(f"    Rasterizing buildings...")
        bldg_gdf = gpd.read_file(osm_buildings_path)
        if bldg_gdf.crs != crs:
            bldg_gdf = bldg_gdf.to_crs(crs)
        bldg_gdf = bldg_gdf[bldg_gdf.geometry.is_valid & ~bldg_gdf.geometry.is_empty]
        
        if len(bldg_gdf) > 0:
            shapes = [(geom, 5) for geom in bldg_gdf.geometry]
            rio_features.rasterize(
                shapes,
                out=label_raster,
                transform=transform,
                all_touched=True,
            )
    
    # 3. Overlay roads (class 6) — on top of everything
    if osm_roads_path and os.path.exists(osm_roads_path):
        print(f"    Rasterizing roads...")
        road_gdf = gpd.read_file(osm_roads_path)
        if road_gdf.crs != crs:
            road_gdf = road_gdf.to_crs(crs)
        road_gdf = road_gdf[road_gdf.geometry.is_valid & ~road_gdf.geometry.is_empty]
        
        if len(road_gdf) > 0:
            shapes = [(geom, 6) for geom in road_gdf.geometry]
            rio_features.rasterize(
                shapes,
                out=label_raster,
                transform=transform,
                all_touched=True,
            )
    
    # Save as GeoTIFF
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    profile = {
        "driver": "GTiff",
        "dtype": "uint8",
        "height": height,
        "width": width,
        "count": 1,
        "crs": crs,
        "transform": transform,
        "compress": "lzw",
    }
    
    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(label_raster, 1)
    
    return label_raster


def extract_patches(
    image_path: str,
    label_path: str,
    output_dir: str,
    patch_size: int = 256,
    overlap: int = 64,
    min_valid_fraction: float = 0.5,
    max_single_class: float = 0.95,
    prefix: str = "",
) -> int:
    """
    Cut image and label rasters into training patches.
    
    Applies quality filters to skip useless patches:
    - Too much nodata (cloud gaps, edges)
    - Too homogeneous (all one class, nothing to learn from)
    
    Returns:
        Number of patches saved
    """
    # Load imagery
    with rasterio.open(image_path) as src:
        image = src.read()  # (C, H, W)
    
    # Load labels
    with rasterio.open(label_path) as src:
        label = src.read(1)  # (H, W)
    
    C, H, W = image.shape
    stride = patch_size - overlap
    
    patch_count = 0
    
    for y in range(0, H - patch_size + 1, stride):
        for x in range(0, W - patch_size + 1, stride):
            # Extract patch
            img_patch = image[:, y:y+patch_size, x:x+patch_size]
            lbl_patch = label[y:y+patch_size, x:x+patch_size]
            
            # --- Quality filters ---
            
            # Filter 1: Too much nodata (zeros across all bands)
            valid_pixels = np.any(img_patch != 0, axis=0)  # (H, W)
            valid_fraction = valid_pixels.sum() / (patch_size * patch_size)
            
            if valid_fraction < min_valid_fraction:
                continue
            
            # Filter 2: Too homogeneous
            unique, counts = np.unique(lbl_patch, return_counts=True)
            max_class_fraction = counts.max() / (patch_size * patch_size)
            
            if max_class_fraction > max_single_class:
                continue
            
            # Save patch
            patch_name = f"{prefix}_{patch_count:06d}"
            np.save(os.path.join(output_dir, f"{patch_name}_image.npy"), img_patch.astype(np.float32))
            np.save(os.path.join(output_dir, f"{patch_name}_mask.npy"), lbl_patch.astype(np.uint8))
            
            patch_count += 1
    
    return patch_count


def stack_seasonal_composites(
    imagery_dir: str,
    province: str,
    seasons: dict,
) -> dict:
    """
    Find and match seasonal composite GeoTIFFs for a province.
    
    Returns dict mapping tile_name → list of GeoTIFF paths (one per season).
    """
    tile_files = {}
    
    for season_name in seasons:
        pattern = f"{province}_{season_name}_*.tif"
        season_dir = Path(imagery_dir) / province
        
        if not season_dir.exists():
            # Try flat directory
            season_dir = Path(imagery_dir)
        
        for f in sorted(season_dir.glob(pattern)):
            # Extract tile name
            tile_name = f.stem.replace(f"{province}_{season_name}_", "")
            
            if tile_name not in tile_files:
                tile_files[tile_name] = {}
            
            tile_files[tile_name][season_name] = str(f)
    
    # Keep only tiles with all seasons
    complete_tiles = {
        k: v for k, v in tile_files.items()
        if len(v) == len(seasons)
    }
    
    return complete_tiles


def create_stacked_image(season_paths: dict, output_path: str) -> str:
    """
    Stack seasonal composites into a single multi-band GeoTIFF.
    
    4 seasons × 14 bands = 56 total bands.
    This captures the full phenological signal needed to distinguish
    evergreen olives from deciduous orchards.
    """
    season_order = ["spring", "summer", "fall", "winter"]
    
    all_bands = []
    reference_profile = None
    
    for season in season_order:
        if season not in season_paths:
            continue
        
        with rasterio.open(season_paths[season]) as src:
            data = src.read()  # (C, H, W)
            all_bands.append(data)
            
            if reference_profile is None:
                reference_profile = src.profile.copy()
    
    if not all_bands:
        return None
    
    stacked = np.concatenate(all_bands, axis=0)  # (C_total, H, W)
    
    # Replace NaN/inf with 0
    stacked = np.nan_to_num(stacked, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Save stacked image
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    reference_profile.update({
        "count": stacked.shape[0],
        "dtype": "float32",
        "compress": "lzw",
    })
    
    with rasterio.open(output_path, "w", **reference_profile) as dst:
        dst.write(stacked.astype(np.float32))
    
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Create training dataset from Sentinel-2 imagery and labels"
    )
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    
    config = load_config(args.config)
    ensure_dirs(config)
    
    print("=" * 60)
    print("ZitounEye — Dataset Creation")
    print("=" * 60)
    
    patch_size = config["dataset"]["patch_size"]
    overlap = config["dataset"]["patch_overlap"]
    imagery_dir = config["paths"]["imagery_dir"]
    labels_dir = config["paths"]["labels_dir"]
    patches_dir = config["paths"]["patches_dir"]
    seasons = config["dataset"]["seasons"]
    
    # Process each split
    splits = {
        "train": config["provinces"]["train"],
        "val": config["provinces"]["val"],
        "test": config["provinces"]["test"],
    }
    
    total_patches = {"train": 0, "val": 0, "test": 0}
    class_distribution = {"train": Counter(), "val": Counter(), "test": Counter()}
    
    for split_name, provinces in splits.items():
        split_dir = os.path.join(patches_dir, split_name)
        os.makedirs(split_dir, exist_ok=True)
        
        print(f"\n{'='*40}")
        print(f"Split: {split_name} ({len(provinces)} provinces)")
        print(f"{'='*40}")
        
        for prov in provinces:
            print(f"\n  Province: {prov}")
            
            # Find label file
            label_path = os.path.join(labels_dir, f"{prov}_labels.gpkg")
            if not os.path.exists(label_path):
                print(f"    ⚠️  Labels not found: {label_path}")
                continue
            
            # Find and stack seasonal imagery
            tile_dict = stack_seasonal_composites(imagery_dir, prov, seasons)
            
            if not tile_dict:
                print(f"    ⚠️  No complete seasonal imagery found for {prov}")
                print(f"    Looking in: {imagery_dir}/{prov}/")
                
                # Fallback: try individual GeoTIFFs
                prov_dir = Path(imagery_dir) / prov
                if prov_dir.exists():
                    tifs = list(prov_dir.glob("*.tif"))
                    print(f"    Found {len(tifs)} individual GeoTIFFs")
                    
                    for tif in tifs:
                        tile_name = tif.stem
                        
                        # Rasterize labels for this tile
                        label_raster_path = os.path.join(
                            labels_dir, "rasterized", f"{tile_name}_labels.tif"
                        )
                        
                        osm_bldg = os.path.join(labels_dir, f"{prov}_buildings.gpkg")
                        osm_road = os.path.join(labels_dir, f"{prov}_roads.gpkg")
                        
                        print(f"\n    Processing tile: {tile_name}")
                        
                        rasterize_labels(
                            label_gpkg_path=label_path,
                            reference_tif_path=str(tif),
                            output_path=label_raster_path,
                            osm_buildings_path=osm_bldg if os.path.exists(osm_bldg) else None,
                            osm_roads_path=osm_road if os.path.exists(osm_road) else None,
                        )
                        
                        # Extract patches
                        n_patches = extract_patches(
                            image_path=str(tif),
                            label_path=label_raster_path,
                            output_dir=split_dir,
                            patch_size=patch_size,
                            overlap=overlap,
                            min_valid_fraction=config["dataset"]["min_valid_fraction"],
                            max_single_class=config["dataset"]["max_single_class"],
                            prefix=f"{prov}_{tile_name}",
                        )
                        
                        total_patches[split_name] += n_patches
                        print(f"    → {n_patches} patches extracted")
                
                continue
            
            print(f"    Found {len(tile_dict)} complete tiles with all seasons")
            
            for tile_name, season_paths in tqdm(tile_dict.items(), desc=f"    {prov}"):
                # Stack seasons into one multi-band image
                stacked_path = os.path.join(
                    imagery_dir, "stacked", f"{prov}_{tile_name}_stacked.tif"
                )
                
                result = create_stacked_image(season_paths, stacked_path)
                if result is None:
                    continue
                
                # Rasterize labels for this tile
                label_raster_path = os.path.join(
                    labels_dir, "rasterized", f"{prov}_{tile_name}_labels.tif"
                )
                
                osm_bldg = os.path.join(labels_dir, f"{prov}_buildings.gpkg")
                osm_road = os.path.join(labels_dir, f"{prov}_roads.gpkg")
                
                rasterize_labels(
                    label_gpkg_path=label_path,
                    reference_tif_path=stacked_path,
                    output_path=label_raster_path,
                    osm_buildings_path=osm_bldg if os.path.exists(osm_bldg) else None,
                    osm_roads_path=osm_road if os.path.exists(osm_road) else None,
                )
                
                # Extract patches
                n_patches = extract_patches(
                    image_path=stacked_path,
                    label_path=label_raster_path,
                    output_dir=split_dir,
                    patch_size=patch_size,
                    overlap=overlap,
                    min_valid_fraction=config["dataset"]["min_valid_fraction"],
                    max_single_class=config["dataset"]["max_single_class"],
                    prefix=f"{prov}_{tile_name}",
                )
                
                total_patches[split_name] += n_patches
    
    # Compute normalization stats from training set
    print(f"\n{'='*60}")
    print("Computing channel normalization statistics...")
    print(f"{'='*60}")
    
    train_dir = os.path.join(patches_dir, "train")
    train_files = list(Path(train_dir).glob("*_image.npy"))
    
    if len(train_files) > 0:
        mean, std = compute_channel_stats(train_dir)
        
        stats_path = os.path.join(patches_dir, "normalize_stats.npz")
        np.savez(stats_path, mean=mean, std=std)
        print(f"  Saved to: {stats_path}")
        print(f"  Channels: {len(mean)}")
        print(f"  Mean range: [{mean.min():.4f}, {mean.max():.4f}]")
        print(f"  Std range:  [{std.min():.4f}, {std.max():.4f}]")
    
    # Final summary
    print(f"\n{'='*60}")
    print("DATASET SUMMARY")
    print(f"{'='*60}")
    
    for split_name, count in total_patches.items():
        print(f"  {split_name:<10}: {count:>8,} patches")
    
    print(f"\n  Patch size: {patch_size}×{patch_size} pixels ({patch_size * 10}m × {patch_size * 10}m)")
    print(f"  Overlap:    {overlap} pixels")
    print(f"  Output dir: {patches_dir}")
    
    print("\nDone! ✅")
    print(f"\nNext step: python scripts/04_train_model.py")


if __name__ == "__main__":
    main()
