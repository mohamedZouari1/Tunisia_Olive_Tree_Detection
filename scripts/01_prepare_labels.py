"""
ZitounEye — Step 1: Label Preparation
=======================================
Reads Spain EuroCrop 2023 shapefiles, reclassifies ~500 crop codes
into 7 training classes, and optionally fetches building/road
geometries from OpenStreetMap.

Usage:
    python scripts/01_prepare_labels.py
    python scripts/01_prepare_labels.py --provinces es_j_2023 es_co_2023
    python scripts/01_prepare_labels.py --provinces es_j_2023 --dry-run
    python scripts/01_prepare_labels.py --skip-osm   # skip building/road download
"""

import os
import sys
import argparse
import warnings
import numpy as np
import pandas as pd
import geopandas as gpd
from pathlib import Path
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.utils import load_config, ensure_dirs

warnings.filterwarnings("ignore", category=FutureWarning)


# ─── Spanish crop code → class ID mapping ────────────────────────────
# This is the critical mapping that determines label quality.
# Based on es_2023.csv and HCAT3.csv analysis.

def build_full_crop_mapping(config: dict, es_csv_path: str) -> dict:
    """
    Build comprehensive mapping from every Spanish crop code to our 7-class schema.
    
    Strategy:
        1. Explicit olive/vineyard/orchard codes from config
        2. HCAT3 prefix rules for remaining codes
        3. Everything unmapped → class 0 (background) or class 4 (arable)
    
    Returns:
        dict mapping original_code (int) → class_id (int)
    """
    # Load the Spanish crop code mapping CSV
    df = pd.read_csv(es_csv_path)
    
    mapping = {}
    
    # --- Priority 1: Explicit codes from config ---
    olive_codes = set(config["classes"]["olive_codes"])
    vineyard_codes = set(config["classes"]["vineyard_codes"])
    orchard_codes = set(config["classes"]["other_orchard_codes"])
    
    for _, row in df.iterrows():
        code = int(row["original_code"])
        hcat3_code = str(row["HCAT3_code"])
        hcat3_name = str(row["HCAT3_name"]) if pd.notna(row["HCAT3_name"]) else ""
        
        if code in olive_codes:
            mapping[code] = 1  # Olive
        elif code in vineyard_codes:
            mapping[code] = 2  # Vineyard
        elif code in orchard_codes:
            mapping[code] = 3  # Other orchard
        else:
            # --- Priority 2: HCAT3 taxonomy rules ---
            if hcat3_code.startswith("3303"):
                # Permanent crops — could be orchard/fruit/berries
                if "olive" in hcat3_name.lower():
                    mapping[code] = 1
                elif "vine" in hcat3_name.lower() or "grape" in hcat3_name.lower():
                    mapping[code] = 2
                else:
                    mapping[code] = 3  # Other permanent crop / orchard
            elif hcat3_code.startswith("3301"):
                mapping[code] = 4  # Arable / annual crop
            elif hcat3_code.startswith("3302"):
                mapping[code] = 4  # Pasture → treat as arable
            elif hcat3_code.startswith("3306"):
                mapping[code] = 0  # Forest → background
            elif hcat3_code.startswith("3308"):
                mapping[code] = 0  # Unmaintained → background
            elif hcat3_code.startswith("3399"):
                mapping[code] = 0  # Unknown → background
            elif hcat3_code.startswith("3304"):
                mapping[code] = 4  # Mushrooms/energy crops → arable
            elif hcat3_code.startswith("3305"):
                mapping[code] = 4  # Greenhouse → arable
            elif hcat3_code.startswith("3307"):
                mapping[code] = 0  # Peat → background
            else:
                mapping[code] = 0  # Anything else → background
    
    return mapping


def process_province(
    shapefile_path: str,
    crop_mapping: dict,
    output_path: str,
    target_crs: str = "EPSG:32630",
    dry_run: bool = False,
) -> dict:
    """
    Process one province shapefile: reclassify crops, reproject, save.
    
    Args:
        shapefile_path: path to .shp file
        crop_mapping: dict mapping original_code → class_id
        output_path: where to save the GeoPackage
        target_crs: target coordinate reference system
        dry_run: if True, just report stats without saving
    
    Returns:
        dict with statistics
    """
    province_name = Path(shapefile_path).stem
    print(f"\n{'='*60}")
    print(f"Processing: {province_name}")
    print(f"{'='*60}")
    
    # Read shapefile
    print(f"  Reading {shapefile_path}...")
    gdf = gpd.read_file(shapefile_path)
    total_parcels = len(gdf)
    print(f"  Total parcels: {total_parcels:,}")
    
    # Find the crop code column
    # EuroCrop shapefiles use various column names depending on version
    code_col = None
    possible_cols = ["EC_OCODE", "EC_ocode", "ec_ocode", "original_code", "OCODE", "EC_hcat_c"]
    
    for col in possible_cols:
        if col in gdf.columns:
            code_col = col
            break
    
    if code_col is None:
        # Try to find any column with numeric crop codes
        print(f"  ⚠️  Available columns: {list(gdf.columns)}")
        # Try the first column that looks like codes
        for col in gdf.columns:
            if col.lower() not in ("geometry", "shape_leng", "shape_area"):
                try:
                    sample = gdf[col].dropna().iloc[:10]
                    if all(isinstance(v, (int, float, np.integer, np.floating)) for v in sample):
                        code_col = col
                        print(f"  Using column '{col}' as crop code")
                        break
                except Exception:
                    continue
    
    if code_col is None:
        print(f"  ❌ Could not find crop code column! Skipping.")
        return {"province": province_name, "status": "failed", "reason": "no_code_column"}
    
    # Map crop codes to class IDs
    gdf["class_id"] = gdf[code_col].apply(
        lambda x: crop_mapping.get(int(x), 0) if pd.notna(x) else 0
    )
    
    # Statistics
    class_names = ["background", "olive", "vineyard", "other_orchard", "arable", "building", "road"]
    stats = {}
    print(f"\n  Class distribution:")
    for cid in range(7):
        count = (gdf["class_id"] == cid).sum()
        pct = count / total_parcels * 100
        stats[class_names[cid]] = count
        marker = " ★" if cid == 1 else ""
        print(f"    {class_names[cid]:<20}: {count:>8,} parcels ({pct:>5.1f}%){marker}")
    
    # Reproject to UTM
    if gdf.crs is not None and str(gdf.crs) != target_crs:
        print(f"\n  Reprojecting {gdf.crs} → {target_crs}...")
        gdf = gdf.to_crs(target_crs)
    
    # Keep only needed columns for efficiency
    keep_cols = ["geometry", "class_id"]
    if code_col in gdf.columns:
        keep_cols.append(code_col)
    # Also keep any name column for debugging
    for nc in ["EC_trans_n", "EC_TRANS_N", "ec_trans_n", "translated_name"]:
        if nc in gdf.columns:
            keep_cols.append(nc)
            break
    
    gdf = gdf[keep_cols]
    
    if dry_run:
        print(f"\n  [DRY RUN] Would save to {output_path}")
    else:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        gdf.to_file(output_path, driver="GPKG")
        print(f"\n  ✅ Saved: {output_path}")
        print(f"     Size: {os.path.getsize(output_path) / 1024 / 1024:.1f} MB")
    
    stats["province"] = province_name
    stats["total_parcels"] = total_parcels
    stats["status"] = "success"
    
    return stats


def fetch_osm_data(bounds: tuple, output_dir: str, province_name: str) -> None:
    """
    Download building footprints and road networks from OpenStreetMap
    for the given bounding box.
    
    Args:
        bounds: (minx, miny, maxx, maxy) in WGS84
        output_dir: directory to save GeoPackage files
        province_name: for naming output files
    """
    try:
        import osmnx as ox
    except ImportError:
        print("  ⚠️  osmnx not installed. Skipping OSM download.")
        print("     Install with: pip install osmnx")
        return
    
    minx, miny, maxx, maxy = bounds
    
    # Limit area to avoid huge downloads — split into sub-tiles if needed
    # For now, sample a representative area (center 20km × 20km)
    cx, cy = (minx + maxx) / 2, (miny + maxy) / 2
    half_deg = 0.1  # ~11km at Spanish latitudes
    
    bbox = (cy - half_deg, cy + half_deg, cx - half_deg, cx + half_deg)  # N, S, E, W
    
    print(f"\n  Fetching OSM buildings for {province_name}...")
    try:
        buildings = ox.features_from_bbox(
            bbox=bbox,
            tags={"building": True}
        )
        
        if len(buildings) > 0:
            # Keep only polygons (not points or lines)
            buildings = buildings[buildings.geometry.type.isin(["Polygon", "MultiPolygon"])]
            buildings["class_id"] = 5  # Building class
            
            bldg_path = os.path.join(output_dir, f"{province_name}_buildings.gpkg")
            buildings[["geometry", "class_id"]].to_file(bldg_path, driver="GPKG")
            print(f"    ✅ Buildings: {len(buildings):,} features → {bldg_path}")
        else:
            print(f"    ⚠️  No buildings found in sample area")
    except Exception as e:
        print(f"    ❌ Building download failed: {e}")
    
    print(f"  Fetching OSM roads for {province_name}...")
    try:
        G = ox.graph_from_bbox(bbox=bbox, network_type="drive")
        nodes, edges = ox.graph_to_gdfs(G)
        
        if len(edges) > 0:
            # Buffer road centerlines to create road polygons
            # Use 5m buffer (~1 pixel at 10m resolution) for small roads
            # and 10m for larger roads
            edges = edges.to_crs("EPSG:32630")
            edges["geometry"] = edges.geometry.buffer(7.5)  # 15m total width
            edges["class_id"] = 6  # Road class
            
            road_path = os.path.join(output_dir, f"{province_name}_roads.gpkg")
            edges[["geometry", "class_id"]].to_file(road_path, driver="GPKG")
            print(f"    ✅ Roads: {len(edges):,} segments → {road_path}")
        else:
            print(f"    ⚠️  No roads found in sample area")
    except Exception as e:
        print(f"    ❌ Road download failed: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Prepare training labels from Spain EuroCrop 2023 shapefiles"
    )
    parser.add_argument(
        "--config", default="configs/default.yaml",
        help="Path to config file"
    )
    parser.add_argument(
        "--provinces", nargs="+", default=None,
        help="Specific province folders to process (e.g., es_j_2023 es_co_2023). "
             "If not specified, uses provinces from config."
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Report statistics without saving files"
    )
    parser.add_argument(
        "--skip-osm", action="store_true",
        help="Skip downloading OSM buildings/roads"
    )
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    ensure_dirs(config)
    
    print("=" * 60)
    print("ZitounEye — Label Preparation")
    print("=" * 60)
    
    # Build crop code mapping
    es_csv = os.path.join(config["paths"]["project_root"], "es_2023.csv")
    crop_mapping = build_full_crop_mapping(config, es_csv)
    print(f"\nCrop code mapping: {len(crop_mapping)} codes → 7 classes")
    
    # Determine provinces to process
    shapefile_root = config["paths"]["raw_shapefiles"]
    
    if args.provinces:
        province_dirs = args.provinces
    else:
        # Use all provinces from config (train + val + test)
        province_dirs = (
            config["provinces"]["train"] +
            config["provinces"]["val"] +
            config["provinces"]["test"]
        )
    
    print(f"Provinces to process: {province_dirs}")
    
    all_stats = []
    
    for prov in province_dirs:
        shp_path = os.path.join(shapefile_root, prov, f"{prov}.shp")
        
        if not os.path.exists(shp_path):
            print(f"\n⚠️  Shapefile not found: {shp_path}")
            continue
        
        output_path = os.path.join(
            config["paths"]["labels_dir"],
            f"{prov}_labels.gpkg"
        )
        
        stats = process_province(
            shapefile_path=shp_path,
            crop_mapping=crop_mapping,
            output_path=output_path,
            dry_run=args.dry_run,
        )
        all_stats.append(stats)
        
        # Fetch OSM data for this province's extent
        if not args.skip_osm and not args.dry_run:
            # Get bounds in WGS84
            gdf_temp = gpd.read_file(shp_path)
            if gdf_temp.crs and gdf_temp.crs != "EPSG:4326":
                bounds_gdf = gdf_temp.to_crs("EPSG:4326")
            else:
                bounds_gdf = gdf_temp
            
            bounds = bounds_gdf.total_bounds  # (minx, miny, maxx, maxy)
            
            fetch_osm_data(
                bounds=tuple(bounds),
                output_dir=config["paths"]["labels_dir"],
                province_name=prov,
            )
    
    # Summary
    print("\n\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    total_olive = 0
    total_parcels = 0
    
    for s in all_stats:
        if s.get("status") == "success":
            olive = s.get("olive", 0)
            total = s.get("total_parcels", 0)
            total_olive += olive
            total_parcels += total
            print(f"  {s['province']:<20}: {olive:>8,} olive / {total:>10,} total")
    
    if total_parcels > 0:
        print(f"\n  TOTAL: {total_olive:,} olive parcels / {total_parcels:,} total "
              f"({total_olive / total_parcels * 100:.1f}%)")
    
    print("\nDone! ✅")


if __name__ == "__main__":
    main()
