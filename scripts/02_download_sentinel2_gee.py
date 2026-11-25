"""
ZitounEye — Step 2: Download Sentinel-2 from Google Earth Engine
=================================================================
Generates cloud-free monthly composites from Sentinel-2 L2A (SR Harmonized)
and exports multi-band GeoTIFFs to Google Drive.

Prerequisites:
    1. pip install earthengine-api geemap
    2. Run `earthengine authenticate` in terminal (one-time)
    3. Have a Google Cloud project with Earth Engine enabled

Usage:
    python scripts/02_download_sentinel2_gee.py
    python scripts/02_download_sentinel2_gee.py --province es_j_2023 --year 2023
    python scripts/02_download_sentinel2_gee.py --province es_j_2023 --months 6 7 8
    python scripts/02_download_sentinel2_gee.py --aoi-bbox -4.5 37.0 -2.5 38.5
"""

import os
import sys
import argparse
import json
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def initialize_ee():
    """Initialize Google Earth Engine with authentication."""
    import ee
    
    try:
        ee.Initialize(opt_url='https://earthengine-highvolume.googleapis.com')
        print("✅ Earth Engine initialized (high-volume endpoint)")
    except Exception:
        try:
            ee.Initialize()
            print("✅ Earth Engine initialized")
        except Exception:
            print("⚠️  Earth Engine not authenticated. Running authentication...")
            ee.Authenticate()
            ee.Initialize()
            print("✅ Earth Engine initialized after authentication")
    
    return ee


def cloud_mask_s2(image):
    """
    Mask clouds and cloud shadows using the Scene Classification Layer (SCL).
    
    SCL values:
        1 = Saturated/defective     → mask
        2 = Dark area pixel         → keep
        3 = Cloud shadow            → mask
        4 = Vegetation              → keep
        5 = Bare soil               → keep
        6 = Water                   → keep  
        7 = Unclassified            → keep (often olive groves!)
        8 = Cloud (medium prob)     → mask
        9 = Cloud (high prob)       → mask
        10 = Thin cirrus            → mask
        11 = Snow/ice               → mask
    """
    import ee
    
    scl = image.select("SCL")
    
    # Keep only clear pixels
    clear = (
        scl.eq(2)   # Dark area
        .Or(scl.eq(4))  # Vegetation
        .Or(scl.eq(5))  # Bare soil
        .Or(scl.eq(6))  # Water
        .Or(scl.eq(7))  # Unclassified
    )
    
    return image.updateMask(clear)


def add_indices(image):
    """
    Add spectral indices critical for olive grove detection.
    
    - NDVI: Normalized Difference Vegetation Index
      Olives maintain NDVI ~0.3-0.5 year-round (evergreen)
      Deciduous orchards drop to <0.2 in winter
    
    - EVI: Enhanced Vegetation Index  
      Better than NDVI for dense canopy areas, reduces atmosphere effects
    
    - NDWI: Normalized Difference Water Index
      Helps distinguish irrigated vs rainfed olives
    
    - SAVI: Soil-Adjusted Vegetation Index
      Critical for sparse traditional olive groves where soil is visible
    """
    import ee
    
    ndvi = image.normalizedDifference(["B8", "B4"]).rename("NDVI")
    
    evi = image.expression(
        "2.5 * ((NIR - RED) / (NIR + 6 * RED - 7.5 * BLUE + 1))",
        {
            "NIR": image.select("B8"),
            "RED": image.select("B4"),
            "BLUE": image.select("B2"),
        }
    ).rename("EVI")
    
    ndwi = image.normalizedDifference(["B3", "B8"]).rename("NDWI")
    
    savi = image.expression(
        "1.5 * ((NIR - RED) / (NIR + RED + 0.5))",
        {
            "NIR": image.select("B8"),
            "RED": image.select("B4"),
        }
    ).rename("SAVI")
    
    return image.addBands([ndvi, evi, ndwi, savi])


def get_monthly_composite(ee, year, month, aoi, max_cloud=30):
    """
    Create a cloud-free median composite for one month.
    
    Uses median compositing which:
    1. Effectively removes remaining cloud pixels (bright outliers)
    2. Reduces noise from any single acquisition
    3. Produces a representative "typical" view for that month
    """
    start_date = f"{year}-{month:02d}-01"
    
    if month == 12:
        end_date = f"{year + 1}-01-01"
    else:
        end_date = f"{year}-{month + 1:02d}-01"
    
    collection = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterDate(start_date, end_date)
        .filterBounds(aoi)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", max_cloud))
        .map(cloud_mask_s2)
        .map(add_indices)
    )
    
    # Select bands we need
    bands = [
        "B2", "B3", "B4",      # Blue, Green, Red (10m)
        "B5", "B6", "B7",      # Red Edge 1/2/3 (20m → resampled to 10m)
        "B8",                    # NIR (10m)
        "B8A",                   # Narrow NIR (20m → resampled to 10m)
        "B11", "B12",           # SWIR 1/2 (20m → resampled to 10m)
        "NDVI", "EVI",          # Computed indices
        "NDWI", "SAVI",         # Additional indices
    ]
    
    composite = collection.select(bands).median()
    
    # Scale reflectance from 0-10000 to 0-1 for spectral bands
    # Indices are already in -1 to 1 range
    spectral_bands = ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"]
    index_bands = ["NDVI", "EVI", "NDWI", "SAVI"]
    
    scaled_spectral = composite.select(spectral_bands).divide(10000)
    indices = composite.select(index_bands)
    
    return scaled_spectral.addBands(indices).set("month", month)


def get_seasonal_composite(ee, year, months, aoi, max_cloud=30):
    """
    Create a seasonal composite by merging multiple months.
    More robust than single-month composites for cloudy seasons.
    """
    start_month = min(months)
    end_month = max(months)
    
    if start_month > end_month:
        # Winter wraps around year boundary
        start_date = f"{year}-{start_month:02d}-01"
        end_date = f"{year + 1}-{end_month + 1:02d}-01"
    else:
        start_date = f"{year}-{start_month:02d}-01"
        if end_month == 12:
            end_date = f"{year + 1}-01-01"
        else:
            end_date = f"{year}-{end_month + 1:02d}-01"
    
    collection = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterDate(start_date, end_date)
        .filterBounds(aoi)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", max_cloud))
        .map(cloud_mask_s2)
        .map(add_indices)
    )
    
    bands = [
        "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12",
        "NDVI", "EVI", "NDWI", "SAVI",
    ]
    
    composite = collection.select(bands).median()
    
    spectral_bands = ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"]
    index_bands = ["NDVI", "EVI", "NDWI", "SAVI"]
    
    scaled_spectral = composite.select(spectral_bands).divide(10000)
    indices = composite.select(index_bands)
    
    return scaled_spectral.addBands(indices)


def get_province_bounds(province_name: str, shapefile_root: str):
    """
    Get the bounding box of a province from its shapefile.
    Returns ee.Geometry.Rectangle and bounds tuple.
    """
    import geopandas as gpd
    import ee
    
    shp_path = os.path.join(shapefile_root, province_name, f"{province_name}.shp")
    
    if not os.path.exists(shp_path):
        raise FileNotFoundError(f"Shapefile not found: {shp_path}")
    
    # Read just enough to get bounds (much faster than loading all geometries)
    gdf = gpd.read_file(shp_path, rows=0)
    
    # Get bounds from the file metadata
    import fiona
    with fiona.open(shp_path) as src:
        bounds = src.bounds  # (minx, miny, maxx, maxy)
    
    # Ensure WGS84
    if gdf.crs and str(gdf.crs) != "EPSG:4326":
        from pyproj import Transformer
        transformer = Transformer.from_crs(str(gdf.crs), "EPSG:4326", always_xy=True)
        minx, miny = transformer.transform(bounds[0], bounds[1])
        maxx, maxy = transformer.transform(bounds[2], bounds[3])
        bounds = (minx, miny, maxx, maxy)
    
    aoi = ee.Geometry.Rectangle([bounds[0], bounds[1], bounds[2], bounds[3]])
    
    return aoi, bounds


def create_tile_grid(bounds, tile_size_deg=0.1):
    """
    Create a grid of tiles covering the bounding box.
    
    tile_size_deg ≈ 0.1° ≈ 10km at Spanish latitudes.
    Each tile becomes one export task and one GeoTIFF.
    """
    import ee
    
    minx, miny, maxx, maxy = bounds
    
    tiles = []
    x = minx
    while x < maxx:
        y = miny
        while y < maxy:
            tile_bounds = [x, y, min(x + tile_size_deg, maxx), min(y + tile_size_deg, maxy)]
            tile_geom = ee.Geometry.Rectangle(tile_bounds)
            tile_name = f"tile_{x:.2f}_{y:.2f}".replace(".", "p").replace("-", "m")
            tiles.append((tile_name, tile_geom, tile_bounds))
            y += tile_size_deg
        x += tile_size_deg
    
    return tiles


def export_composite(
    ee,
    image,
    tile_name: str,
    tile_geom,
    province: str,
    season_name: str,
    drive_folder: str,
    scale: int = 10,
):
    """
    Export a single composite tile to Google Drive.
    
    Returns the export task object.
    """
    description = f"{province}_{season_name}_{tile_name}"
    
    task = ee.batch.Export.image.toDrive(
        image=image.toFloat(),
        description=description[:100],  # GEE description limit
        folder=drive_folder,
        fileNamePrefix=description,
        region=tile_geom,
        scale=scale,
        crs="EPSG:32630",  # UTM zone 30N for Spain
        maxPixels=1e13,
        fileFormat="GeoTIFF",
    )
    
    return task


def main():
    parser = argparse.ArgumentParser(
        description="Download Sentinel-2 composites from Google Earth Engine"
    )
    parser.add_argument(
        "--config", default="configs/default.yaml",
        help="Path to config file"
    )
    parser.add_argument(
        "--province", default=None,
        help="Single province to process (e.g., es_j_2023)"
    )
    parser.add_argument(
        "--year", type=int, default=2023,
        help="Year for composites"
    )
    parser.add_argument(
        "--months", nargs="+", type=int, default=None,
        help="Specific months to export (e.g., 6 7 8 for summer)"
    )
    parser.add_argument(
        "--mode", choices=["seasonal", "monthly", "single"], default="seasonal",
        help="Temporal compositing mode"
    )
    parser.add_argument(
        "--aoi-bbox", nargs=4, type=float, default=None,
        metavar=("WEST", "SOUTH", "EAST", "NORTH"),
        help="Custom AOI bounding box in WGS84 (lon_min lat_min lon_max lat_max)"
    )
    parser.add_argument(
        "--drive-folder", default="ZitounEye_Sentinel2",
        help="Google Drive folder for exports"
    )
    parser.add_argument(
        "--tile-size", type=float, default=0.1,
        help="Tile size in degrees (~10km at Spanish latitudes)"
    )
    parser.add_argument(
        "--scale", type=int, default=10,
        help="Output resolution in meters"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print export plan without starting tasks"
    )
    args = parser.parse_args()
    
    # Load config
    sys.path.insert(0, ".")
    from src.utils import load_config
    config = load_config(args.config)
    
    print("=" * 60)
    print("ZitounEye — Sentinel-2 Download from GEE")
    print("=" * 60)
    
    # Initialize Earth Engine
    ee = initialize_ee()
    
    # Determine AOI
    if args.aoi_bbox:
        west, south, east, north = args.aoi_bbox
        aoi = ee.Geometry.Rectangle([west, south, east, north])
        bounds = (west, south, east, north)
        province_name = "custom_aoi"
        print(f"\nAOI: Custom bbox [{west}, {south}, {east}, {north}]")
    elif args.province:
        aoi, bounds = get_province_bounds(
            args.province, config["paths"]["raw_shapefiles"]
        )
        province_name = args.province
        print(f"\nAOI: Province {args.province}")
        print(f"  Bounds: {bounds}")
    else:
        # Default: first training province (Jaén — olive capital of the world)
        default_prov = config["provinces"]["train"][0]
        aoi, bounds = get_province_bounds(
            default_prov, config["paths"]["raw_shapefiles"]
        )
        province_name = default_prov
        print(f"\nAOI: Province {default_prov} (default)")
        print(f"  Bounds: {bounds}")
    
    year = args.year
    
    # Create tile grid
    tiles = create_tile_grid(bounds, tile_size_deg=args.tile_size)
    print(f"\nTile grid: {len(tiles)} tiles (each ~{args.tile_size}° ≈ {args.tile_size * 111:.0f}km)")
    
    # Determine temporal composites
    tasks_to_start = []
    
    if args.mode == "seasonal":
        seasons = config["dataset"]["seasons"]
        print(f"\nMode: Seasonal composites")
        
        for season_name, months in seasons.items():
            # Handle winter wrapping
            if season_name == "winter":
                # Winter of 2023 = Dec 2022 + Jan-Feb 2023
                # For simplicity, use Dec 2022–Feb 2023
                actual_year = year - 1 if 12 in months else year
            else:
                actual_year = year
            
            print(f"  {season_name}: months {months}")
            composite = get_seasonal_composite(
                ee, year, months, aoi, 
                max_cloud=config["sentinel2"]["cloud_cover_max"]
            )
            
            for tile_name, tile_geom, tile_bounds in tiles:
                task = export_composite(
                    ee, composite, tile_name, tile_geom,
                    province_name, season_name, args.drive_folder, args.scale
                )
                tasks_to_start.append((f"{season_name}_{tile_name}", task))
    
    elif args.mode == "monthly":
        months = args.months or list(range(1, 13))
        print(f"\nMode: Monthly composites for months {months}")
        
        for month in months:
            composite = get_monthly_composite(
                ee, year, month, aoi,
                max_cloud=config["sentinel2"]["cloud_cover_max"]
            )
            month_name = f"m{month:02d}"
            
            for tile_name, tile_geom, tile_bounds in tiles:
                task = export_composite(
                    ee, composite, tile_name, tile_geom,
                    province_name, month_name, args.drive_folder, args.scale
                )
                tasks_to_start.append((f"{month_name}_{tile_name}", task))
    
    elif args.mode == "single":
        months = args.months or [7]  # Default: July (peak growing season)
        print(f"\nMode: Single composite from months {months}")
        
        composite = get_seasonal_composite(
            ee, year, months, aoi,
            max_cloud=config["sentinel2"]["cloud_cover_max"]
        )
        
        for tile_name, tile_geom, tile_bounds in tiles:
            task = export_composite(
                ee, composite, tile_name, tile_geom,
                province_name, "single", args.drive_folder, args.scale
            )
            tasks_to_start.append((f"single_{tile_name}", task))
    
    # Summary
    print(f"\n{'='*60}")
    print(f"EXPORT PLAN")
    print(f"{'='*60}")
    print(f"  Province:     {province_name}")
    print(f"  Year:         {year}")
    print(f"  Mode:         {args.mode}")
    print(f"  Resolution:   {args.scale}m")
    print(f"  Tiles:        {len(tiles)}")
    print(f"  Total tasks:  {len(tasks_to_start)}")
    print(f"  Drive folder: {args.drive_folder}")
    print(f"  Bands per composite: 14 (10 spectral + NDVI + EVI + NDWI + SAVI)")
    
    if args.mode == "seasonal":
        total_bands = 14 * 4
    elif args.mode == "monthly":
        total_bands = 14 * len(months)
    else:
        total_bands = 14
    print(f"  Total band stack: {total_bands} (to be stacked in step 03)")
    
    if args.dry_run:
        print(f"\n[DRY RUN] Would start {len(tasks_to_start)} export tasks.")
        print("Remove --dry-run to actually start exports.")
        return
    
    # Start all tasks
    print(f"\nStarting {len(tasks_to_start)} export tasks...")
    
    started = 0
    failed = 0
    
    for name, task in tasks_to_start:
        try:
            task.start()
            started += 1
            if started % 10 == 0:
                print(f"  Started {started}/{len(tasks_to_start)}...")
        except Exception as e:
            print(f"  ❌ Failed to start {name}: {e}")
            failed += 1
    
    print(f"\n{'='*60}")
    print(f"✅ Started {started} export tasks ({failed} failed)")
    print(f"{'='*60}")
    print(f"\nExports are running on Google's servers.")
    print(f"Check progress at: https://code.earthengine.google.com/tasks")
    print(f"Files will appear in Google Drive folder: '{args.drive_folder}'")
    print(f"\nExpected completion: 2-12 hours depending on area size.")
    print(f"\nOnce complete, download the GeoTIFFs from Drive to:")
    print(f"  {config['paths']['imagery_dir']}/{province_name}/")
    print(f"\nThen run: python scripts/03_create_dataset.py")


if __name__ == "__main__":
    main()
