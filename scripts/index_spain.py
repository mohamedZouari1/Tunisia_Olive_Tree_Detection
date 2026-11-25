import os
import glob
import rasterio
from pyproj import Transformer

def get_wgs84_bounds(filepath):
    # Using pyproj instead of rasterio.warp because of PROJ_DATA mismatch
    with rasterio.open(filepath) as src:
        bounds = src.bounds
        crs_str = src.crs.to_string()
        
    transformer = Transformer.from_crs(crs_str, 'EPSG:4326', always_xy=True)
    # bounds are (left, bottom, right, top)
    minx, miny = transformer.transform(bounds.left, bounds.bottom)
    maxx, maxy = transformer.transform(bounds.right, bounds.top)
    
    return (minx, miny, maxx, maxy), crs_str

def index_spain(folder):
    # Group by the grid id, e.g. es_j_2023_fall_2023-0000000000-0000000000-003.tif
    # The grid ID is everything after the season
    files = glob.glob(os.path.join(folder, '*.tif'))
    groups = {}
    for f in files:
        base = os.path.basename(f)
        parts = base.split('_')
        # parts: ['es', 'j', '2023', 'fall', '2023-0000...003.tif']
        season = parts[3]
        grid_id = "_".join(parts[4:]).replace('.tif', '')
        if grid_id not in groups:
            groups[grid_id] = {}
        groups[grid_id][season] = f
    
    collections = []
    for grid_id, season_files in groups.items():
        if len(season_files) == 4:
            # Check bounds on one
            wgs84_bounds, crs = get_wgs84_bounds(season_files['spring'])
            collections.append({
                'name': f"{folder}_{grid_id}",
                'files': season_files,
                'bounds': wgs84_bounds, # (w, s, e, n)
                'crs': crs
            })
    return collections

j_colls = index_spain('spain_j_images')
co_colls = index_spain('spain_phase2_images')

print(f"Found {len(j_colls)} complete 4-season tiles in spain_j_images")
for c in j_colls:
    print(c['name'], c['bounds'], c['crs'])

print(f"Found {len(co_colls)} complete 4-season tiles in spain_phase2_images")
for c in co_colls:
    print(c['name'], c['bounds'], c['crs'])
