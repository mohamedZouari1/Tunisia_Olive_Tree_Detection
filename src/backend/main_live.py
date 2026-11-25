"""
Zitoun Eye LIVE — AI Inference Backend with Google Earth Engine
Fetches 4-season Sentinel-2 composites on-the-fly from GEE and runs olive detection.

GEE Project: projet1-498400
Max area: 0.15° x 0.15° (~16km²) to keep download times short (<30s)
"""
import os, sys, io, base64, time, tempfile, zipfile
import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from PIL import Image
import uvicorn
import requests

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(ROOT, 'scripts'))

CHECKPOINT  = os.path.join(ROOT, 'checkpoints', 'final_20260620_104109_best.pth')
NORM_STATS  = os.path.join(ROOT, 'data', 'patches', 'normalize_stats.npz')
GEE_PROJECT = 'projet1-498400'

# Max ~0.15° per side to keep download fast (<30s on a typical connection)
MAX_DEGREES = 0.15

# ── RGBA class colors ─────────────────────────────────────────────────────────
CLASS_COLORS = [
    (0,   0,   0,   0),   # background – transparent
    (56,  120, 22,  210),  # olive_grove
    (140, 100, 22,  200),  # vineyard
    (22,  130, 80,  200),  # other_orchard
    (195, 158, 48,  200),  # arable_crop
    (215, 68,  68,  200),  # building
    (110, 110, 110, 200),  # road
]
CLASS_NAMES  = ['background','olive_grove','vineyard','other_orchard','arable_crop','building','road']
CLASS_LABELS = ['Background','Olive Grove','Vineyard','Other Orchard','Arable Crop','Building','Road']

# ── Model (lazy load) ─────────────────────────────────────────────────────────
_model = _device = _norm = None

def get_model():
    global _model, _device, _norm
    if _model is None:
        from predict import load_model
        _device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"[ZitounEye-Live] Loading model on {_device}...")
        _model = load_model(CHECKPOINT, _device)
        _norm  = np.load(NORM_STATS)
        print("[ZitounEye-Live] Model ready.")
    return _model, _device, _norm

# ── GEE (lazy init) ───────────────────────────────────────────────────────────
_ee = None

def get_ee():
    global _ee
    if _ee is None:
        import ee
        try:
            ee.Initialize(project=GEE_PROJECT)
            print("[ZitounEye-Live] GEE initialized.")
        except Exception:
            ee.Authenticate()
            ee.Initialize(project=GEE_PROJECT)
            print("[ZitounEye-Live] GEE initialized after auth.")
        _ee = ee
    return _ee

# ── Sentinel-2 helpers ────────────────────────────────────────────────────────
SEASON_MONTHS = {
    'spring': [3, 4, 5],
    'summer': [6, 7, 8],
    'fall':   [9, 10, 11],
    'winter': [12, 1, 2],
}

S2_BANDS = ['B2','B3','B4','B5','B6','B7','B8','B8A','B11','B12']

def cloud_mask(image):
    ee = get_ee()
    scl = image.select('SCL')
    clear = scl.eq(4).Or(scl.eq(5)).Or(scl.eq(6)).Or(scl.eq(7)).Or(scl.eq(2))
    return image.updateMask(clear)

def get_season_composite(aoi, season, year=2023):
    ee = get_ee()
    months = SEASON_MONTHS[season]

    if season == 'winter':
        start = f'{year-1}-12-01'
        end   = f'{year}-03-01'
    else:
        m0 = months[0]; m1 = months[-1]
        start = f'{year}-{m0:02d}-01'
        end   = f'{year}-{m1:02d}-28'

    col = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
             .filterDate(start, end)
             .filterBounds(aoi)
             .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 40))
             .map(cloud_mask))

    composite = col.select(S2_BANDS).median()

    # Compute indices
    B2 = composite.select('B2')
    B4 = composite.select('B4')
    B8 = composite.select('B8')
    B3 = composite.select('B3')

    ndvi = composite.normalizedDifference(['B8','B4']).rename('NDVI')
    evi  = composite.expression(
        '2.5 * ((NIR - RED) / (NIR + 6*RED - 7.5*BLUE + 1))',
        {'NIR': B8, 'RED': B4, 'BLUE': B2}
    ).rename('EVI')
    ndwi = composite.normalizedDifference(['B3','B8']).rename('NDWI')
    savi = composite.expression(
        '1.5 * ((NIR - RED) / (NIR + RED + 0.5))',
        {'NIR': B8, 'RED': B4}
    ).rename('SAVI')

    return composite.addBands([ndvi, evi, ndwi, savi])

def download_season_as_array(aoi, season, year=2023):
    """
    Download one seasonal composite as a numpy array [14, H, W].
    Uses GEE getDownloadURL → local ZIP → GeoTIFF → numpy.
    Scale=20m for speed (still 14 bands at 20m = ~excellent for model input).
    """
    import rasterio

    ee = get_ee()
    img = get_season_composite(aoi, season, year)
    bands_14 = S2_BANDS + ['NDVI','EVI','NDWI','SAVI']

    # Download via getDownloadURL (synchronous, no Drive needed)
    url = img.select(bands_14).getDownloadURL({
        'scale': 20,          # 20m resolution for speed
        'region': aoi,
        'format': 'GEO_TIFF',
        'crs': 'EPSG:4326',
    })

    print(f"[GEE] Downloading {season}… ", end='', flush=True)
    t0 = time.time()
    response = requests.get(url, timeout=120)
    response.raise_for_status()
    print(f"{time.time()-t0:.1f}s")

    # Write to a temp file and read back with rasterio
    with tempfile.NamedTemporaryFile(suffix='.tif', delete=False) as f:
        f.write(response.content)
        tmp_path = f.name

    with rasterio.open(tmp_path) as src:
        data = src.read().astype(np.float32)

    os.unlink(tmp_path)

    # Normalize: spectral bands /10000, indices already in [-1,1]
    data[:10] /= 10000.0
    np.nan_to_num(data, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    return data  # [14, H, W]


# ── Inference ─────────────────────────────────────────────────────────────────
def run_gee_inference(w, s, e, n, year=2023):
    ee = get_ee()
    from predict import predict_image

    # Clamp to max area
    mid_lon = (w + e) / 2
    mid_lat = (s + n) / 2
    half    = MAX_DEGREES / 2
    w2 = max(w, mid_lon - half); e2 = min(e, mid_lon + half)
    s2 = max(s, mid_lat - half); n2 = min(n, mid_lat + half)

    aoi = ee.Geometry.Rectangle([w2, s2, e2, n2])

    # Download 4 seasons
    season_arrays = []
    for season in ['spring', 'summer', 'fall', 'winter']:
        arr = download_season_as_array(aoi, season, year)
        if arr.size == 0:
            raise ValueError(f"No satellite data for season {season} — try another area or year.")
        season_arrays.append(arr)

    # Align shapes (resample to smallest common size)
    min_h = min(a.shape[1] for a in season_arrays)
    min_w = min(a.shape[2] for a in season_arrays)
    season_arrays = [a[:, :min_h, :min_w] for a in season_arrays]

    stacked = np.concatenate(season_arrays, axis=0)  # [56, H, W]
    if stacked.shape[1] < 4 or stacked.shape[2] < 4:
        raise ValueError("Selected area too small after download. Draw a bigger rectangle.")

    model, device, norm = get_model()
    pred = predict_image(model, stacked, norm, device)

    # Colorize → RGBA PNG
    rgba = np.zeros((pred.shape[0], pred.shape[1], 4), dtype=np.uint8)
    for i, c in enumerate(CLASS_COLORS):
        rgba[pred == i] = c

    buf = io.BytesIO()
    Image.fromarray(rgba, 'RGBA').save(buf, format='PNG')
    img_b64 = base64.b64encode(buf.getvalue()).decode()

    total = pred.size
    stats = [
        {
            'class': CLASS_NAMES[i],
            'label': CLASS_LABELS[i],
            'pct':   round(float((pred == i).sum() / total * 100), 2),
            'color': f"rgb({CLASS_COLORS[i][0]},{CLASS_COLORS[i][1]},{CLASS_COLORS[i][2]})",
        }
        for i in range(len(CLASS_NAMES))
    ]

    return {
        'image_b64': img_b64,
        'bounds': {'west': w2, 'south': s2, 'east': e2, 'north': n2},
        'width': stacked.shape[2], 'height': stacked.shape[1],
        'stats': stats,
        'resolution_m': 20,
        'year': year,
    }

# ── FastAPI ───────────────────────────────────────────────────────────────────
app = FastAPI(title="Zitoun Eye LIVE", version="2.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True,
                   allow_methods=["*"], allow_headers=["*"])

frontend_dir = os.path.join(os.path.dirname(__file__), '..', 'frontend_live')
if os.path.isdir(frontend_dir):
    app.mount("/app", StaticFiles(directory=frontend_dir, html=True), name="frontend_live")

class BBoxRequest(BaseModel):
    west:  float
    south: float
    east:  float
    north: float
    year:  int = 2023

@app.get("/api/health")
def health():
    ee_ok = False
    try:
        get_ee()
        ee_ok = True
    except Exception:
        pass
    return {
        "status": "online",
        "gee_project": GEE_PROJECT,
        "gee_ready": ee_ok,
        "model": "final_20260620_104109_best.pth",
        "max_area_deg": MAX_DEGREES,
    }

@app.post("/api/analyze")
def analyze(req: BBoxRequest):
    t0 = time.time()
    try:
        result = run_gee_inference(req.west, req.south, req.east, req.north, req.year)
        result['elapsed_s'] = round(time.time() - t0, 2)
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        import traceback; traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"GEE/Inference error: {e}")

if __name__ == "__main__":
    print("\n🌍 Zitoun Eye LIVE — Real-Time GEE Backend")
    print(f"   GEE Project: {GEE_PROJECT}")
    print(f"   Model:       {CHECKPOINT}")
    print(f"   Max area:    {MAX_DEGREES}° × {MAX_DEGREES}° (~{MAX_DEGREES*111:.0f}km per side)")
    print("   Serving:     http://localhost:8001\n")
    uvicorn.run(app, host="0.0.0.0", port=8001, reload=False)
