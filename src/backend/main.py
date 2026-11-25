"""
Zitoun Eye — AI Inference Backend
Serves 4-season olive detection using EfficientNet-B4 U-Net (IoU 0.68)
Supports both single-file (Tunisia WGS84) and true 4-season stacks (Spain UTM 30N)
"""
import os, sys, io, base64, time
import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import rasterio
from rasterio.windows import from_bounds as rio_from_bounds
from pyproj import Transformer
from PIL import Image
import uvicorn

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT       = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(ROOT, 'scripts'))

CHECKPOINT = os.path.join(ROOT, 'checkpoints', 'final_20260620_104109_best.pth')
NORM_STATS = os.path.join(ROOT, 'data', 'patches', 'normalize_stats.npz')

# RGBA colors per class (semi-transparent overlays)
CLASS_COLORS = [
    (0,   0,   0,   0),   # 0 background – fully transparent
    (56,  120, 22,  210),  # 1 olive_grove – deep green
    (140, 100, 22,  200),  # 2 vineyard – amber
    (22,  130, 80,  200),  # 3 other_orchard – teal
    (195, 158, 48,  200),  # 4 arable_crop – golden
    (215, 68,  68,  200),  # 5 building – red
    (110, 110, 110, 200),  # 6 road – grey
]
CLASS_NAMES  = ['background','olive_grove','vineyard','other_orchard','arable_crop','building','road']
CLASS_LABELS = ['Background','Olive Grove','Vineyard','Other Orchard','Arable Crop','Building','Road']

# ── Tile index (WGS84 bounds) ──────────────────────────────────────────────────
# Hardcoded from previous analysis for stability.
INDEX = [
    # Tunisia WGS84 single-season tiles
    {'type': 'single', 'path': os.path.join(ROOT, 'S2_olive_test.tif'),     'w': 10.0500, 's': 34.6999, 'e': 10.2500, 'n': 34.8501, 'crs': 'EPSG:4326'},
    {'type': 'single', 'path': os.path.join(ROOT, 'S2_olive_test (1).tif'), 'w': 10.5999, 's': 34.6500, 'e': 10.8001, 'n': 34.8000, 'crs': 'EPSG:4326'},
    {'type': 'single', 'path': os.path.join(ROOT, 'S2_olive_test (2).tif'), 'w': 10.4578, 's': 36.6565, 'e': 10.4779, 'n': 36.6766, 'crs': 'EPSG:4326'},
    {'type': 'single', 'path': os.path.join(ROOT, 'S2_olive_test (3).tif'), 'w': 10.2919, 's': 36.6148, 'e': 10.3420, 'n': 36.6649, 'crs': 'EPSG:4326'},
    {'type': 'single', 'path': os.path.join(ROOT, 'S2_olive_test (4).tif'), 'w': 8.8761,  's': 33.5991, 'e': 9.0761,  'n': 33.7992, 'crs': 'EPSG:4326'},
    
    # Spain UTM 30N 4-season groups
    {'type': 'stack', 'name': 'es_j_2023_8960_8960', 'w': -3.1078, 's': 37.3909, 'e': -2.4973, 'n': 37.7868, 'crs': 'EPSG:32630',
     'files': [
         os.path.join(ROOT, 'spain_j_images', 'es_j_2023_spring_2023-0000008960-0000008960.tif'),
         os.path.join(ROOT, 'spain_j_images', 'es_j_2023_summer_2023-0000008960-0000008960.tif'),
         os.path.join(ROOT, 'spain_j_images', 'es_j_2023_fall_2023-0000008960-0000008960.tif'),
         os.path.join(ROOT, 'spain_j_images', 'es_j_2023_winter_2023-0000008960-0000008960.tif')
     ]},
    {'type': 'stack', 'name': 'es_co_2023_0_0_001', 'w': -5.6743, 's': 37.9221, 'e': -4.6739, 'n': 38.7480, 'crs': 'EPSG:32630',
     'files': [
         os.path.join(ROOT, 'spain_phase2_images', 'es_co_2023_spring_2023-0000000000-0000000000-001.tif'),
         os.path.join(ROOT, 'spain_phase2_images', 'es_se_2023_summer_2023-0000000000-0000000000-001.tif'),
         os.path.join(ROOT, 'spain_phase2_images', 'es_co_2023_fall_2023-0000000000-0000000000-001.tif'),
         os.path.join(ROOT, 'spain_phase2_images', 'es_se_2023_winter_2023-0000000000-0000000000-001.tif')
     ]},
    {'type': 'stack', 'name': 'es_se_2023_8960_8960', 'w': -5.5351, 's': 36.8274, 'e': -4.8408, 'n': 37.2514, 'crs': 'EPSG:32630',
     'files': [
         os.path.join(ROOT, 'spain_phase2_images', 'es_se_2023_spring_2023-0000008960-0000008960.tif'),
         os.path.join(ROOT, 'spain_phase2_images', 'es_se_2023_summer_2023-0000008960-0000008960.tif'),
         os.path.join(ROOT, 'spain_phase2_images', 'es_se_2023_fall_2023-0000008960-0000008960.tif'),
         os.path.join(ROOT, 'spain_phase2_images', 'es_se_2023_winter_2023-0000008960-0000008960.tif')
     ]}
]

# ── Model (lazy load on first request) ────────────────────────────────────────
_model = _device = _norm = None

def get_model():
    global _model, _device, _norm
    if _model is None:
        from predict import load_model
        _device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"[ZitounEye] Loading model on {_device}...")
        _model = load_model(CHECKPOINT, _device)
        _norm  = np.load(NORM_STATS)
        print("[ZitounEye] Model ready.")
    return _model, _device, _norm

# ── Helpers ───────────────────────────────────────────────────────────────────
def overlap(req_w, req_s, req_e, req_n, t):
    ow = min(req_e, t['e']) - max(req_w, t['w'])
    oh = min(req_n, t['n']) - max(req_s, t['s'])
    return ow * oh if ow > 0 and oh > 0 else 0

def best_tile(w, s, e, n):
    scored = [(overlap(w,s,e,n,t), t) for t in INDEX]
    scored = [(sc, t) for sc,t in scored if sc > 0]
    return max(scored, key=lambda x: x[0])[1] if scored else None

def compute_indices(data):
    B2, B4, B8 = data[0], data[2], data[6]
    eps = 1e-10
    NDVI = (B8-B4)/(B8+B4+eps)
    EVI  = 2.5*(B8-B4)/(B8+6*B4-7.5*B2+1+eps)
    NDWI = (data[1]-B8)/(data[1]+B8+eps)
    SAVI = 1.5*(B8-B4)/(B8+B4+0.5+eps)
    return np.concatenate([data[:10], NDVI[None], EVI[None], NDWI[None], SAVI[None]], axis=0)

# ── Inference ─────────────────────────────────────────────────────────────────
def run_inference(w, s, e, n):
    tile = best_tile(w, s, e, n)
    if tile is None:
        raise ValueError(
            "No satellite data covers your selection. "
            "Draw within one of the highlighted green zones on the map."
        )

    model, device, norm = get_model()
    from predict import predict_image

    # Clamp bbox to tile bounds (WGS84)
    w2 = max(w, tile['w']); s2 = max(s, tile['s'])
    e2 = min(e, tile['e']); n2 = min(n, tile['n'])

    # Prepare window bounds based on CRS
    if tile['crs'] != 'EPSG:4326':
        transformer = Transformer.from_crs('EPSG:4326', tile['crs'], always_xy=True)
        tw_w, tw_s = transformer.transform(w2, s2)
        tw_e, tw_n = transformer.transform(e2, n2)
    else:
        tw_w, tw_s, tw_e, tw_n = w2, s2, e2, n2

    if tile['type'] == 'single':
        with rasterio.open(tile['path']) as src:
            window = rio_from_bounds(tw_w, tw_s, tw_e, tw_n, src.transform)
            data = src.read(window=window).astype(np.float32)

        if data.size == 0:
            raise ValueError("Selection resulted in an empty area. Draw a larger rectangle inside the green zone.")
        if data.max() > 10: data /= 10000.0
        np.nan_to_num(data, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        
        d14 = compute_indices(data)
        stacked = np.concatenate([d14]*4, axis=0)
        tile_name = os.path.basename(tile['path'])
        H, W = data.shape[1], data.shape[2]

    else: # 'stack' (Spain)
        season_data = []
        H, W = 0, 0
        for fpath in tile['files']:
            if not os.path.exists(fpath):
                raise ValueError(f"Missing season file: {fpath}")
            with rasterio.open(fpath) as src:
                window = rio_from_bounds(tw_w, tw_s, tw_e, tw_n, src.transform)
                data = src.read(window=window).astype(np.float32)
                H, W = data.shape[1], data.shape[2]
            
            if data.size == 0:
                raise ValueError("Selection resulted in an empty area. Draw a larger rectangle inside the green zone.")
            if data.max() > 10: data /= 10000.0
            np.nan_to_num(data, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
            
            d14 = compute_indices(data)
            season_data.append(d14)
        
        stacked = np.concatenate(season_data, axis=0)
        tile_name = tile['name']

    if H < 4 or W < 4:
        raise ValueError("Selection too small — please draw a larger rectangle.")

    pred = predict_image(model, stacked, norm, device)

    # Colorize to RGBA PNG
    rgba = np.zeros((pred.shape[0], pred.shape[1], 4), dtype=np.uint8)
    for i, c in enumerate(CLASS_COLORS):
        rgba[pred == i] = c

    buf = io.BytesIO()
    Image.fromarray(rgba, 'RGBA').save(buf, format='PNG')
    img_b64 = base64.b64encode(buf.getvalue()).decode()

    # Per-class stats
    total = pred.size
    stats = [
        {
            "class": CLASS_NAMES[i],
            "label": CLASS_LABELS[i],
            "pct":   round(float((pred == i).sum() / total * 100), 2),
            "color": f"rgb({CLASS_COLORS[i][0]},{CLASS_COLORS[i][1]},{CLASS_COLORS[i][2]})",
        }
        for i in range(len(CLASS_NAMES))
    ]

    return {
        "image_b64": img_b64,
        "bounds": {"west": w2, "south": s2, "east": e2, "north": n2},
        "width": W, "height": H,
        "stats": stats,
        "tile": tile_name,
    }

# ── FastAPI App ───────────────────────────────────────────────────────────────
app = FastAPI(title="Zitoun Eye Engine", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

frontend_dir = os.path.join(os.path.dirname(__file__), '..', 'frontend')
if os.path.isdir(frontend_dir):
    app.mount("/app", StaticFiles(directory=frontend_dir, html=True), name="frontend")

class BBoxRequest(BaseModel):
    west:  float
    south: float
    east:  float
    north: float

@app.get("/api/health")
def health():
    return {
        "status": "online",
        "model": "final_20260620_104109_best.pth",
        "tiles_available": len(INDEX),
    }

@app.get("/api/coverage")
def get_coverage():
    return [
        {"west": t['w'], "south": t['s'], "east": t['e'], "north": t['n'], "name": t.get('name', 'tile')}
        for t in INDEX
    ]

@app.post("/api/analyze")
def analyze(req: BBoxRequest):
    t0 = time.time()
    try:
        result = run_inference(req.west, req.south, req.east, req.north)
        result["elapsed_s"] = round(time.time() - t0, 2)
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Inference error: {e}")

if __name__ == "__main__":
    print("\n🫒 Zitoun Eye AI Backend")
    print(f"   Model:    {CHECKPOINT}")
    print(f"   Tiles:    {len(INDEX)} available")
    print("   Serving:  http://localhost:8000\n")
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)
