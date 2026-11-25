#!/usr/bin/env python3
"""
Olive Grove Detection — Web Inference Server
=============================================
Flask server that serves a web UI for testing the trained U-Net model
on new Sentinel-2 GeoTIFF images.

Usage:
  python app.py
  → Open http://localhost:5000
"""

import os
import sys
import json
import uuid
import numpy as np
import torch
import torch.nn.functional as F
import rasterio
from rasterio.windows import Window
from pathlib import Path
from flask import Flask, request, jsonify, send_from_directory, send_file
import logging
import base64
from io import BytesIO

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s | %(levelname)s | %(message)s')
log = logging.getLogger(__name__)

app = Flask(__name__, static_folder='web', static_url_path='')

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
CHIP_SIZE = 128
OVERLAP = 32
N_BANDS = 14
N_CLASSES = 3
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
UPLOAD_DIR = Path('uploads')
RESULT_DIR = Path('results')
UPLOAD_DIR.mkdir(exist_ok=True)
RESULT_DIR.mkdir(exist_ok=True)

# Global model
MODEL = None
BAND_MEAN = None
BAND_STD = None


def load_model(checkpoint_path='best_unet_olive.pth'):
    """Load the trained model."""
    global MODEL, BAND_MEAN, BAND_STD

    if not os.path.exists(checkpoint_path):
        log.warning(f'Checkpoint not found: {checkpoint_path}')
        return False

    ckpt = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
    n_bands = ckpt.get('n_bands', N_BANDS)
    n_classes = ckpt.get('n_classes', N_CLASSES)

    try:
        import segmentation_models_pytorch as smp
        MODEL = smp.Unet(
            encoder_name="resnet34",
            encoder_weights=None,
            in_channels=n_bands,
            classes=n_classes,
            activation=None,
        ).to(DEVICE)
        log.info('Using smp.Unet architecture')
    except ImportError:
        from train_unet_olive import UNet
        MODEL = UNet(in_channels=n_bands, out_channels=n_classes).to(DEVICE)
        log.info('Using custom UNet architecture')

    MODEL.load_state_dict(ckpt['model_state'])
    MODEL.eval()

    BAND_MEAN = np.array(ckpt.get('band_mean', [0]*14), dtype=np.float32)[:, None, None]
    BAND_STD = np.array(ckpt.get('band_std', [1]*14), dtype=np.float32)[:, None, None]

    log.info(f'Model loaded: epoch={ckpt.get("epoch")}, '
             f'mIoU={ckpt.get("miou", "?"):.4f}, device={DEVICE}')
    return True


def compute_indices(bands_10):
    """
    Compute NDVI, NDWI, EVI, SAVI from 10 raw Sentinel-2 bands.
    Input order: B2,B3,B4,B5,B6,B7,B8,B8A,B11,B12
    Returns: 14-band stack with indices appended.
    """
    B3  = bands_10[1]   # Green
    B4  = bands_10[2]   # Red
    B8  = bands_10[6]   # NIR
    B11 = bands_10[8]   # SWIR

    eps = 1e-8
    ndvi = (B8 - B4) / (B8 + B4 + eps)
    ndwi = (B3 - B8) / (B3 + B8 + eps)
    evi  = 2.5 * (B8 - B4) / (B8 + 6*B4 - 7.5*B3 + 1 + eps)
    savi = 1.5 * (B8 - B4) / (B8 + B4 + 0.5 + eps)

    return np.concatenate([
        bands_10,
        ndvi[np.newaxis], ndwi[np.newaxis],
        evi[np.newaxis], savi[np.newaxis]
    ], axis=0)


@torch.no_grad()
def run_inference(image_path):
    """Run sliding-window inference on a GeoTIFF."""
    if MODEL is None:
        raise RuntimeError('Model not loaded')

    with rasterio.open(image_path) as src:
        H, W = src.height, src.width
        profile = src.profile
        full = src.read().astype(np.float32)
        n_bands = src.count

    # Handle NaN
    full = np.nan_to_num(full, nan=0.0)

    # Auto-compute indices if user uploaded 10 raw bands
    if n_bands == 10:
        log.info('10-band input detected — auto-computing NDVI, NDWI, EVI, SAVI')
        full = compute_indices(full)
        n_bands = 14
    elif n_bands != 14:
        raise ValueError(
            f'Expected 10 or 14 bands, got {n_bands}. '
            f'Upload a Sentinel-2 GeoTIFF with bands B2-B12 + indices.'
        )

    # Normalize
    mean = BAND_MEAN[:n_bands]
    std = BAND_STD[:n_bands]
    full_norm = (full - mean) / (std + 1e-8)

    # Sliding window
    step = CHIP_SIZE - OVERLAP
    prob_acc = np.zeros((N_CLASSES, H, W), np.float32)
    count_acc = np.zeros((H, W), np.float32)

    ys = sorted(set(list(range(0, max(1, H - CHIP_SIZE + 1), step)) + [max(0, H - CHIP_SIZE)]))
    xs = sorted(set(list(range(0, max(1, W - CHIP_SIZE + 1), step)) + [max(0, W - CHIP_SIZE)]))

    total = len(ys) * len(xs)
    done = 0

    for y in ys:
        for x in xs:
            chip = full_norm[:, y:y+CHIP_SIZE, x:x+CHIP_SIZE]
            # Pad if necessary
            _, ch, cw = chip.shape
            if ch < CHIP_SIZE or cw < CHIP_SIZE:
                padded = np.zeros((n_bands, CHIP_SIZE, CHIP_SIZE), dtype=np.float32)
                padded[:, :ch, :cw] = chip
                chip = padded

            t = torch.from_numpy(chip).unsqueeze(0).to(DEVICE)
            prob = F.softmax(MODEL(t), dim=1).squeeze(0).cpu().numpy()
            prob_acc[:, y:y+ch, x:x+cw] += prob[:, :ch, :cw]
            count_acc[y:y+ch, x:x+cw] += 1
            done += 1

    # Avoid division by zero
    count_acc = np.maximum(count_acc, 1)
    prob_acc /= count_acc[np.newaxis]

    pred_mask = prob_acc.argmax(axis=0).astype(np.uint8)
    confidence = prob_acc.max(axis=0)

    return pred_mask, confidence, profile, full


def mask_to_rgba(mask, alpha=180):
    """Convert class mask to RGBA overlay image."""
    H, W = mask.shape
    rgba = np.zeros((H, W, 4), dtype=np.uint8)

    # Class 1: Extensif → green
    rgba[mask == 1] = [76, 175, 80, alpha]
    # Class 2: Intensif → orange
    rgba[mask == 2] = [255, 152, 0, alpha]
    # Class 0: Background → transparent
    rgba[mask == 0] = [0, 0, 0, 0]

    return rgba


def create_rgb_preview(full_image, bands=(2, 1, 0)):
    """Create an RGB preview from the 14-band image (using B4, B3, B2)."""
    rgb = np.stack([full_image[b] for b in bands], axis=-1)
    # Percentile stretch for better visualization
    for i in range(3):
        lo, hi = np.percentile(rgb[:, :, i][rgb[:, :, i] > 0], [2, 98])
        if hi > lo:
            rgb[:, :, i] = np.clip((rgb[:, :, i] - lo) / (hi - lo), 0, 1)
    return (rgb * 255).astype(np.uint8)


def array_to_png_base64(arr):
    """Convert numpy array to base64 PNG string."""
    from PIL import Image
    if arr.ndim == 2:
        img = Image.fromarray(arr)
    elif arr.shape[2] == 4:
        img = Image.fromarray(arr, 'RGBA')
    else:
        img = Image.fromarray(arr, 'RGB')

    # Resize if too large for web display
    max_dim = 1200
    if max(img.size) > max_dim:
        ratio = max_dim / max(img.size)
        new_size = (int(img.size[0] * ratio), int(img.size[1] * ratio))
        img = img.resize(new_size, Image.NEAREST)

    buf = BytesIO()
    img.save(buf, format='PNG')
    return base64.b64encode(buf.getvalue()).decode('utf-8')


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.route('/')
def index():
    return send_from_directory('web', 'index.html')


@app.route('/api/status')
def status():
    return jsonify({
        'model_loaded': MODEL is not None,
        'device': DEVICE,
        'cuda_available': torch.cuda.is_available(),
        'gpu_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    })


@app.route('/api/predict', methods=['POST'])
def predict():
    if MODEL is None:
        return jsonify({'error': 'Model not loaded. Train the model first.'}), 503

    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if not file.filename.endswith(('.tif', '.tiff')):
        return jsonify({'error': 'Please upload a GeoTIFF file (.tif)'}), 400

    # Save uploaded file
    file_id = str(uuid.uuid4())[:8]
    upload_path = UPLOAD_DIR / f'{file_id}_{file.filename}'
    file.save(upload_path)

    try:
        # Run inference
        log.info(f'Running inference on {file.filename}...')
        mask, confidence, profile, full_image = run_inference(str(upload_path))

        # Save mask as GeoTIFF
        mask_path = RESULT_DIR / f'{file_id}_mask.tif'
        profile.update(count=1, dtype='uint8', compress='lzw', nodata=255)
        with rasterio.open(str(mask_path), 'w', **profile) as dst:
            dst.write(mask[np.newaxis])

        # Create visualizations
        rgb_preview = create_rgb_preview(full_image)
        mask_overlay = mask_to_rgba(mask)

        # Statistics
        total_px = mask.size
        classes, counts = np.unique(mask, return_counts=True)
        class_stats = {}
        labels = {0: 'Background', 1: 'Olive Extensif', 2: 'Olive Intensif'}
        colors = {0: '#1a1a2e', 1: '#4CAF50', 2: '#FF9800'}
        for c, cnt in zip(classes, counts):
            class_stats[int(c)] = {
                'label': labels.get(c, f'Class {c}'),
                'pixels': int(cnt),
                'percentage': round(cnt / total_px * 100, 2),
                'area_ha': round(cnt * 0.01, 2),  # 10m × 10m = 100m² = 0.01 ha
                'color': colors.get(c, '#888'),
            }

        result = {
            'success': True,
            'file_id': file_id,
            'filename': file.filename,
            'dimensions': f'{mask.shape[1]}×{mask.shape[0]}',
            'total_pixels': total_px,
            'mean_confidence': round(float(confidence.mean()) * 100, 1),
            'class_stats': class_stats,
            'rgb_image': array_to_png_base64(rgb_preview),
            'mask_image': array_to_png_base64(mask_overlay),
            'mask_download': f'/api/download/{file_id}_mask.tif',
        }

        log.info(f'Inference complete: {mask.shape[1]}×{mask.shape[0]}, '
                 f'confidence={result["mean_confidence"]}%')
        return jsonify(result)

    except Exception as e:
        log.exception(f'Inference failed: {e}')
        return jsonify({'error': str(e)}), 500
    finally:
        # Clean up upload
        if upload_path.exists():
            upload_path.unlink()


@app.route('/api/download/<filename>')
def download(filename):
    return send_from_directory(str(RESULT_DIR), filename, as_attachment=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    # Try to load model
    checkpoint = 'best_unet_olive.pth'
    if os.path.exists(checkpoint):
        load_model(checkpoint)
    else:
        log.warning(f'No checkpoint found at {checkpoint}. '
                    f'Train the model first, then restart.')

    log.info(f'Starting server on http://localhost:5000')
    app.run(host='0.0.0.0', port=5000, debug=False)
