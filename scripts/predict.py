"""
ZitounEye - Olive Grove Detector
================================
Usage:  python scripts/predict.py --input your_image.tif
Output: your_image_prediction.tif  (GeoTIFF with color overlay)
        your_image_prediction.png  (visual preview)

The model was trained on 4-season 56-band input.
For single-season input, we replicate the image across all 4 slots.
This still works well because the spectral pattern of olive trees is very distinct.
"""

import sys
import os
import argparse
import warnings
warnings.filterwarnings("ignore")

import io
try:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
except Exception:
    pass

import numpy as np
import torch
import rasterio
from rasterio.windows import Window
import glob

# ---------------------------------------------------------------
# Config
# ---------------------------------------------------------------
CHECKPOINTS_DIR = r"c:\Users\AORUS\Documents\zitoun_eye\checkpoints"
STATS_FILE      = r"c:\Users\AORUS\Documents\zitoun_eye\data\patches\normalize_stats.npz"
PATCH_SIZE      = 256
OVERLAP         = 64
NUM_CLASSES     = 7
MODEL_BANDS     = 56  # 14 bands x 4 seasons

CLASS_NAMES = [
    "background", "olive_grove", "vineyard",
    "other_orchard", "arable_crop", "building", "road"
]

# Color map: BGR for each class
CLASS_COLORS_RGB = {
    0: (30,  30,  30),    # background   -> dark grey
    1: (0,   200,  60),   # olive_grove  -> bright green
    2: (180, 60,  220),   # vineyard     -> purple
    3: (255, 140,  0),    # other_orchard-> orange
    4: (255, 220, 50),    # arable_crop  -> yellow
    5: (220,  50,  50),   # building     -> red
    6: (120, 120, 120),   # road         -> grey
}


def find_checkpoint():
    """Find best Phase 1 checkpoint automatically."""
    pattern = os.path.join(CHECKPOINTS_DIR, "phase1_*_best.pth")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(
            f"No checkpoint found in {CHECKPOINTS_DIR}. Train the model first!"
        )
    return files[-1]  # most recent


def load_model(checkpoint_path, device):
    """Load trained U-Net model."""
    try:
        import segmentation_models_pytorch as smp
    except ImportError:
        raise ImportError("Run: pip install segmentation-models-pytorch")

    model = smp.Unet(
        encoder_name="efficientnet-b4",
        encoder_weights=None,
        in_channels=MODEL_BANDS,
        classes=NUM_CLASSES,
    )
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state)
    model.eval()
    model.to(device)
    print(f"  Loaded: {os.path.basename(checkpoint_path)}")
    return model


def build_56band_input(tif_path):
    """
    Read a single GeoTIFF and build the 56-band model input.
    - If 14 bands: perfect single season -> replicate x4
    - If 10 bands: standard S2 -> compute NDVI,EVI,NDWI,SAVI then replicate x4
    - If 56 bands: already stacked multi-season, use as-is
    """
    with rasterio.open(tif_path) as src:
        data = src.read().astype(np.float32)
        meta = src.meta.copy()
        h, w = src.height, src.width

    n_bands = data.shape[0]
    print(f"  Input: {w}x{h} pixels, {n_bands} bands")

    # Scale to reflectance if values look like DN (>10)
    if data.max() > 10:
        data = data / 10000.0

    np.nan_to_num(data, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

    if n_bands == MODEL_BANDS:
        print("  -> Already 56 bands (multi-season stack). Using as-is.")
        return data, meta

    # Compute vegetation indices if we have standard S2 bands
    # Assume band order from GEE: B2,B3,B4,B5,B6,B7,B8,B8A,B11,B12
    if n_bands >= 10:
        B4  = data[2]   # Red
        B8  = data[6]   # NIR
        B11 = data[8]   # SWIR1
        B2  = data[0]   # Blue

        eps = 1e-10
        NDVI = (B8 - B4) / (B8 + B4 + eps)
        EVI  = 2.5 * (B8 - B4) / (B8 + 6*B4 - 7.5*B2 + 1 + eps)
        NDWI = (data[1] - B8) / (data[1] + B8 + eps)  # B3/NIR
        SAVI = 1.5 * (B8 - B4) / (B8 + B4 + 0.5 + eps)

        one_season = np.concatenate([
            data[:10],
            NDVI[np.newaxis], EVI[np.newaxis],
            NDWI[np.newaxis], SAVI[np.newaxis],
        ], axis=0)  # (14, H, W)
        print("  -> 10-band input -> computed 4 indices -> 14 bands")
    elif n_bands >= 4:
        # Minimal bands, pad to 14
        pad = np.zeros((14 - n_bands, h, w), dtype=np.float32)
        one_season = np.concatenate([data[:n_bands], pad], axis=0)
        print(f"  -> {n_bands} bands -> padded to 14 bands")
    else:
        one_season = data[:n_bands]
        pad = np.zeros((14 - n_bands, h, w), dtype=np.float32)
        one_season = np.concatenate([one_season, pad], axis=0)

    # Replicate across 4 seasons -> 56 bands
    stacked = np.concatenate([one_season] * 4, axis=0)  # (56, H, W)
    print(f"  -> Replicated to 56 bands (4 x single season)")
    return stacked, meta


def predict_image(model, image_56band, norm_stats, device):
    """
    Run sliding window inference on the full image.
    Returns (H, W) prediction map.
    """
    C, H, W = image_56band.shape
    assert C == MODEL_BANDS, f"Expected {MODEL_BANDS} bands, got {C}"

    # Normalize
    mean = norm_stats["mean"][:, np.newaxis, np.newaxis]
    std  = norm_stats["std"][:, np.newaxis, np.newaxis]
    std  = np.where(std < 1e-6, 1.0, std)
    image_norm = (image_56band - mean) / std

    # Accumulate logits with overlap averaging
    pred_sum   = np.zeros((NUM_CLASSES, H, W), dtype=np.float32)
    pred_count = np.zeros((H, W), dtype=np.float32)

    stride = PATCH_SIZE - OVERLAP

    with torch.no_grad():
        patches = []
        coords  = []

        for y in range(0, H, stride):
            for x in range(0, W, stride):
                y2 = min(y + PATCH_SIZE, H)
                x2 = min(x + PATCH_SIZE, W)
                y1 = max(0, y2 - PATCH_SIZE)
                x1 = max(0, x2 - PATCH_SIZE)

                patch = image_norm[:, y1:y2, x1:x2]
                ph, pw = patch.shape[1], patch.shape[2]

                # Pad if needed
                if ph < PATCH_SIZE or pw < PATCH_SIZE:
                    pad_patch = np.zeros((MODEL_BANDS, PATCH_SIZE, PATCH_SIZE), dtype=np.float32)
                    pad_patch[:, :ph, :pw] = patch
                    patch = pad_patch

                patches.append(patch)
                coords.append((y1, x1, ph, pw))

                # Process in batches of 8
                if len(patches) == 8:
                    batch = torch.tensor(np.stack(patches), dtype=torch.float32).to(device)
                    out = torch.softmax(model(batch), dim=1).cpu().numpy()
                    for i, (py, px, pph, ppw) in enumerate(coords):
                        pred_sum[:, py:py+pph, px:px+ppw] += out[i, :, :pph, :ppw]
                        pred_count[py:py+pph, px:px+ppw]  += 1
                    patches = []
                    coords  = []

        # Remaining patches
        if patches:
            batch = torch.tensor(np.stack(patches), dtype=torch.float32).to(device)
            out = torch.softmax(model(batch), dim=1).cpu().numpy()
            for i, (py, px, pph, ppw) in enumerate(coords):
                pred_sum[:, py:py+pph, px:px+ppw] += out[i, :, :pph, :ppw]
                pred_count[py:py+pph, px:px+ppw]  += 1

    # Average overlapping predictions
    pred_count = np.where(pred_count == 0, 1, pred_count)
    pred_map = (pred_sum / pred_count[np.newaxis]).argmax(axis=0).astype(np.uint8)
    
    return pred_map


def save_outputs(pred_map, meta, input_path):
    """Save GeoTIFF + PNG visualization."""
    base = os.path.splitext(input_path)[0]
    out_tif = base + "_prediction.tif"
    out_png = base + "_prediction.png"

    # Save GeoTIFF (class IDs, georeferenced)
    out_meta = meta.copy()
    out_meta.update(count=1, dtype=rasterio.uint8)
    with rasterio.open(out_tif, "w", **out_meta) as dst:
        dst.write(pred_map[np.newaxis])
    print(f"  Saved GeoTIFF: {out_tif}")

    # Save color PNG
    H, W = pred_map.shape
    color_img = np.zeros((H, W, 3), dtype=np.uint8)
    for cls_id, rgb in CLASS_COLORS_RGB.items():
        mask = pred_map == cls_id
        color_img[mask] = rgb

    try:
        from PIL import Image
        img = Image.fromarray(color_img)
        img.save(out_png)
        print(f"  Saved PNG:     {out_png}")
    except ImportError:
        # Fallback: use matplotlib
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 1, figsize=(12, 12))
        ax.imshow(color_img)
        ax.set_title("Olive Grove Detection\n(Green=Olive, Purple=Vineyard, Orange=Other Orchard, Yellow=Arable)")
        ax.axis("off")
        plt.tight_layout()
        plt.savefig(out_png, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved PNG:     {out_png}")

    return out_tif, out_png


def summarize_predictions(pred_map, input_path):
    """Print a summary of detected classes."""
    H, W = pred_map.shape
    total = H * W
    print()
    print("  Detection Summary:")
    print(f"  {'Class':<18} {'Pixels':>10} {'Area %':>8}  {'Result'}")
    print(f"  {'-'*52}")
    any_olive = False
    for cls_id, name in enumerate(CLASS_NAMES):
        count = (pred_map == cls_id).sum()
        pct   = count / total * 100
        if pct > 0.1:
            marker = ""
            if cls_id == 1:
                marker = "  <-- OLIVE GROVES DETECTED!" if pct > 1.0 else "  <-- trace amount"
                any_olive = True
            print(f"  {name:<18} {count:>10,} {pct:>7.1f}%{marker}")

    print()
    if any_olive:
        olive_pct = (pred_map == 1).sum() / total * 100
        print(f"  *** YES! This image contains olive groves ({olive_pct:.1f}% of area) ***")
    else:
        print(f"  *** NO olive groves detected in this image ***")
    print()


# ---------------------------------------------------------------
# Main
# ---------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="ZitounEye - Olive Grove Detector")
    parser.add_argument("--input", nargs="+", required=False,
                        help="Input GeoTIFF file(s)")
    parser.add_argument("--checkpoint", default=None,
                        help="Path to model checkpoint (auto-detected if not given)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nZitounEye Olive Detector")
    print(f"Device: {device}")
    print()

    # Find checkpoint
    ckpt_path = args.checkpoint or find_checkpoint()
    print(f"Loading model...")
    model = load_model(ckpt_path, device)

    # Load normalization stats
    if not os.path.exists(STATS_FILE):
        raise FileNotFoundError(f"Stats file not found: {STATS_FILE}. Run training first!")
    norm_stats = np.load(STATS_FILE)
    print(f"  Normalization stats loaded (56 channels)")

    # Collect input files
    if args.input:
        input_files = args.input
    else:
        # Default: all S2_olive_test*.tif in the project root
        input_files = sorted(glob.glob(
            r"c:\Users\AORUS\Documents\zitoun_eye\S2_olive_test*.tif"
        ))

    if not input_files:
        print("No input files found. Use --input your_image.tif")
        return

    print(f"\nProcessing {len(input_files)} image(s)...\n")

    for tif_path in input_files:
        fname = os.path.basename(tif_path)
        print(f"{'='*60}")
        print(f"Image: {fname}")
        print(f"{'='*60}")

        try:
            # Build 56-band input
            image_56band, meta = build_56band_input(tif_path)

            # Run inference
            print(f"  Running inference ({image_56band.shape[1]}x{image_56band.shape[2]} pixels)...")
            pred_map = predict_image(model, image_56band, norm_stats, device)

            # Summarize
            summarize_predictions(pred_map, tif_path)

            # Save outputs
            save_outputs(pred_map, meta, tif_path)

        except Exception as e:
            import traceback
            print(f"  ERROR: {e}")
            traceback.print_exc()

    print("Done!")


if __name__ == "__main__":
    main()
