"""
Process GEE auto-generated training data (v2) into chips.
Expects files from Google Drive → olive_training_v2/ folder:
  S2_sfax_north.tif, labels_sfax_north.tif, etc.

Usage:
  1. Download all files from Google Drive to this folder
  2. python chip_training_v2.py
  3. python train_unet_olive.py --mode stats --data_dir ./chips_v2
  4. python train_unet_olive.py --mode train --data_dir ./chips_v2
"""
import numpy as np
import rasterio
from rasterio.windows import Window
from pathlib import Path
import logging, glob

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
log = logging.getLogger(__name__)

CHIP_SIZE = 128
STRIDE = 64  # 50% overlap
MIN_OLIVE_RATIO = 0.01  # 1% olive to keep chip
BG_KEEP_RATIO = 0.15    # Keep 15% of all-background chips

def compute_indices(bands_10):
    """Add NDVI, NDWI, EVI, SAVI to 10-band input → 14 bands."""
    B3, B4, B8 = bands_10[1], bands_10[2], bands_10[6]
    eps = 1e-8
    ndvi = (B8 - B4) / (B8 + B4 + eps)
    ndwi = (B3 - B8) / (B3 + B8 + eps)
    evi  = 2.5 * (B8 - B4) / (B8 + 6*B4 - 7.5*B3 + 1 + eps)
    savi = 1.5 * (B8 - B4) / (B8 + B4 + 0.5 + eps)
    return np.concatenate([bands_10, ndvi[None], ndwi[None], evi[None], savi[None]])

def chip_pair(img_path, mask_path, out_dir, tag):
    """Chip an image+mask pair into 128x128 training tiles."""
    img_out = Path(out_dir) / 'images'
    mask_out = Path(out_dir) / 'masks'
    img_out.mkdir(parents=True, exist_ok=True)
    mask_out.mkdir(parents=True, exist_ok=True)

    start_idx = len(list(img_out.glob('*.tif')))
    saved = 0
    saved_bg = 0
    skipped = 0

    with rasterio.open(img_path) as img_src, rasterio.open(mask_path) as msk_src:
        H, W = img_src.height, img_src.width
        n_bands = img_src.count
        log.info(f'{tag}: {img_path} → {W}x{H}, {n_bands} bands')

        for y in range(0, H - CHIP_SIZE + 1, STRIDE):
            for x in range(0, W - CHIP_SIZE + 1, STRIDE):
                win = Window(x, y, CHIP_SIZE, CHIP_SIZE)
                mask = msk_src.read(1, window=win)
                img = img_src.read(window=win).astype(np.float32)
                img = np.nan_to_num(img, nan=0.0)

                # Skip nodata chips
                if np.abs(img).sum() < 1:
                    continue

                olive_ratio = (mask > 0).mean()

                # Decide whether to keep
                if olive_ratio >= MIN_OLIVE_RATIO:
                    pass  # Always keep olive chips
                elif np.random.random() < BG_KEEP_RATIO:
                    saved_bg += 1  # Keep some background
                else:
                    skipped += 1
                    continue

                # Add spectral indices if only 10 bands
                if n_bands == 10:
                    img = compute_indices(img)
                    n_bands_out = 14
                else:
                    n_bands_out = n_bands

                name = f'chip_{tag}_{start_idx + saved:05d}.tif'

                prof = img_src.profile.copy()
                prof.update(width=CHIP_SIZE, height=CHIP_SIZE, count=n_bands_out,
                            dtype='float32',
                            transform=rasterio.windows.transform(win, img_src.transform))
                with rasterio.open(img_out / name, 'w', **prof) as dst:
                    dst.write(img)

                prof_m = msk_src.profile.copy()
                prof_m.update(width=CHIP_SIZE, height=CHIP_SIZE, count=1, dtype='uint8',
                              transform=rasterio.windows.transform(win, msk_src.transform))
                with rasterio.open(mask_out / name, 'w', **prof_m) as dst:
                    dst.write(mask[np.newaxis])

                saved += 1

    log.info(f'{tag}: {saved} chips ({saved - saved_bg} olive, {saved_bg} bg), {skipped} skipped')
    return saved

if __name__ == '__main__':
    out_dir = './chips_v2'
    total = 0

    # Find all S2_*.tif files and match with labels_*.tif
    for s2_file in sorted(glob.glob('S2_*.tif')):
        region = s2_file.replace('S2_', '').replace('.tif', '')
        label_file = f'labels_{region}.tif'

        if not Path(label_file).exists():
            log.warning(f'No label file for {s2_file}, skipping')
            continue

        total += chip_pair(s2_file, label_file, out_dir, region)

    print(f'\n{"="*60}')
    print(f'Total chips: {total}')
    print(f'Output dir:  {out_dir}/')
    print(f'{"="*60}')
    print(f'\nNext steps:')
    print(f'  python train_unet_olive.py --mode stats --data_dir ./chips_v2')
    print(f'  python train_unet_olive.py --mode train --data_dir ./chips_v2')
