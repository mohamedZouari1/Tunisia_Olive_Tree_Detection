"""Chip city and desert TIFFs into background training chips with zero masks."""
import numpy as np, rasterio
from rasterio.windows import Window
from pathlib import Path

CHIP = 128
out_img = Path('chips/images')
out_msk = Path('chips/masks')
start = len(list(out_img.glob('*.tif')))
n = 0

for src_path, tag in [('tunis_city_chip.tif','city'), ('desert.tif','desert')]:
    with rasterio.open(src_path) as src:
        H, W, bands = src.height, src.width, src.count
        print(f'{src_path}: {W}x{H}, {bands} bands')
        # Compute indices if 10 bands
        full = src.read().astype(np.float32)
        full = np.nan_to_num(full, nan=0.0)
        if bands == 10:
            B3,B4,B8 = full[1],full[2],full[6]
            eps = 1e-8
            ndvi = (B8-B4)/(B8+B4+eps)
            ndwi = (B3-B8)/(B3+B8+eps)
            evi = 2.5*(B8-B4)/(B8+6*B4-7.5*B3+1+eps)
            savi = 1.5*(B8-B4)/(B8+B4+0.5+eps)
            full = np.concatenate([full,ndvi[None],ndwi[None],evi[None],savi[None]])
            bands = 14
        prof = src.profile.copy()
        for y in range(0, H-CHIP+1, CHIP//2):
            for x in range(0, W-CHIP+1, CHIP//2):
                chip = full[:, y:y+CHIP, x:x+CHIP]
                if chip.shape[1]!=CHIP or chip.shape[2]!=CHIP: continue
                if np.abs(chip).sum() < 1: continue
                name = f'chip_{tag}_{start+n:05d}.tif'
                p = prof.copy()
                p.update(width=CHIP,height=CHIP,count=bands,dtype='float32',
                         transform=rasterio.windows.transform(Window(x,y,CHIP,CHIP),src.transform))
                with rasterio.open(out_img/name,'w',**p) as d: d.write(chip)
                # Zero mask = all background
                pm = prof.copy()
                pm.update(width=CHIP,height=CHIP,count=1,dtype='uint8')
                with rasterio.open(out_msk/name,'w',**pm) as d:
                    d.write(np.zeros((1,CHIP,CHIP),dtype=np.uint8))
                n += 1
print(f'\nAdded {n} background chips (city+desert)')
