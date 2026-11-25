import os
import rasterio
from rasterio.windows import Window
import numpy as np
import torch
import sys

# Import predict functions from the existing script
sys.path.append(r"c:\Users\AORUS\Documents\zitoun_eye\scripts")
from predict import load_model, predict_image, save_outputs, summarize_predictions, MODEL_BANDS

def compute_indices_for_season(data):
    """Takes 10-band data, computes 4 indices, returns 14 bands."""
    B2  = data[0]   # Blue
    B4  = data[2]   # Red
    B8  = data[6]   # NIR
    B11 = data[8]   # SWIR1
    
    eps = 1e-10
    NDVI = (B8 - B4) / (B8 + B4 + eps)
    EVI  = 2.5 * (B8 - B4) / (B8 + 6*B4 - 7.5*B2 + 1 + eps)
    NDWI = (data[1] - B8) / (data[1] + B8 + eps)
    SAVI = 1.5 * (B8 - B4) / (B8 + B4 + 0.5 + eps)
    
    return np.concatenate([
        data[:10],
        NDVI[np.newaxis], EVI[np.newaxis],
        NDWI[np.newaxis], SAVI[np.newaxis]
    ], axis=0)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Loading BEST model...")
    ckpt_path = r"c:\Users\AORUS\Documents\zitoun_eye\checkpoints\final_20260620_104109_best.pth"
    model = load_model(ckpt_path, device)
    
    stats_file = r"c:\Users\AORUS\Documents\zitoun_eye\data\patches\normalize_stats.npz"
    norm_stats = np.load(stats_file)
    
    base_dir = r"c:\Users\AORUS\Documents\zitoun_eye\spain_j_images"
    seasons = ["spring", "summer", "fall", "winter"]
    
    windows_to_test = [
        (0, Window(0, 0, 2500, 2500)),       # The empty blind spot (Almond farm)
        (1, Window(3000, 3000, 2500, 2500))  # The true olive grove
    ]
    
    for idx, window in windows_to_test:
        print(f"\n--- Testing Window {idx} with BEST model: {window} ---")
        
        season_data = []
        out_meta = None
        
        for s in seasons:
            fname = f"es_j_2023_{s}_2023-0000008960-0000008960.tif"
            fpath = os.path.join(base_dir, fname)
            with rasterio.open(fpath) as src:
                data = src.read(window=window).astype(np.float32)
                if out_meta is None:
                    out_meta = src.meta.copy()
                    out_meta.update({
                        "height": window.height,
                        "width": window.width,
                        "transform": src.window_transform(window)
                    })
            
            if data.max() > 10:
                data = data / 10000.0
            np.nan_to_num(data, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
            
            data_14 = compute_indices_for_season(data)
            season_data.append(data_14)
            
        stacked_56 = np.concatenate(season_data, axis=0)
        
        pred_map = predict_image(model, stacked_56, norm_stats, device)
        
        out_name = rf"c:\Users\AORUS\Documents\zitoun_eye\S2_true_4season_BEST_test_window{idx}.tif"
        summarize_predictions(pred_map, out_name)
        save_outputs(pred_map, out_meta, out_name)


if __name__ == "__main__":
    main()
