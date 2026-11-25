"""
ZitounEye — Step 6: Inference on New Areas
============================================
Takes any Sentinel-2 GeoTIFF and produces a georeferenced
olive grove classification map.

Features:
- Sliding window with overlap + Gaussian blending (no tile artifacts)
- Test-Time Augmentation (TTA) — flip + rotate for ~2-3% IoU boost
- Ensemble of U-Net + DeepLabV3+ (if both checkpoints provided)
- Outputs GeoTIFF with class probabilities or hard predictions

Usage:
    python scripts/06_inference.py --checkpoint checkpoints/best.pth --input sentinel2.tif
    python scripts/06_inference.py --checkpoint best.pth --input image.tif --tta
    python scripts/06_inference.py --ckpt1 unet.pth --ckpt2 deeplab.pth --input image.tif --ensemble
"""

import os
import sys
import argparse
import warnings
import numpy as np
import torch
import torch.nn.functional as F
import rasterio
from rasterio.transform import from_bounds
from pathlib import Path
from tqdm import tqdm

import segmentation_models_pytorch as smp

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.utils import load_config, colorize_mask

warnings.filterwarnings("ignore")


def create_gaussian_kernel(patch_size: int, sigma_fraction: float = 0.25) -> np.ndarray:
    """
    Create 2D Gaussian weight kernel for blending overlapping patches.
    
    Center pixels get higher weight, edges get lower weight.
    This prevents visible seams between adjacent patches.
    """
    sigma = patch_size * sigma_fraction
    x = np.arange(patch_size) - patch_size / 2
    kernel_1d = np.exp(-x**2 / (2 * sigma**2))
    kernel_2d = np.outer(kernel_1d, kernel_1d)
    kernel_2d = kernel_2d / kernel_2d.max()
    return kernel_2d.astype(np.float32)


def load_model_from_checkpoint(checkpoint_path: str, device: torch.device):
    """Load a model from checkpoint, auto-detecting architecture."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint.get("config", {})
    
    # Try each tier
    for tier in [1, 2]:
        tier_key = f"tier{tier}"
        if tier_key not in config.get("model", {}):
            continue
        
        model_cfg = config["model"][tier_key]
        arch = model_cfg["architecture"]
        
        try:
            if arch == "Unet":
                model = smp.Unet(
                    encoder_name=model_cfg["encoder"],
                    encoder_weights=None,
                    in_channels=model_cfg["in_channels"],
                    classes=model_cfg["classes"],
                )
            elif arch == "DeepLabV3Plus":
                model = smp.DeepLabV3Plus(
                    encoder_name=model_cfg["encoder"],
                    encoder_weights=None,
                    in_channels=model_cfg["in_channels"],
                    classes=model_cfg["classes"],
                )
            else:
                continue
            
            model.load_state_dict(checkpoint["model_state_dict"])
            model = model.to(device)
            model.eval()
            
            print(f"  Loaded: {arch} ({model_cfg['encoder']})")
            print(f"  Channels: {model_cfg['in_channels']}, Classes: {model_cfg['classes']}")
            return model, config
        except Exception as e:
            continue
    
    raise RuntimeError(f"Could not load model from {checkpoint_path}")


def predict_with_tta(model, image_tensor, num_classes=7):
    """
    Test-Time Augmentation: average predictions across geometric transforms.
    
    Transforms used:
    - Original
    - Horizontal flip
    - Vertical flip
    - 180° rotation (= horizontal + vertical flip)
    
    Typically gives +2-3% IoU improvement for free.
    """
    predictions = []
    
    # Original
    with torch.no_grad():
        pred = F.softmax(model(image_tensor), dim=1)
        predictions.append(pred)
    
    # Horizontal flip
    with torch.no_grad():
        flipped = torch.flip(image_tensor, dims=[3])
        pred = F.softmax(model(flipped), dim=1)
        pred = torch.flip(pred, dims=[3])  # flip back
        predictions.append(pred)
    
    # Vertical flip
    with torch.no_grad():
        flipped = torch.flip(image_tensor, dims=[2])
        pred = F.softmax(model(flipped), dim=1)
        pred = torch.flip(pred, dims=[2])
        predictions.append(pred)
    
    # 180° rotation
    with torch.no_grad():
        rotated = torch.flip(image_tensor, dims=[2, 3])
        pred = F.softmax(model(rotated), dim=1)
        pred = torch.flip(pred, dims=[2, 3])
        predictions.append(pred)
    
    # Average
    return torch.stack(predictions).mean(dim=0)


def run_inference(
    model,
    image: np.ndarray,
    patch_size: int = 256,
    overlap: int = 128,
    batch_size: int = 32,
    num_classes: int = 7,
    device: torch.device = torch.device("cpu"),
    use_tta: bool = False,
    normalize_stats: dict = None,
) -> np.ndarray:
    """
    Run sliding-window inference with Gaussian blending.
    
    Args:
        model: trained segmentation model
        image: (C, H, W) input image
        patch_size: size of inference patches
        overlap: overlap between adjacent patches
        batch_size: number of patches per forward pass
        num_classes: number of output classes
        device: torch device
        use_tta: whether to use test-time augmentation
        normalize_stats: dict with "mean" and "std" arrays
    
    Returns:
        (H, W) array of predicted class IDs
    """
    C, H, W = image.shape
    stride = patch_size - overlap
    
    # Gaussian blending kernel
    kernel = create_gaussian_kernel(patch_size)
    
    # Accumulate weighted predictions
    pred_sum = np.zeros((num_classes, H, W), dtype=np.float64)
    weight_sum = np.zeros((H, W), dtype=np.float64)
    
    # Normalize
    if normalize_stats is not None:
        mean = normalize_stats["mean"][:C, None, None]
        std = normalize_stats["std"][:C, None, None]
        image = (image - mean) / (std + 1e-8)
    
    # Collect patches
    patches = []
    positions = []
    
    for y in range(0, max(H - patch_size + 1, 1), stride):
        for x in range(0, max(W - patch_size + 1, 1), stride):
            y_end = min(y + patch_size, H)
            x_end = min(x + patch_size, W)
            y_start = y_end - patch_size
            x_start = x_end - patch_size
            
            patch = image[:, y_start:y_end, x_start:x_end]
            patches.append(patch)
            positions.append((y_start, x_start))
    
    # Process in batches
    model.eval()
    
    for i in tqdm(range(0, len(patches), batch_size), desc="  Inference"):
        batch_patches = patches[i:i+batch_size]
        batch_positions = positions[i:i+batch_size]
        
        batch_tensor = torch.from_numpy(np.stack(batch_patches)).float().to(device)
        
        with torch.no_grad():
            if use_tta:
                batch_pred = predict_with_tta(model, batch_tensor, num_classes)
            else:
                batch_pred = F.softmax(model(batch_tensor), dim=1)
        
        batch_pred = batch_pred.cpu().numpy()
        
        for j, (y, x) in enumerate(batch_positions):
            pred_sum[:, y:y+patch_size, x:x+patch_size] += batch_pred[j] * kernel[None, :, :]
            weight_sum[y:y+patch_size, x:x+patch_size] += kernel
    
    # Normalize by weights
    weight_sum = np.maximum(weight_sum, 1e-8)
    prob_map = pred_sum / weight_sum[None, :, :]
    
    # Hard predictions
    class_map = prob_map.argmax(axis=0).astype(np.uint8)
    
    return class_map, prob_map


def main():
    parser = argparse.ArgumentParser(description="Run inference on Sentinel-2 imagery")
    parser.add_argument("--checkpoint", "--ckpt1", required=True, help="Path to model checkpoint")
    parser.add_argument("--ckpt2", default=None, help="Path to second model for ensemble")
    parser.add_argument("--input", required=True, help="Input Sentinel-2 GeoTIFF")
    parser.add_argument("--output", default=None, help="Output classification GeoTIFF")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--tta", action="store_true", help="Use Test-Time Augmentation")
    parser.add_argument("--ensemble", action="store_true", help="Ensemble multiple models")
    parser.add_argument("--patch-size", type=int, default=256)
    parser.add_argument("--overlap", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--save-probabilities", action="store_true", 
                        help="Also save per-class probability map")
    parser.add_argument("--save-rgb", action="store_true",
                        help="Also save colorized RGB classification map")
    args = parser.parse_args()
    
    config = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("=" * 60)
    print("ZitounEye — Inference")
    print("=" * 60)
    print(f"  Device: {device}")
    print(f"  Input: {args.input}")
    print(f"  TTA: {args.tta}")
    
    # Load model(s)
    models = []
    model1, model_config = load_model_from_checkpoint(args.checkpoint, device)
    models.append(model1)
    
    if args.ckpt2 and args.ensemble:
        model2, _ = load_model_from_checkpoint(args.ckpt2, device)
        models.append(model2)
        print(f"  Ensemble: {len(models)} models")
    
    # Load normalization stats
    patches_dir = config["paths"]["patches_dir"]
    stats_path = os.path.join(patches_dir, "normalize_stats.npz")
    normalize_stats = None
    if os.path.exists(stats_path):
        stats = np.load(stats_path)
        normalize_stats = {"mean": stats["mean"], "std": stats["std"]}
    
    # Load input image
    print(f"\n  Loading image...")
    with rasterio.open(args.input) as src:
        image = src.read().astype(np.float32)  # (C, H, W)
        profile = src.profile.copy()
        transform = src.transform
        crs = src.crs
    
    C, H, W = image.shape
    print(f"  Image shape: {C} bands × {H} rows × {W} cols")
    print(f"  CRS: {crs}")
    
    # Replace NaN/inf
    image = np.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Run inference for each model
    num_classes = config["classes"]["num_classes"]
    
    all_prob_maps = []
    
    for i, model in enumerate(models):
        print(f"\n  Running model {i+1}/{len(models)}...")
        class_map, prob_map = run_inference(
            model=model,
            image=image,
            patch_size=args.patch_size,
            overlap=args.overlap,
            batch_size=args.batch_size,
            num_classes=num_classes,
            device=device,
            use_tta=args.tta,
            normalize_stats=normalize_stats,
        )
        all_prob_maps.append(prob_map)
    
    # Ensemble averaging
    if len(all_prob_maps) > 1:
        print(f"\n  Ensembling {len(all_prob_maps)} models...")
        ensemble_prob = np.mean(all_prob_maps, axis=0)
        final_class_map = ensemble_prob.argmax(axis=0).astype(np.uint8)
        final_prob_map = ensemble_prob
    else:
        final_class_map = class_map
        final_prob_map = prob_map
    
    # Determine output path
    if args.output:
        output_path = args.output
    else:
        input_stem = Path(args.input).stem
        output_dir = config["paths"]["predictions_dir"]
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"{input_stem}_prediction.tif")
    
    # Save classification map as GeoTIFF
    print(f"\n  Saving classification map...")
    
    out_profile = profile.copy()
    out_profile.update({
        "count": 1,
        "dtype": "uint8",
        "compress": "lzw",
    })
    
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    
    with rasterio.open(output_path, "w", **out_profile) as dst:
        dst.write(final_class_map, 1)
        
        # Add class name descriptions
        dst.update_tags(
            class_0="background",
            class_1="olive_grove",
            class_2="vineyard",
            class_3="other_orchard",
            class_4="arable_crop",
            class_5="building",
            class_6="road",
            model=args.checkpoint,
            tta=str(args.tta),
            ensemble=str(len(models) > 1),
        )
    
    print(f"  ✅ Classification map: {output_path}")
    
    # Save probability map if requested
    if args.save_probabilities:
        prob_path = output_path.replace(".tif", "_probabilities.tif")
        prob_profile = profile.copy()
        prob_profile.update({
            "count": num_classes,
            "dtype": "float32",
            "compress": "lzw",
        })
        
        with rasterio.open(prob_path, "w", **prob_profile) as dst:
            for c in range(num_classes):
                dst.write(final_prob_map[c].astype(np.float32), c + 1)
        
        print(f"  ✅ Probability map: {prob_path}")
    
    # Save colorized RGB map
    if args.save_rgb:
        rgb_path = output_path.replace(".tif", "_rgb.tif")
        rgb_map = colorize_mask(final_class_map, config)  # (H, W, 3)
        
        rgb_profile = profile.copy()
        rgb_profile.update({
            "count": 3,
            "dtype": "uint8",
            "compress": "lzw",
        })
        
        with rasterio.open(rgb_path, "w", **rgb_profile) as dst:
            for c in range(3):
                dst.write(rgb_map[:, :, c], c + 1)
        
        print(f"  ✅ RGB classification map: {rgb_path}")
    
    # Print statistics
    print(f"\n{'='*60}")
    print("PREDICTION STATISTICS")
    print(f"{'='*60}")
    
    class_names = config["classes"]["names"]
    unique, counts = np.unique(final_class_map, return_counts=True)
    total_pixels = H * W
    
    for cls_id, count in zip(unique, counts):
        name = class_names[cls_id] if cls_id < len(class_names) else f"class_{cls_id}"
        pct = count / total_pixels * 100
        area_km2 = count * (10 * 10) / 1e6  # 10m resolution → km²
        marker = " 🫒" if cls_id == 1 else ""
        print(f"  {name:<20}: {count:>10,} pixels ({pct:>5.1f}%) = {area_km2:>8.2f} km²{marker}")
    
    olive_pixels = (final_class_map == 1).sum()
    olive_area = olive_pixels * 100 / 1e6  # 100 m² per pixel → km²
    print(f"\n  🫒 Total olive area detected: {olive_area:.2f} km² ({olive_area * 100:.0f} hectares)")
    
    print(f"\n{'='*60}")
    print("✅ Inference complete!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
