"""
check_patches.py - Scan ZitounEye patch directories for corrupted .npy files
and remove the (image, mask) pair if either file is bad.

Usage:
    python scripts/check_patches.py
    python scripts/check_patches.py --delete     (actually delete, default is dry-run)
"""
import os, sys, glob, argparse
import numpy as np

PATCHES_DIR = r"c:\Users\AORUS\Documents\zitoun_eye\data\patches_final"

def is_valid_npy(path):
    try:
        arr = np.load(path)
        # force-read the full array into memory to catch truncation
        _ = arr.shape, arr.sum() if arr.size < 5_000_000 else arr[..., :1].sum()
        return True
    except Exception as e:
        return False, str(e)

def scan(split_dir, delete=False):
    image_files = sorted(glob.glob(os.path.join(split_dir, "*_image.npy")))
    bad = []
    for img_path in image_files:
        msk_path = img_path.replace("_image.npy", "_mask.npy")

        ok_img = is_valid_npy(img_path)
        ok_msk = os.path.exists(msk_path) and is_valid_npy(msk_path)

        if ok_img is not True or ok_msk is not True:
            reason = ok_img if ok_img is not True else ok_msk
            bad.append((img_path, msk_path, reason))

    print(f"  {split_dir}: {len(image_files)} pairs checked, {len(bad)} corrupted")
    for img_path, msk_path, reason in bad:
        print(f"    BAD: {os.path.basename(img_path)}  ({reason[1] if isinstance(reason, tuple) else 'missing mask'})")
        if delete:
            for p in (img_path, msk_path):
                if os.path.exists(p):
                    os.remove(p)
    return len(bad)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--delete", action="store_true",
                         help="Actually delete corrupted pairs (default: dry-run report only)")
    parser.add_argument("--patches-dir", default=PATCHES_DIR)
    args = parser.parse_args()

    total_bad = 0
    for split in ("train", "val"):
        split_dir = os.path.join(args.patches_dir, split)
        if not os.path.isdir(split_dir):
            print(f"  {split_dir}: does not exist, skipping")
            continue
        total_bad += scan(split_dir, delete=args.delete)

    print()
    if total_bad == 0:
        print("No corrupted patches found.")
    elif args.delete:
        print(f"Deleted {total_bad} corrupted pairs.")
    else:
        print(f"Found {total_bad} corrupted pairs. Re-run with --delete to remove them.")

if __name__ == "__main__":
    main()
