import sys
from pathlib import Path
from PIL import Image, ImageFile
import imagehash
import numpy as np
import cv2
from tqdm import tqdm
import shutil
import pandas as pd
from collections import defaultdict, Counter

# ---------------- PARAMETERS ----------------
MIN_WIDTH = 50
MIN_HEIGHT = 50
VAR_LAPLACIAN_THRESH = 100.0
UNIFORM_STDDEV_THRESH = 6.0
BRIGHTNESS_LOW = 10
BRIGHTNESS_HIGH = 245
HASH_SIZE = 8
HAMMING_DUPLICATE_THRESH = 5
RESIZE_TO = (224, 224)
DRY_RUN = False              # must be False to save filtered images
ARCHIVE_COPY = True
CHECK_DUPLICATES_GLOBAL = False
VERBOSE = True
NORMALIZATION_STATS = True   # record mean/std per breed
ImageFile.LOAD_TRUNCATED_IMAGES = True
# --------------------------------------------

# ---------------- PATHS ----------------
RAW_DATASET = Path("/Users/boddisuryakathyakeya/Desktop/Infosys Project/Dog Breeds Image Dataset")
CLEANED_DATASET = Path("/Users/boddisuryakathyakeya/Desktop/Infosys Project/Dog Breeds Cleaned Dataset")
# ---------------------------------------

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def is_image_corrupt(path: Path) -> bool:
    try:
        with Image.open(path) as im:
            im.verify()
        return False
    except Exception:
        return True

def is_too_small(img: Image.Image):
    w, h = img.size
    return w < MIN_WIDTH or h < MIN_HEIGHT

def is_uniform_or_extreme(img: Image.Image):
    arr = np.array(img.convert('L'))
    mean, std = float(arr.mean()), float(arr.std())
    if std < UNIFORM_STDDEV_THRESH:
        return True, f'uniform_std_{std:.2f}'
    if mean < BRIGHTNESS_LOW:
        return True, f'too_dark_mean_{mean:.1f}'
    if mean > BRIGHTNESS_HIGH:
        return True, f'too_bright_mean_{mean:.1f}'
    return False, ''

def is_blurry_cv(img: Image.Image):
    arr = np.array(img.convert('L'))
    lap = cv2.Laplacian(arr, cv2.CV_64F)
    var = float(lap.var())
    return var < VAR_LAPLACIAN_THRESH, var

def compute_ahash(img: Image.Image):
    return imagehash.average_hash(img, hash_size=HASH_SIZE)

def unique_dest_path(dest_dir: Path, original_name: str) -> Path:
    base, ext = Path(original_name).stem, Path(original_name).suffix or ".jpg"
    candidate = dest_dir / (base + ext)
    i = 1
    while candidate.exists():
        candidate = dest_dir / f"{base}_{i}{ext}"
        i += 1
    return candidate

def archive_file(src: Path, cleaned_breed_dir: Path, subfolder: str, reason: str):
    target_dir = cleaned_breed_dir / subfolder / reason
    ensure_dir(target_dir)
    dest = unique_dest_path(target_dir, src.name)
    if DRY_RUN:
        return dest
    if ARCHIVE_COPY:
        shutil.copy2(src, dest)
    else:
        shutil.move(str(src), str(dest))
    return dest

def save_kept_image(pil_img: Image.Image, cleaned_breed_dir: Path, original_name: str):
    ensure_dir(cleaned_breed_dir)
    dest = unique_dest_path(cleaned_breed_dir, original_name)
    if DRY_RUN:
        return dest
    rgb = pil_img.convert('RGB').resize(RESIZE_TO, Image.LANCZOS)
    rgb.save(dest, format='JPEG', quality=95)
    return dest

def compute_mean_std(img: Image.Image):
    arr = np.array(img.convert('RGB')).astype(np.float32) / 255.0
    mean = arr.mean(axis=(0,1))
    std = arr.std(axis=(0,1))
    return mean, std

def process_dataset(raw_root: Path, cleaned_root: Path):
    log_rows = []
    ensure_dir(cleaned_root)
    breed_dirs = [d for d in raw_root.iterdir() if d.is_dir()]
    print(f'Found {len(breed_dirs)} breed folders to process.')

    global_seen_hashes = {}
    per_breed_counts = defaultdict(Counter)
    breed_stats = {}

    for breed in tqdm(sorted(breed_dirs), desc='Breeds'):
        cleaned_breed_dir = cleaned_root / breed.name
        ensure_dir(cleaned_breed_dir)
        files = [f for f in sorted(breed.iterdir()) if f.is_file()]
        if VERBOSE:
            print(f'Processing breed: {breed.name} ({len(files)} files)')

        seen_hashes = {}
        breed_mean = []
        breed_std = []

        for f in tqdm(files, desc=f' Files in {breed.name}', leave=False):
            try:
                if is_image_corrupt(f):
                    reason = 'corrupt'
                    dest = archive_file(f, cleaned_breed_dir, "_removed", reason)
                    log_rows.append((breed.name, str(f), 'removed', reason, str(dest)))
                    per_breed_counts[breed.name][reason] += 1
                    continue

                with Image.open(f) as pil_img:
                    pil_img.load()

                    if is_too_small(pil_img):
                        reason = 'too_small'
                        dest = archive_file(f, cleaned_breed_dir, "_removed", reason)
                        log_rows.append((breed.name, str(f), 'removed', reason, str(dest)))
                        per_breed_counts[breed.name][reason] += 1
                        continue

                    uni, uni_reason = is_uniform_or_extreme(pil_img)
                    if uni:
                        dest = archive_file(f, cleaned_breed_dir, "_removed", uni_reason)
                        log_rows.append((breed.name, str(f), 'removed', uni_reason, str(dest)))
                        per_breed_counts[breed.name][uni_reason] += 1
                        continue

                    blurry, blur_val = is_blurry_cv(pil_img)
                    if blurry:
                        reason = f'blurry_varlap_{blur_val:.2f}'
                        dest = archive_file(f, cleaned_breed_dir, "_removed", reason)
                        log_rows.append((breed.name, str(f), 'removed', reason, str(dest)))
                        per_breed_counts[breed.name]['blurry'] += 1
                        continue

                    ah = compute_ahash(pil_img)
                    duplicate_found = False

                    if CHECK_DUPLICATES_GLOBAL:
                        for seen_h, (seen_path, seen_breed) in global_seen_hashes.items():
                            if (ah - seen_h) <= HAMMING_DUPLICATE_THRESH:
                                reason = f'duplicate_of_{Path(seen_path).name}_dist{ah-seen_h}_breed_{seen_breed}'
                                dest = archive_file(f, cleaned_breed_dir, "_duplicates", reason)
                                log_rows.append((breed.name, str(f), 'duplicate', reason, str(dest)))
                                per_breed_counts[breed.name]['duplicates'] += 1
                                duplicate_found = True
                                break
                        if duplicate_found:
                            continue
                    else:
                        for seen_h, seen_path in seen_hashes.items():
                            if (ah - seen_h) <= HAMMING_DUPLICATE_THRESH:
                                reason = f'duplicate_of_{Path(seen_path).name}_dist{ah-seen_h}'
                                dest = archive_file(f, cleaned_breed_dir, "_duplicates", reason)
                                log_rows.append((breed.name, str(f), 'duplicate', reason, str(dest)))
                                per_breed_counts[breed.name]['duplicates'] += 1
                                duplicate_found = True
                                break
                        if duplicate_found:
                            continue

                    # Save cleaned image
                    saved = save_kept_image(pil_img, cleaned_breed_dir, f.name)
                    log_rows.append((breed.name, str(f), 'kept', str(saved), ''))
                    per_breed_counts[breed.name]['kept'] += 1

                    # Compute mean/std for training normalization
                    if NORMALIZATION_STATS:
                        m, s = compute_mean_std(pil_img)
                        breed_mean.append(m)
                        breed_std.append(s)

                    if CHECK_DUPLICATES_GLOBAL:
                        global_seen_hashes[ah] = (str(saved), breed.name)
                    else:
                        seen_hashes[ah] = str(saved)

            except Exception as e:
                reason = f'error_{type(e).__name__}'
                dest = archive_file(f, cleaned_breed_dir, "_removed", reason)
                log_rows.append((breed.name, str(f), 'removed', f'error:{repr(e)}', str(dest)))
                per_breed_counts[breed.name]['errors'] += 1
                continue

        # Save breed-level mean/std
        if breed_mean:
            breed_mean = np.mean(breed_mean, axis=0)
            breed_std = np.mean(breed_std, axis=0)
            breed_stats[breed.name] = {'mean': breed_mean.tolist(), 'std': breed_std.tolist()}

    # Write log CSV
    df = pd.DataFrame(log_rows, columns=['breed','original_path','action','note','dest_path'])
    csv_path = cleaned_root / 'cleaning_log.csv'
    df.to_csv(csv_path, index=False)

    # Save normalization stats
    if NORMALIZATION_STATS:
        stats_df = pd.DataFrame.from_dict(breed_stats, orient='index')
        stats_df.to_csv(cleaned_root / 'breed_normalization_stats.csv')

    # Summary
    total_counts = Counter(df['action'])
    print("\n--- Cleaning Summary ---")
    print(f"Total files processed: {len(df)}")
    for k,v in total_counts.items():
        print(f"  {k}: {v}")

    removals_by_breed = {b: sum(cnts[r] for r in cnts if r != 'kept') for b, cnts in per_breed_counts.items()}
    sorted_removals = sorted(removals_by_breed.items(), key=lambda x: x[1], reverse=True)
    print("\nTop 10 breeds by removed/duplicated files:")
    for breed_name, rem in sorted_removals[:10]:
        kept = per_breed_counts[breed_name].get('kept', 0)
        print(f"  {breed_name}: removed_or_dup={rem}, kept={kept}")

    return df, breed_stats

def main():
    raw = RAW_DATASET
    cleaned = CLEANED_DATASET
    if not raw.exists():
        print("Raw dataset path doesn't exist:", raw)
        sys.exit(1)
    print("Raw dataset root:", raw)
    print("Cleaned dataset root (output):", cleaned)
    print(f"DRY_RUN={DRY_RUN}, CHECK_DUPLICATES_GLOBAL={CHECK_DUPLICATES_GLOBAL}, ARCHIVE_COPY={ARCHIVE_COPY}")
    df, stats = process_dataset(raw, cleaned)
    print("Done.")

if __name__ == '__main__':
    main()
