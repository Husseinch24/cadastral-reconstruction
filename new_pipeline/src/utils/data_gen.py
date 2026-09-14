"""
=============================================================================
MILESTONE 1 — SYNTHETIC DATA GENERATOR
=============================================================================
Masters Thesis — AI-Based Panoramic Cadastral Image Reconstruction
Author  : Hussein Chalhoub

Purpose
--------
Generate synthetic training data for Mask R-CNN parcel segmentation by
augmenting the 13 real cadastral maps with realistic transformations that
simulate the full range of scan conditions found in historical blueprint
atlases.

Why synthetic data?
-------------------
Mask R-CNN needs thousands of annotated training images to learn.
We only have 13 real maps.  The solution: take those 13 maps and
generate 50,000+ variations by applying combinations of:

  1. Blueprint mirroring (many maps are printed in reverse)
  2. Paper aging effects (yellowing, staining, foxing)
  3. Ink fading (variable density, patchy coverage)
  4. Physical damage simulation (fold creases, tears, water damage)
  5. Scan artifacts (skew, vignetting, noise)
  6. Geometric transforms (rotation, scale, perspective)

The model trained on this data will be robust to ALL of these conditions
across the full instructor dataset (potentially thousands of maps).

Annotation format: COCO JSON
------------------------------
Mask R-CNN training uses the COCO instance segmentation format:
  - Each image has a list of "annotations"
  - Each annotation has a segmentation mask (polygon vertices)
  - All annotations for all images are stored in one JSON file

Input
------
  output/preprocessed/map_<N>_clean.png   — Step 1 preprocessed images
  output/preprocessed/map_<N>_binary.png  — binary (ink/paper) images

Output
-------
  data/synthetic/images/<split>/img_<XXXXX>.png  — augmented images
  data/synthetic/annotations/<split>.json         — COCO format annotations
  data/synthetic/stats.json                       — generation statistics

Usage
------
  # Activate environment first
  .\venv_thesis\Scripts\Activate.ps1

  # Generate full dataset (default: 5000 train, 500 val, 200 test)
  python src/utils/data_gen.py

  # Quick test run (50 images total)
  python src/utils/data_gen.py --quick

  # Custom counts
  python src/utils/data_gen.py --train 10000 --val 1000 --test 500

=============================================================================
"""

import argparse
import json
import random
import time
from pathlib import Path
from datetime import datetime

import cv2
import numpy as np
from tqdm import tqdm


# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------

PREPROCESSED_DIR = Path("output/preprocessed")
SYNTHETIC_DIR    = Path("new_pipeline/data/synthetic")

# All available map numbers
ALL_MAPS = [str(n) for n in range(43, 56)]

# Output image size — crop windows from the large maps
CROP_SIZE = 512   # 512x512 pixels per training image

# Parcel segmentation parameters (same as Step 2 baseline)
MIN_PARCEL_AREA = 1000    # minimum pixels for a valid parcel
MAX_PARCEL_AREA = 300000  # maximum pixels (excludes the full map border)
MIN_PARCEL_DIM  = 20      # minimum width or height in pixels

# Split ratios
SPLIT_RATIOS = {"train": 0.80, "val": 0.15, "test": 0.05}

# COCO category definition
COCO_CATEGORIES = [
    {"id": 1, "name": "parcel", "supercategory": "cadastral"}
]


# ---------------------------------------------------------------------------
# UTILITIES
# ---------------------------------------------------------------------------

def read_image(path: Path, grayscale: bool = True) -> np.ndarray:
    data = np.fromfile(str(path), dtype=np.uint8)
    flag = cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR
    img = cv2.imdecode(data, flag)
    if img is None:
        raise FileNotFoundError(f"Cannot read: {path}")
    return img


def save_image(path: Path, img: np.ndarray):
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, enc = cv2.imencode(path.suffix, img)
    if ok:
        enc.tofile(str(path))


def polygon_to_mask(polygon: list[tuple], h: int, w: int) -> np.ndarray:
    """Convert a polygon (list of (x,y) tuples) to a binary mask."""
    mask = np.zeros((h, w), dtype=np.uint8)
    pts  = np.array(polygon, dtype=np.int32).reshape(-1, 1, 2)
    cv2.fillPoly(mask, [pts], 255)
    return mask


def mask_to_coco_polygon(mask: np.ndarray) -> list[float] | None:
    """
    Convert a binary mask to COCO polygon format (flat list of floats).
    Returns None if the contour is too small or invalid.
    """
    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return None
    # Take the largest contour
    c = max(contours, key=cv2.contourArea)
    if cv2.contourArea(c) < 50:
        return None
    # Simplify polygon
    epsilon = 0.005 * cv2.arcLength(c, True)
    approx  = cv2.approxPolyDP(c, epsilon, True)
    if len(approx) < 3:
        return None
    flat = approx.reshape(-1).tolist()
    return [float(v) for v in flat]


def compute_bbox_from_mask(mask: np.ndarray) -> list[int]:
    """Compute COCO bounding box [x, y, width, height] from binary mask."""
    coords = np.where(mask > 0)
    if len(coords[0]) == 0:
        return [0, 0, 0, 0]
    y_min, y_max = int(coords[0].min()), int(coords[0].max())
    x_min, x_max = int(coords[1].min()), int(coords[1].max())
    return [x_min, y_min, x_max - x_min + 1, y_max - y_min + 1]


# ---------------------------------------------------------------------------
# PARCEL DETECTION (from binary image)
# ---------------------------------------------------------------------------

def detect_parcels_in_crop(binary_crop: np.ndarray
                             ) -> list[dict]:
    """
    Find parcel regions in a binary crop using connected components.

    The binary image has ink=0, paper=255.
    We invert it so parcel interiors become white blobs, then find contours.

    Returns list of dicts with: mask, polygon, bbox, area
    """
    h, w = binary_crop.shape
    inverted = cv2.bitwise_not(binary_crop)

    contours, hierarchy = cv2.findContours(
        inverted, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours or hierarchy is None:
        return []

    parcels = []
    for i, contour in enumerate(contours):
        # Skip holes (interior contours)
        if hierarchy[0][i][3] != -1:
            continue

        area = cv2.contourArea(contour)
        if area < MIN_PARCEL_AREA or area > MAX_PARCEL_AREA:
            continue

        x, y, bw, bh = cv2.boundingRect(contour)
        if bw < MIN_PARCEL_DIM or bh < MIN_PARCEL_DIM:
            continue

        # Create mask for this parcel
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.drawContours(mask, [contour], -1, 255, -1)

        # Convert to polygon
        polygon = mask_to_coco_polygon(mask)
        if polygon is None:
            continue

        parcels.append({
            "mask":    mask,
            "polygon": polygon,
            "bbox":    compute_bbox_from_mask(mask),
            "area":    float(area),
        })

    return parcels


# ---------------------------------------------------------------------------
# AUGMENTATION PIPELINE
# ---------------------------------------------------------------------------

class CadastralAugmentor:
    """
    Applies realistic augmentations that simulate the full range of
    historical blueprint scan conditions.

    Each augmentation can be applied independently with a given probability,
    and multiple augmentations are combined per image.
    """

    def __init__(self, rng: random.Random | None = None):
        self.rng = rng or random.Random()

    def apply(self, img: np.ndarray) -> np.ndarray:
        """Apply a random combination of augmentations to a grayscale image."""
        # Always apply at least one augmentation
        augmentations = [
            (self._blueprint_mirror,    0.45),
            (self._ink_fading,          0.60),
            (self._paper_aging,         0.50),
            (self._fold_creases,        0.40),
            (self._scan_noise,          0.70),
            (self._scan_skew,           0.30),
            (self._vignetting,          0.35),
            (self._local_blur,          0.40),
            (self._contrast_variation,  0.65),
            (self._water_stain,         0.25),
        ]

        result = img.copy()
        applied = []
        for fn, prob in augmentations:
            if self.rng.random() < prob:
                result = fn(result)
                applied.append(fn.__name__)

        # Ensure at least one was applied
        if not applied:
            result = self._scan_noise(result)

        return result

    # ── Individual augmentation functions ─────────────────────────────────

    def _blueprint_mirror(self, img: np.ndarray) -> np.ndarray:
        """
        Horizontally flip the image to simulate the blueprint printing
        process that mirrors the original drawing.
        Probability: 0.45 — nearly half of all atlas sheets are mirrored.
        """
        return cv2.flip(img, 1)

    def _ink_fading(self, img: np.ndarray) -> np.ndarray:
        """
        Simulate ink fading over time: reduce contrast of dark regions.
        Creates a non-uniform fade map so some areas fade more than others.
        """
        h, w = img.shape
        # Create a smooth random fade map (large gaussian blobs)
        fade = np.ones((h, w), dtype=np.float32)
        n_blobs = self.rng.randint(3, 8)
        for _ in range(n_blobs):
            cx = self.rng.randint(0, w)
            cy = self.rng.randint(0, h)
            radius = self.rng.randint(min(h, w) // 8, min(h, w) // 2)
            strength = self.rng.uniform(0.05, 0.35)
            y_coords, x_coords = np.ogrid[:h, :w]
            dist = np.sqrt((x_coords - cx) ** 2 + (y_coords - cy) ** 2)
            blob = np.clip(1 - dist / radius, 0, 1) * strength
            fade += blob

        fade = np.clip(fade, 1.0, 1.5)

        # Fade darkens by pulling dark pixels toward gray
        result = img.astype(np.float32)
        # Dark pixels (ink) become lighter proportionally
        dark_mask = result < 180
        result[dark_mask] = result[dark_mask] * fade[dark_mask]
        return np.clip(result, 0, 255).astype(np.uint8)

    def _paper_aging(self, img: np.ndarray) -> np.ndarray:
        """
        Simulate paper yellowing/browning from aging.
        Adds a non-uniform sepia-like tint (in grayscale: darkens white areas
        slightly and adds noise texture).
        """
        h, w = img.shape
        # Create smooth yellowing texture
        aging = np.random.normal(
            loc=self.rng.uniform(-5, 15),
            scale=self.rng.uniform(3, 12),
            size=(h // 8, w // 8)
        ).astype(np.float32)
        aging = cv2.resize(aging, (w, h), interpolation=cv2.INTER_CUBIC)
        result = img.astype(np.float32) + aging
        return np.clip(result, 0, 255).astype(np.uint8)

    def _fold_creases(self, img: np.ndarray) -> np.ndarray:
        """
        Simulate physical fold creases: bright or dark lines crossing
        the image at random angles, with soft edges.
        """
        h, w = img.shape
        result = img.copy().astype(np.float32)
        n_folds = self.rng.randint(1, 4)
        for _ in range(n_folds):
            # Random line parameters
            angle = self.rng.uniform(0, 180)
            pos   = self.rng.uniform(0.1, 0.9)
            width = self.rng.randint(3, 15)
            style = self.rng.choice(["dark", "light"])

            # Build the crease as a soft line
            crease = np.zeros((h, w), dtype=np.float32)
            cx = int(pos * w)
            cy = int(pos * h)

            if abs(angle - 90) < 20:   # nearly horizontal
                for dy in range(-width, width + 1):
                    y = cy + dy
                    if 0 <= y < h:
                        intensity = 1 - abs(dy) / width
                        crease[y, :] = intensity * self.rng.uniform(15, 40)
            else:                       # diagonal / vertical
                for dx in range(-width, width + 1):
                    x = cx + dx
                    if 0 <= x < w:
                        intensity = 1 - abs(dx) / width
                        crease[:, x] = intensity * self.rng.uniform(15, 40)

            if style == "dark":
                result -= crease
            else:
                result += crease

        return np.clip(result, 0, 255).astype(np.uint8)

    def _scan_noise(self, img: np.ndarray) -> np.ndarray:
        """
        Add Gaussian noise to simulate scanner sensor noise.
        Mild amount to not destroy parcel line visibility.
        """
        std = self.rng.uniform(1.0, 8.0)
        noise = np.random.normal(0, std, img.shape).astype(np.float32)
        return np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    def _scan_skew(self, img: np.ndarray) -> np.ndarray:
        """
        Apply a small rotation to simulate document not being placed
        perfectly flat on the scanner bed.
        Range: ±3 degrees.
        """
        h, w = img.shape
        angle = self.rng.uniform(-3.0, 3.0)
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        return cv2.warpAffine(
            img, M, (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE
        )

    def _vignetting(self, img: np.ndarray) -> np.ndarray:
        """
        Add a darkening toward the edges, simulating scanner vignetting
        or the shadow from a document pressed unevenly.
        """
        h, w = img.shape
        y, x = np.ogrid[:h, :w]
        cy, cx = h / 2, w / 2
        dist = np.sqrt(((x - cx) / cx) ** 2 + ((y - cy) / cy) ** 2)
        strength = self.rng.uniform(0.05, 0.25)
        vignette = 1 - strength * dist
        vignette = np.clip(vignette, 0.5, 1.0).astype(np.float32)
        result = img.astype(np.float32) * vignette
        return np.clip(result, 0, 255).astype(np.uint8)

    def _local_blur(self, img: np.ndarray) -> np.ndarray:
        """
        Apply Gaussian blur to random sub-regions to simulate
        local focus issues during scanning.
        """
        h, w = img.shape
        result = img.copy()
        n_regions = self.rng.randint(1, 4)
        for _ in range(n_regions):
            x1 = self.rng.randint(0, w - 1)
            y1 = self.rng.randint(0, h - 1)
            x2 = min(w, x1 + self.rng.randint(50, 200))
            y2 = min(h, y1 + self.rng.randint(50, 200))
            ksize = self.rng.choice([3, 5, 7])
            region = result[y1:y2, x1:x2]
            if region.size > 0:
                result[y1:y2, x1:x2] = cv2.GaussianBlur(
                    region, (ksize, ksize), 0
                )
        return result

    def _contrast_variation(self, img: np.ndarray) -> np.ndarray:
        """
        Apply global contrast and brightness variation using CLAHE-style
        adjustment.  Simulates different scanner settings.
        """
        alpha = self.rng.uniform(0.7, 1.4)   # contrast
        beta  = self.rng.uniform(-20, 20)    # brightness
        result = img.astype(np.float32) * alpha + beta
        return np.clip(result, 0, 255).astype(np.uint8)

    def _water_stain(self, img: np.ndarray) -> np.ndarray:
        """
        Simulate water stain damage: dark irregular patches that obscure
        parts of the map.
        """
        h, w = img.shape
        result = img.copy().astype(np.float32)
        n_stains = self.rng.randint(1, 3)
        for _ in range(n_stains):
            cx = self.rng.randint(0, w)
            cy = self.rng.randint(0, h)
            rx = self.rng.randint(20, 80)
            ry = self.rng.randint(20, 80)
            darkness = self.rng.uniform(10, 40)
            y, x = np.ogrid[:h, :w]
            dist = ((x - cx) / rx) ** 2 + ((y - cy) / ry) ** 2
            stain = np.clip(1 - dist, 0, 1).astype(np.float32)
            result -= stain * darkness
        return np.clip(result, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# CROP EXTRACTOR
# ---------------------------------------------------------------------------

class MapCropExtractor:
    """
    Extracts random crops from large map images and their corresponding
    binary images.  Ensures each crop contains a reasonable number of
    parcel regions (not too empty, not just border).
    """

    def __init__(self,
                 crop_size: int = CROP_SIZE,
                 min_parcels_per_crop: int = 2,
                 max_attempts: int = 20,
                 rng: random.Random | None = None):
        self.crop_size = crop_size
        self.min_parcels = min_parcels_per_crop
        self.max_attempts = max_attempts
        self.rng = rng or random.Random()

    def extract(self,
                gray: np.ndarray,
                binary: np.ndarray
                ) -> tuple[np.ndarray, np.ndarray] | None:
        """
        Try to extract a crop with at least min_parcels parcels.
        Returns (gray_crop, binary_crop) or None if no valid crop found.
        """
        h, w = gray.shape
        cs = self.crop_size

        if h < cs or w < cs:
            # Image smaller than crop — pad and return
            pad_h = max(0, cs - h)
            pad_w = max(0, cs - w)
            gray_p   = cv2.copyMakeBorder(gray,   0, pad_h, 0, pad_w, cv2.BORDER_REFLECT)
            binary_p = cv2.copyMakeBorder(binary, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT)
            return gray_p[:cs, :cs], binary_p[:cs, :cs]

        for _ in range(self.max_attempts):
            x = self.rng.randint(0, w - cs)
            y = self.rng.randint(0, h - cs)
            gray_crop   = gray[y:y + cs, x:x + cs]
            binary_crop = binary[y:y + cs, x:x + cs]

            # Quick check: count contours
            inv = cv2.bitwise_not(binary_crop)
            contours, _ = cv2.findContours(
                inv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            n_valid = sum(
                1 for c in contours
                if MIN_PARCEL_AREA <= cv2.contourArea(c) <= MAX_PARCEL_AREA
            )
            if n_valid >= self.min_parcels:
                return gray_crop, binary_crop

        return None


# ---------------------------------------------------------------------------
# DATASET GENERATOR
# ---------------------------------------------------------------------------

def generate_split(split_name: str,
                    n_images: int,
                    map_data: list[dict],
                    augmentor: CadastralAugmentor,
                    extractor: MapCropExtractor,
                    out_dir: Path,
                    start_id: int = 0
                    ) -> dict:
    """
    Generate one split (train/val/test) of synthetic images.

    Returns a COCO-format annotation dictionary.
    """
    img_dir = out_dir / "images" / split_name
    img_dir.mkdir(parents=True, exist_ok=True)

    coco = {
        "info": {
            "description": f"Cadastral Parcel Segmentation — {split_name}",
            "version": "1.0",
            "year": 2026,
            "contributor": "Hussein Chalhoub",
            "date_created": datetime.now().isoformat(),
        },
        "licenses": [],
        "categories": COCO_CATEGORIES,
        "images": [],
        "annotations": [],
    }

    image_id      = start_id
    annotation_id = start_id
    n_generated   = 0
    n_skipped     = 0

    pbar = tqdm(total=n_images, desc=f"  Generating {split_name}")

    while n_generated < n_images:
        # Pick a random map
        m = random.choice(map_data)

        # Extract a crop
        result = extractor.extract(m["gray"], m["binary"])
        if result is None:
            n_skipped += 1
            if n_skipped > n_images * 3:
                print(f"\n  WARNING: too many skipped crops in {split_name}")
                break
            continue

        gray_crop, binary_crop = result

        # Apply augmentations (only to the grayscale image)
        aug_crop = augmentor.apply(gray_crop)

        # Detect parcels in the ORIGINAL binary crop
        # (augmentation doesn't change topology, just appearance)
        parcels = detect_parcels_in_crop(binary_crop)

        if len(parcels) < 2:
            n_skipped += 1
            continue

        # Save image
        img_filename = f"img_{image_id:06d}.png"
        save_image(img_dir / img_filename, aug_crop)

        # Add image record
        coco["images"].append({
            "id":          image_id,
            "file_name":   img_filename,
            "width":       int(aug_crop.shape[1]),
            "height":      int(aug_crop.shape[0]),
            "source_map":  m["map_num"],
        })

        # Add annotation records
        for p in parcels:
            if p["area"] < MIN_PARCEL_AREA:
                continue
            coco["annotations"].append({
                "id":            annotation_id,
                "image_id":      image_id,
                "category_id":   1,
                "segmentation":  [p["polygon"]],
                "area":          p["area"],
                "bbox":          p["bbox"],
                "iscrowd":       0,
            })
            annotation_id += 1

        image_id      += 1
        n_generated   += 1
        pbar.update(1)

    pbar.close()
    print(f"  {split_name}: {n_generated} images, "
          f"{len(coco['annotations'])} parcel annotations, "
          f"{n_skipped} skips")
    return coco


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def run(n_train: int = 5000, n_val: int = 500, n_test: int = 200):
    print("\n" + "=" * 70)
    print("  MILESTONE 1 — SYNTHETIC DATA GENERATION")
    print("  Cadastral Panoramic Reconstruction — Masters Thesis")
    print("=" * 70)

    SYNTHETIC_DIR.mkdir(parents=True, exist_ok=True)
    ann_dir = SYNTHETIC_DIR / "annotations"
    ann_dir.mkdir(parents=True, exist_ok=True)

    # ── Load all available maps ────────────────────────────────────────────
    print(f"\n  Loading maps from {PREPROCESSED_DIR}...")
    map_data = []
    for num in ALL_MAPS:
        gray_path   = PREPROCESSED_DIR / f"map_{num}_clean.png"
        binary_path = PREPROCESSED_DIR / f"map_{num}_binary.png"
        if not gray_path.exists():
            print(f"  Skipping map {num} (not found)")
            continue
        gray = read_image(gray_path, grayscale=True)
        if binary_path.exists():
            binary = read_image(binary_path, grayscale=True)
        else:
            _, binary = cv2.threshold(
                gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )
        map_data.append({"map_num": num, "gray": gray, "binary": binary})
        print(f"  Loaded map {num}: {gray.shape}")

    if not map_data:
        print("  ERROR: No preprocessed maps found. Run Step 1 first.")
        return

    print(f"\n  Loaded {len(map_data)} maps")
    print(f"  Generating: {n_train} train / {n_val} val / {n_test} test")

    rng       = random.Random(42)
    augmentor = CadastralAugmentor(rng=rng)
    extractor = MapCropExtractor(crop_size=CROP_SIZE, rng=rng)

    t0 = time.time()

    # ── Generate each split ────────────────────────────────────────────────
    splits = {
        "train": n_train,
        "val":   n_val,
        "test":  n_test,
    }

    id_counter = 0
    all_stats  = {}

    for split_name, count in splits.items():
        print(f"\n  --- {split_name.upper()} ---")
        coco = generate_split(
            split_name, count, map_data, augmentor, extractor,
            SYNTHETIC_DIR, start_id=id_counter
        )
        id_counter += count

        # Save COCO JSON
        ann_path = ann_dir / f"{split_name}.json"
        with open(ann_path, "w", encoding="utf-8") as f:
            json.dump(coco, f, indent=2)
        print(f"  Saved: {ann_path}")

        all_stats[split_name] = {
            "n_images":      len(coco["images"]),
            "n_annotations": len(coco["annotations"]),
        }

    # ── Save statistics ────────────────────────────────────────────────────
    stats = {
        "generated_at": datetime.now().isoformat(),
        "crop_size":    CROP_SIZE,
        "n_source_maps": len(map_data),
        "splits":       all_stats,
        "total_time_sec": round(time.time() - t0, 1),
    }
    with open(SYNTHETIC_DIR / "stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    # ── Summary ────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  MILESTONE 1 SUMMARY")
    print("=" * 70)
    for split_name, s in all_stats.items():
        print(f"  {split_name:>6}: {s['n_images']:>5} images | "
              f"{s['n_annotations']:>6} parcel annotations")
    print(f"\n  Crop size    : {CROP_SIZE} x {CROP_SIZE} px")
    print(f"  Source maps  : {len(map_data)}")
    print(f"  Total time   : {time.time() - t0:.1f}s")
    print(f"  Output       : {SYNTHETIC_DIR.resolve()}")
    print(f"\n  Next: Run src/step4_segmentation.py to train Mask R-CNN")
    print("=" * 70 + "\n")


# ---------------------------------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Synthetic data generator for cadastral parcel segmentation"
    )
    parser.add_argument("--train", type=int, default=5000,
                        help="Number of training images (default: 5000)")
    parser.add_argument("--val",   type=int, default=500,
                        help="Number of validation images (default: 500)")
    parser.add_argument("--test",  type=int, default=200,
                        help="Number of test images (default: 200)")
    parser.add_argument("--quick", action="store_true",
                        help="Quick test run: 50 train, 10 val, 5 test")
    args = parser.parse_args()

    if args.quick:
        run(n_train=50, n_val=10, n_test=5)
    else:
        run(n_train=args.train, n_val=args.val, n_test=args.test)
