"""
=============================================================================
STEP 1: IMAGE PREPROCESSING & ENHANCEMENT
=============================================================================
Masters Thesis — AI-Based Panoramic Cadastral Image Reconstruction
Author  : Hussein Chalhoub
Purpose : Clean and normalise raw cadastral blueprint scans before any
          feature-detection or stitching is attempted.

Why preprocessing matters for these specific maps
-------------------------------------------------
The 13 cadastral maps (nos. 43–55) are blue/purple blueprint-style prints,
scanned or photographed under variable conditions.  They exhibit:
  • A dominant blue-purple tint  →  channel separation before grayscale
  • Paper foxing and scanner noise  →  bilateral denoising
  • Fold creases running across parcel boundaries  →  morphological repair
  • Slight rotational skew from hand-held photography  →  deskew
  • Perspective distortion (camera not perfectly parallel to map)  →  warp
  • Uneven illumination / shadow vignette  →  CLAHE contrast equalisation

Output
------
  output/preprocessed/map_<N>_clean.png   — cleaned grayscale image
  output/preprocessed/map_<N>_binary.png  — binarised (black/white) image
  output/preprocessed/map_<N>_debug.png   — side-by-side before/after

=============================================================================
"""

import os
import sys
import time
import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------

DATASET_DIR  = Path("dataset")          # folder containing the 13 raw JPGs
OUTPUT_DIR   = Path("output/preprocessed")
DEBUG_DIR    = Path("output/debug/step1")

# Map file names in stitching order (43 → 55)
MAP_FILES = [
    "الخريطة رقم 43.jpg",
    "الخريطة رقم 44.jpg",
    "الخريطة رقم 45.jpg",
    "الخريطة رقم 46.jpg",
    "الخريطة رقم 47.jpg",
    "الخريطة رقم 48.jpg",
    "الخريطة رقم 49.jpg",
    "الخريطة رقم 50.jpg",
    "الخريطة رقم 51.jpg",
    "الخريطة رقم 52.jpg",
    "الخريطة رقم 53.jpg",
    "الخريطة رقم 54.jpg",
    "الخريطة رقم 55.jpg",
]

# Bilateral filter parameters
# d=5      : neighbourhood diameter — reduced from 9 to preserve fine line sharpness
# sigmaColor=40 : colour similarity range — reduced to avoid blending ink into paper
# sigmaSpace=40 : spatial range — reduced to limit reach of the smoothing kernel
# Previous values (d=9, sigma=75) caused ~90% sharpness loss (Laplacian variance
# dropped from ~90 to ~7), blurring cadastral line detail needed by later steps.
BILATERAL_D           = 5
BILATERAL_SIGMA_COLOR = 40
BILATERAL_SIGMA_SPACE = 40

# Adaptive threshold parameters
# blockSize : pixel neighbourhood size (must be odd)
# C         : constant subtracted from the mean — higher = more white areas
THRESH_BLOCK_SIZE = 25
THRESH_C          = 10

# CLAHE (Contrast Limited Adaptive Histogram Equalisation)
# clipLimit : max contrast amplification (2.0–4.0 typical)
# tileGridSize : local region for equalisation
CLAHE_CLIP_LIMIT   = 2.0
CLAHE_TILE_GRID    = (8, 8)

# Skew correction — maximum angle to attempt correction (degrees)
MAX_SKEW_ANGLE = 10.0


# ---------------------------------------------------------------------------
# UTILITY FUNCTIONS
# ---------------------------------------------------------------------------

def print_step(msg: str):
    """Formatted print for thesis pipeline logging."""
    print(f"  [Step 1] {msg}")


def make_dirs():
    """Create output directories if they don't exist."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    DEBUG_DIR.mkdir(parents=True, exist_ok=True)
    print_step(f"Output directory ready: {OUTPUT_DIR}")
    print_step(f"Debug  directory ready: {DEBUG_DIR}")


def map_number(filename: str) -> str:
    """
    Extract the map number from the Arabic filename.
    e.g. 'الخريطة رقم 43.jpg' → '43'

    Filenames use spaces, not underscores, so we split on whitespace.
    """
    stem = Path(filename).stem          # 'الخريطة رقم 43'
    parts = stem.split()                # split on any whitespace
    return parts[-1]                    # '43'


# ---------------------------------------------------------------------------
# STAGE 1: LOAD IMAGE
# ---------------------------------------------------------------------------

def load_image(filepath: Path) -> np.ndarray:
    """
    Load image using cv2 with Windows Unicode filename support.
    
    On Windows, cv2.imread() fails with non-ASCII filenames (e.g. Arabic).
    Workaround: read file as bytes using np.fromfile and decode with cv2.imdecode.

    Why imread and not PIL?
    cv2 is faster for large images and integrates directly with all
    downstream OpenCV functions.  PIL would require constant conversion.
    """
    data = np.fromfile(str(filepath), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {filepath}")
    h, w, c = img.shape
    print_step(f"Loaded: {filepath.name}  →  shape=({h}, {w}, {c})  dtype={img.dtype}")
    return img


# ---------------------------------------------------------------------------
# STAGE 2: BLUE CHANNEL EXTRACTION
# ---------------------------------------------------------------------------

def extract_blueprint_channel(bgr: np.ndarray) -> np.ndarray:
    """
    The cadastral maps are printed in blue/purple ink on lighter paper.
    In BGR space, the ink is strongest in the Blue channel and sometimes
    also in the Red channel (giving purple).

    Strategy:
      1. Convert to HSV for hue-based isolation
      2. Create a mask for blue-purple hues (hue ≈ 100–160 in OpenCV 0–180)
      3. Return the masked image for grayscale conversion

    This removes the yellow/brown paper aging artifacts that would confuse
    adaptive thresholding later.

    OpenCV HSV ranges:
      Hue       : 0–180  (0=red, 60=yellow, 120=blue, 180=red again)
      Saturation: 0–255
      Value     : 0–255
    """
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

    # Blue-purple hue range for the blueprint ink
    lower_blue = np.array([90,  20, 20])
    upper_blue = np.array([160, 255, 255])
    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    # Dilate mask slightly to include partially saturated ink pixels
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.dilate(mask, kernel, iterations=1)

    # Apply mask — ink regions keep colour, paper becomes white
    result = bgr.copy()
    result[mask == 0] = 255   # paper background → white

    ink_pixels = int(np.sum(mask > 0))
    total_pixels = mask.size
    print_step(f"Channel extraction: ink pixels = {ink_pixels:,} "
               f"({100 * ink_pixels / total_pixels:.1f}% of image)")
    return result


# ---------------------------------------------------------------------------
# STAGE 3: GRAYSCALE CONVERSION
# ---------------------------------------------------------------------------

def to_grayscale(bgr: np.ndarray) -> np.ndarray:
    """
    Convert BGR to grayscale using the standard luminance formula:
      Y = 0.114·B + 0.587·G + 0.299·R

    After channel extraction the background is white (255) and the ink is
    dark, so the grayscale image will have dark lines on a bright field —
    the correct orientation for cadastral line maps.
    """
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    mean_val = float(np.mean(gray))
    print_step(f"Grayscale conversion done — mean pixel value: {mean_val:.1f} / 255")
    return gray


# ---------------------------------------------------------------------------
# STAGE 4: NOISE REDUCTION (BILATERAL FILTER)
# ---------------------------------------------------------------------------

def denoise(gray: np.ndarray) -> np.ndarray:
    """
    Bilateral filter: reduces scanner noise while preserving sharp edges.

    Why bilateral and NOT Gaussian blur?
    Gaussian blur is isotropic — it smooths across all edges, including the
    fine parcel boundary lines and Arabic numerals.  Bilateral filter is
    edge-aware: it measures colour similarity between neighbouring pixels
    and only averages pixels with similar intensities.  This means:
      ✓ Smooth, flat paper regions get blurred (noise removed)
      ✓ Sharp ink-to-paper transitions are preserved

    Parameters chosen:
      d=5 limits the neighbourhood to avoid over-smoothing fine line detail.
      sigmaColor=40 keeps colour blending tight so ink edges are not merged
      into the paper background — critical for detecting cadastral boundaries.

    Trade-off: bilateral is slow (~2–5 s/image).  For thesis demo this is
    acceptable.  Production would use fastNlMeansDenoisingColored instead.
    """
    denoised = cv2.bilateralFilter(gray, BILATERAL_D,
                                   BILATERAL_SIGMA_COLOR,
                                   BILATERAL_SIGMA_SPACE)
    # Measure noise reduction: std dev before vs after
    noise_before = float(np.std(gray.astype(np.float32)))
    noise_after  = float(np.std(denoised.astype(np.float32)))
    print_step(f"Bilateral filter applied — std dev: {noise_before:.1f} → {noise_after:.1f}")
    return denoised


# ---------------------------------------------------------------------------
# STAGE 5: CONTRAST ENHANCEMENT (CLAHE)
# ---------------------------------------------------------------------------

def enhance_contrast(gray: np.ndarray) -> np.ndarray:
    """
    CLAHE — Contrast Limited Adaptive Histogram Equalisation.

    Standard global histogram equalisation would over-amplify the dark
    creases and background variations.  CLAHE divides the image into small
    tiles (tileGridSize) and equalises each tile independently, then
    bilinearly interpolates tile boundaries.  The clipLimit prevents
    excessive amplification in any one tile.

    For cadastral maps this sharpens faint parcel boundaries and makes
    faded parcel numbers more visible before thresholding.

    Note: CLAHE operates on single-channel 8-bit images, which is exactly
    what we have after grayscale conversion.
    """
    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP_LIMIT,
                             tileGridSize=CLAHE_TILE_GRID)
    enhanced = clahe.apply(gray)
    print_step(f"CLAHE applied — clip limit={CLAHE_CLIP_LIMIT}, "
               f"tile grid={CLAHE_TILE_GRID}")
    return enhanced


# ---------------------------------------------------------------------------
# STAGE 6: SKEW DETECTION & CORRECTION
# ---------------------------------------------------------------------------

def detect_skew_angle(gray: np.ndarray) -> float:
    """
    Estimate the rotational skew of the scanned map.

    Method:
      1. Canny edge detection to find all edges
      2. Probabilistic Hough Line Transform to detect line segments
      3. Compute the angle of each line relative to horizontal
      4. Take the circular median of all angles (robust to outliers)
      5. Return the dominant skew angle

    Cadastral maps are full of straight boundary lines, making Hough-based
    skew detection very reliable here compared to text-based methods.

    Returns angle in degrees (positive = clockwise tilt).
    """
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)

    # HoughLinesP returns line segments as (x1,y1,x2,y2)
    lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180,
                             threshold=100, minLineLength=100,
                             maxLineGap=10)

    if lines is None or len(lines) == 0:
        print_step("Skew detection: no lines found — assuming 0° skew")
        return 0.0

    angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        if x2 != x1:
            angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
            # Normalise to (-45°, 45°) range
            if angle < -45:
                angle += 90
            elif angle > 45:
                angle -= 90
            angles.append(angle)

    if not angles:
        return 0.0

    median_angle = float(np.median(angles))
    print_step(f"Skew detection: {len(angles)} lines analysed — "
               f"median angle = {median_angle:.2f}°")
    return median_angle


def correct_skew(gray: np.ndarray, angle: float) -> np.ndarray:
    """
    Rotate the image by +angle to deskew it.

    We rotate around the image centre and use BORDER_REPLICATE to fill
    the corners created by rotation (avoids black corner triangles that
    would confuse later feature detection).

    Only corrects angles within MAX_SKEW_ANGLE to avoid
    over-correcting perspective-distorted images.
    """
    if abs(angle) < 0.5:
        print_step(f"Skew angle {angle:.2f}° < 0.5° — no correction needed")
        return gray

    if abs(angle) > MAX_SKEW_ANGLE:
        print_step(f"Skew angle {angle:.2f}° exceeds max ({MAX_SKEW_ANGLE}°) "
                   f"— skipping (may be perspective distortion instead)")
        return gray

    h, w = gray.shape
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, scale=1.0)
    corrected = cv2.warpAffine(gray, M, (w, h),
                                flags=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_REPLICATE)
    print_step(f"Skew corrected by {angle:.2f}°")
    return corrected


# ---------------------------------------------------------------------------
# STAGE 7: PERSPECTIVE DISTORTION CORRECTION
# ---------------------------------------------------------------------------

def correct_perspective(gray: np.ndarray) -> np.ndarray:
    """
    Detect the map's outer border and warp it to a rectangle.

    Many of the 13 maps were photographed at a slight angle, causing
    trapezoidal distortion.  This stage:
      1. Finds the largest contour in the thresholded image
         (assumed to be the map border / blue outer rectangle)
      2. Approximates it as a quadrilateral
      3. Applies getPerspectiveTransform + warpPerspective

    If no clear quadrilateral border is found (e.g. maps 43, 44 which
    show only partial boundaries) we skip this step gracefully.

    Note: This is a heuristic.  For a full thesis, this would be replaced
    with a deep-learning document corner detector (e.g. DocUNet).
    """
    # Threshold to find the dark border frame
    _, thresh = cv2.threshold(gray, 0, 255,
                               cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        print_step("Perspective correction: no contours found — skipping")
        return gray

    # Largest contour by area
    largest = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest)
    image_area = gray.shape[0] * gray.shape[1]

    # Must cover at least 30% of the image to be the map border
    if area < 0.30 * image_area:
        print_step(f"Perspective correction: largest contour area "
                   f"({area / image_area:.0%}) too small — skipping")
        return gray

    # Approximate to quadrilateral
    peri = cv2.arcLength(largest, True)
    approx = cv2.approxPolyDP(largest, 0.02 * peri, True)

    if len(approx) != 4:
        print_step(f"Perspective correction: corner approx gave "
                   f"{len(approx)} points (need 4) — skipping")
        return gray

    # Order corners: top-left, top-right, bottom-right, bottom-left
    pts = approx.reshape(4, 2).astype(np.float32)
    rect = order_corners(pts)

    # Destination rectangle (full image size)
    h, w = gray.shape
    dst = np.array([[0, 0], [w - 1, 0],
                    [w - 1, h - 1], [0, h - 1]], dtype=np.float32)

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(gray, M, (w, h),
                                  flags=cv2.INTER_LINEAR,
                                  borderMode=cv2.BORDER_REPLICATE)
    print_step("Perspective correction applied (4-corner warp)")
    return warped


def order_corners(pts: np.ndarray) -> np.ndarray:
    """
    Sort 4 corner points into [top-left, top-right, bottom-right, bottom-left]
    order, required by getPerspectiveTransform.
    """
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]   # top-left:     smallest x+y
    rect[2] = pts[np.argmax(s)]   # bottom-right: largest  x+y
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # top-right:    smallest y-x
    rect[3] = pts[np.argmax(diff)]  # bottom-left:  largest  y-x
    return rect


# ---------------------------------------------------------------------------
# STAGE 8: ADAPTIVE THRESHOLDING (BINARISATION)
# ---------------------------------------------------------------------------

def binarise(gray: np.ndarray) -> np.ndarray:
    """
    Convert the cleaned grayscale image to binary (black lines on white).

    Why adaptive and NOT global (Otsu) thresholding?
    Global thresholding finds one single threshold value for the whole image.
    The cadastral maps have uneven illumination — the centre may be brighter
    than the edges due to camera/scanner vignetting.  A global threshold
    would binarise the bright centre correctly but leave the dark edges
    mostly black, wiping out boundary lines.

    Adaptive (Gaussian) thresholding:
      • Divides the image into overlapping blockSize×blockSize windows
      • For each window computes a Gaussian-weighted mean intensity
      • Sets the threshold = mean − C for that window
      • Handles illumination gradients naturally

    Parameters:
      blockSize=25  covers ~10 line-widths at typical DPI, large enough to
                    span the paper texture but small enough to adapt to
                    local brightness changes.
      C=10          conservative constant — reduces false positives from
                    paper texture while keeping faint ink lines.

    Output: THRESH_BINARY_INV → ink = 255 (white), paper = 0 (black)
    We then invert back: ink = 0 (black), paper = 255 (white) — the
    conventional representation for cadastral line maps.
    """
    binary_inv = cv2.adaptiveThreshold(
        gray,
        maxValue=255,
        adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        thresholdType=cv2.THRESH_BINARY_INV,
        blockSize=THRESH_BLOCK_SIZE,
        C=THRESH_C
    )
    # Invert so ink = black (0), paper = white (255)
    binary = cv2.bitwise_not(binary_inv)

    ink_ratio = float(np.sum(binary == 0)) / binary.size
    print_step(f"Adaptive threshold done — ink coverage: {ink_ratio:.2%} of image")
    return binary


# ---------------------------------------------------------------------------
# STAGE 9: MORPHOLOGICAL CREASE & ARTIFACT REMOVAL
# ---------------------------------------------------------------------------

def remove_artifacts(binary: np.ndarray) -> np.ndarray:
    """
    The 13 maps were folded for storage, leaving horizontal and vertical
    crease lines across parcel boundaries.  Morphological operations
    can suppress these without destroying the parcel lines.

    Strategy:
      1. Opening (erosion then dilation): removes thin isolated ink-noise
         pixels that are smaller than the structuring element.
         Kernel: 3×3 ellipse — removes single-pixel scanner salt noise.
      2. Closing (dilation then erosion): closes small white gaps in
         continuous parcel boundary lines caused by faded ink or creases.
         Kernel: 2×2 rect — reconnects broken line segments.

    Why not just a large closing?  Too large a kernel merges adjacent
    parcel boundaries — destroying the segmentation we need later.

    IMPORTANT — convention note:
      The input `binary` has ink=0, paper=255 (dark lines on white).
      OpenCV morphological operations treat 255 as foreground, so we
      invert to ink=255 before applying, then invert back.  Without this,
      OPEN would remove paper islands (not ink noise) and CLOSE would fill
      ink pixels rather than reconnecting broken lines.

    This is a heuristic morphological approach.  For a full thesis
    discussion, contrast with inpainting approaches (cv2.inpaint) which
    can reconstruct missing content from surrounding pixels.
    """
    # Invert so ink becomes foreground (255) for standard morphological semantics
    ink = cv2.bitwise_not(binary)   # ink=255, paper=0

    # Opening: remove isolated ink noise pixels (smaller than 3×3 ellipse)
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    opened = cv2.morphologyEx(ink, cv2.MORPH_OPEN, kernel_open,
                               iterations=1)

    # Closing: reconnect broken ink line segments (fill small white gaps)
    kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel_close,
                               iterations=1)

    # Invert back to original convention: ink=0, paper=255
    result = cv2.bitwise_not(closed)

    pixels_removed = int(np.sum(binary == 0)) - int(np.sum(result == 0))
    print_step(f"Morphological cleanup — noise pixels removed: {pixels_removed:,}")
    return result


# ---------------------------------------------------------------------------
# STAGE 10: SAVE OUTPUTS & QUALITY CHECK
# ---------------------------------------------------------------------------

def compute_quality_metrics(original: np.ndarray,
                             processed: np.ndarray) -> dict:
    """
    Compute basic quality metrics to report in the thesis.

    Metrics:
      - sharpness  : Laplacian variance — higher = sharper edges
      - contrast   : standard deviation of pixel values
      - snr        : signal-to-noise ratio estimate

    These are logged and can be used to compare preprocessing settings.
    """
    # Sharpness via Laplacian variance
    lap_orig = cv2.Laplacian(original,  cv2.CV_64F).var()
    lap_proc = cv2.Laplacian(processed, cv2.CV_64F).var()

    # Contrast = std dev
    std_orig = float(np.std(original))
    std_proc = float(np.std(processed))

    return {
        "sharpness_before": round(lap_orig, 2),
        "sharpness_after":  round(lap_proc, 2),
        "contrast_before":  round(std_orig, 2),
        "contrast_after":   round(std_proc, 2),
    }


def save_outputs(map_num: str,
                 original_bgr: np.ndarray,
                 gray_clean: np.ndarray,
                 binary: np.ndarray):
    """
    Save three outputs per map:
      1. map_<N>_clean.png   — cleaned grayscale (main output for steps 2-6)
      2. map_<N>_binary.png  — binarised version (used for OCR in step 5)
      3. map_<N>_debug.png   — side-by-side comparison for thesis figures
    
    Uses Unicode-safe file writing (cv2.imencode + np.tofile).
    """
    clean_path  = OUTPUT_DIR / f"map_{map_num}_clean.png"
    binary_path = OUTPUT_DIR / f"map_{map_num}_binary.png"
    debug_path  = DEBUG_DIR  / f"map_{map_num}_debug.png"

    # Unicode-safe writing using imencode + tofile
    clean_path.parent.mkdir(parents=True, exist_ok=True)
    _, enc = cv2.imencode('.png', gray_clean)
    enc.tofile(str(clean_path))
    
    _, enc = cv2.imencode('.png', binary)
    enc.tofile(str(binary_path))

    # Debug: side-by-side original (resized) vs clean vs binary
    h = 400
    def resize_h(img, target_h):
        scale = target_h / img.shape[0]
        w = int(img.shape[1] * scale)
        return cv2.resize(img, (w, target_h))

    orig_small  = resize_h(cv2.cvtColor(original_bgr, cv2.COLOR_BGR2GRAY), h)
    clean_small = resize_h(gray_clean, h)
    bin_small   = resize_h(binary, h)

    # Make all same width for hstack
    min_w = min(orig_small.shape[1], clean_small.shape[1], bin_small.shape[1])
    orig_small  = orig_small[:, :min_w]
    clean_small = clean_small[:, :min_w]
    bin_small   = bin_small[:, :min_w]

    divider = np.ones((h, 4), dtype=np.uint8) * 128
    debug_img = np.hstack([orig_small, divider, clean_small, divider, bin_small])
    
    debug_path.parent.mkdir(parents=True, exist_ok=True)
    _, enc = cv2.imencode('.png', debug_img)
    enc.tofile(str(debug_path))

    print_step(f"Saved: {clean_path.name}")
    print_step(f"Saved: {binary_path.name}")
    print_step(f"Saved (debug): {debug_path.name}")


def preprocess_map(filepath: Path) -> dict:
    """
    Run the full 8-stage preprocessing pipeline on a single map image.

    Returns a dict with quality metrics for logging / thesis table.
    """
    map_num = map_number(filepath.name)
    print(f"\n{'='*60}")
    print(f"  Processing map {map_num}: {filepath.name}")
    print(f"{'='*60}")
    t0 = time.time()

    # 1. Load
    bgr = load_image(filepath)

    # 2. Blue channel extraction (blueprint-specific)
    bgr_clean = extract_blueprint_channel(bgr)

    # 3. Grayscale
    gray = to_grayscale(bgr_clean)

    # 4. Noise reduction
    gray = denoise(gray)

    # 5. Contrast enhancement
    gray = enhance_contrast(gray)

    # 6. Skew correction
    angle = detect_skew_angle(gray)
    gray  = correct_skew(gray, angle)

    # 7. Perspective correction
    gray = correct_perspective(gray)

    # 8. Binarisation
    binary = binarise(gray)

    # 9. Artifact removal
    binary = remove_artifacts(binary)

    # 10. Quality metrics
    original_gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    metrics = compute_quality_metrics(original_gray, gray)
    print_step(f"Quality — sharpness: {metrics['sharpness_before']:.0f} → "
               f"{metrics['sharpness_after']:.0f} | "
               f"contrast: {metrics['contrast_before']:.1f} → "
               f"{metrics['contrast_after']:.1f}")

    # 11. Save
    save_outputs(map_num, bgr, gray, binary)

    elapsed = time.time() - t0
    print_step(f"Map {map_num} done in {elapsed:.1f}s")

    return {"map": map_num, "elapsed": elapsed, **metrics}


# ---------------------------------------------------------------------------
# VISUALIZATION & REPORTING
# ---------------------------------------------------------------------------

def create_summary_visualization(all_metrics: list) -> None:
    """
    Create a 3-panel quality metrics plot: sharpness gain, contrast change, time per map.
    Saved as PNG for inclusion in thesis.
    """
    if not all_metrics:
        return

    maps = [m["map"] for m in all_metrics]
    sharp_before = [m["sharpness_before"] for m in all_metrics]
    sharp_after = [m["sharpness_after"] for m in all_metrics]
    contrast_before = [m["contrast_before"] for m in all_metrics]
    contrast_after = [m["contrast_after"] for m in all_metrics]
    times = [m["elapsed"] for m in all_metrics]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle("Step 1 Preprocessing Quality Metrics", fontsize=14, fontweight="bold")

    # Sharpness comparison
    x = np.arange(len(maps))
    width = 0.35
    axes[0].bar(x - width/2, sharp_before, width, label="Before", alpha=0.7)
    axes[0].bar(x + width/2, sharp_after, width, label="After", alpha=0.7)
    axes[0].set_xlabel("Map")
    axes[0].set_ylabel("Laplacian Variance (sharpness)")
    axes[0].set_title("Sharpness (Higher = Sharper)")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(maps, rotation=45, ha="right")
    axes[0].legend()
    axes[0].grid(axis="y", alpha=0.3)

    # Contrast comparison
    axes[1].bar(x - width/2, contrast_before, width, label="Before", alpha=0.7)
    axes[1].bar(x + width/2, contrast_after, width, label="After", alpha=0.7)
    axes[1].set_xlabel("Map")
    axes[1].set_ylabel("Std Dev (contrast)")
    axes[1].set_title("Contrast (Higher = More Detail)")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(maps, rotation=45, ha="right")
    axes[1].legend()
    axes[1].grid(axis="y", alpha=0.3)

    # Processing time
    axes[2].bar(maps, times, color="steelblue", alpha=0.7)
    axes[2].set_xlabel("Map")
    axes[2].set_ylabel("Processing Time (seconds)")
    axes[2].set_title("Speed per Map")
    axes[2].set_xticklabels(maps, rotation=45, ha="right")
    axes[2].grid(axis="y", alpha=0.3)
    avg_time = np.mean(times)
    axes[2].axhline(avg_time, color="red", linestyle="--", linewidth=2, label=f"Avg: {avg_time:.1f}s")
    axes[2].legend()

    plt.tight_layout()
    report_dir = Path("output/reports")
    report_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(report_dir / "step1_quality_metrics.png", dpi=150, bbox_inches="tight")
    print_step(f"Saved quality metrics plot: {report_dir / 'step1_quality_metrics.png'}")
    plt.close()


def create_before_after_grid() -> None:
    """
    Create a grid showing 2–3 sample maps with original → clean → binary progression.
    Useful for thesis figures.
    """
    samples = list(range(0, min(3, len(MAP_FILES))))  # First 0, 1, 2
    if not samples:
        return

    fig = plt.figure(figsize=(14, 8))
    gs = gridspec.GridSpec(len(samples), 3, figure=fig, hspace=0.3, wspace=0.1)
    fig.suptitle("Sample: Original → Cleaned → Binary", fontsize=14, fontweight="bold")

    for row, idx in enumerate(samples):
        fname = MAP_FILES[idx]
        map_num = map_number(fname)

        # Load original
        orig_path = DATASET_DIR / fname
        orig_data = np.fromfile(str(orig_path), dtype=np.uint8)
        orig = cv2.imdecode(orig_data, cv2.IMREAD_COLOR)
        orig_gray = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY)

        # Load processed
        clean_path = OUTPUT_DIR / f"map_{map_num}_clean.png"
        binary_path = OUTPUT_DIR / f"map_{map_num}_binary.png"

        clean_data = np.fromfile(str(clean_path), dtype=np.uint8)
        clean = cv2.imdecode(clean_data, cv2.IMREAD_GRAYSCALE)

        binary_data = np.fromfile(str(binary_path), dtype=np.uint8)
        binary = cv2.imdecode(binary_data, cv2.IMREAD_GRAYSCALE)

        # Resize for display
        scale = min(1.0, 400 / orig_gray.shape[0])
        h_display = int(orig_gray.shape[0] * scale)
        w_display = int(orig_gray.shape[1] * scale)

        orig_small = cv2.resize(orig_gray, (w_display, h_display), interpolation=cv2.INTER_AREA)
        clean_small = cv2.resize(clean, (w_display, h_display), interpolation=cv2.INTER_AREA)
        binary_small = cv2.resize(binary, (w_display, h_display), interpolation=cv2.INTER_AREA)

        # Plot
        ax1 = fig.add_subplot(gs[row, 0])
        ax1.imshow(orig_small, cmap="gray")
        ax1.set_title(f"Map {map_num}: Original", fontsize=10)
        ax1.axis("off")

        ax2 = fig.add_subplot(gs[row, 1])
        ax2.imshow(clean_small, cmap="gray")
        ax2.set_title(f"Map {map_num}: Cleaned", fontsize=10)
        ax2.axis("off")

        ax3 = fig.add_subplot(gs[row, 2])
        ax3.imshow(binary_small, cmap="gray")
        ax3.set_title(f"Map {map_num}: Binary", fontsize=10)
        ax3.axis("off")

    report_dir = Path("output/reports")
    report_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(report_dir / "step1_before_after_samples.png", dpi=150, bbox_inches="tight")
    print_step(f"Saved before/after samples: {report_dir / 'step1_before_after_samples.png'}")
    plt.close()


def generate_html_report(all_metrics: list, missing: list) -> str:
    """
    Generate an HTML summary report for easy viewing and inclusion in thesis.
    """
    report_dir = Path("output/reports")
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / "step1_summary.html"

    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <title>Step 1 Preprocessing Summary</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }
            h1 { color: #333; border-bottom: 3px solid #007acc; padding-bottom: 10px; }
            h2 { color: #555; margin-top: 30px; }
            table { border-collapse: collapse; width: 100%; background-color: white; margin-top: 10px; }
            th, td { border: 1px solid #ddd; padding: 12px; text-align: left; }
            th { background-color: #007acc; color: white; font-weight: bold; }
            tr:nth-child(even) { background-color: #f9f9f9; }
            tr:hover { background-color: #f0f0f0; }
            .success { color: green; font-weight: bold; }
            .warning { color: orange; font-weight: bold; }
            .metric { margin: 10px 0; padding: 10px; background-color: #e8f4f8; border-left: 4px solid #007acc; }
            img { max-width: 100%; height: auto; margin-top: 15px; border: 1px solid #ccc; }
        </style>
    </head>
    <body>
        <h1>Step 1: Image Preprocessing &amp; Enhancement</h1>
        <p><strong>Thesis:</strong> AI-Based Panoramic Cadastral Image Reconstruction</p>
        <p><strong>Date:</strong> """ + time.strftime("%Y-%m-%d %H:%M:%S") + """</p>

        <h2>Processing Summary</h2>
        <div class="metric">
            <strong>Maps Processed:</strong> <span class="success">""" + str(len(all_metrics)) + """</span>
        </div>
    """

    if missing:
        html_content += f"""
        <div class="metric">
            <strong>Maps Missing:</strong> <span class="warning">{len(missing)}</span> — {', '.join(missing)}
        </div>
        """

    if all_metrics:
        avg_sharp_gain = np.mean([m["sharpness_after"] - m["sharpness_before"] for m in all_metrics])
        avg_time = np.mean([m["elapsed"] for m in all_metrics])
        total_time = sum(m["elapsed"] for m in all_metrics)

        html_content += f"""
        <div class="metric">
            <strong>Average Processing Time:</strong> {avg_time:.1f}s per map
        </div>
        <div class="metric">
            <strong>Total Processing Time:</strong> {total_time:.1f}s ({total_time/60:.1f} minutes)
        </div>
        <div class="metric">
            <strong>Average Sharpness Gain:</strong> {avg_sharp_gain:.1f}
        </div>

        <h2>Per-Map Quality Metrics</h2>
        <table>
            <tr>
                <th>Map</th>
                <th>Sharpness Before</th>
                <th>Sharpness After</th>
                <th>Contrast Before</th>
                <th>Contrast After</th>
                <th>Processing Time</th>
            </tr>
        """

        for m in all_metrics:
            html_content += f"""
            <tr>
                <td><strong>{m['map']}</strong></td>
                <td>{m['sharpness_before']:.0f}</td>
                <td>{m['sharpness_after']:.0f}</td>
                <td>{m['contrast_before']:.1f}</td>
                <td>{m['contrast_after']:.1f}</td>
                <td>{m['elapsed']:.1f}s</td>
            </tr>
            """

        html_content += """
        </table>

        <h2>Quality Metrics Visualization</h2>
        <img src="step1_quality_metrics.png" alt="Quality Metrics">

        <h2>Sample Preprocessing Results</h2>
        <img src="step1_before_after_samples.png" alt="Before/After Samples">

        <h2>Output Files</h2>
        <p>All preprocessed images saved to: <code>output/preprocessed/</code></p>
        <ul>
            <li><strong>map_&lt;N&gt;_clean.png</strong> — Cleaned grayscale image (main output)</li>
            <li><strong>map_&lt;N&gt;_binary.png</strong> — Binarized version for OCR/segmentation</li>
        </ul>

        <p>Debug comparisons saved to: <code>output/debug/step1/</code></p>
        <ul>
            <li><strong>map_&lt;N&gt;_debug.png</strong> — Side-by-side original/clean/binary</li>
        </ul>

        <h2>Processing Pipeline Stages</h2>
        <ol>
            <li>Blue channel extraction (blueprint-specific color isolation)</li>
            <li>Grayscale conversion</li>
            <li>Bilateral denoising (edge-aware noise reduction)</li>
            <li>CLAHE contrast enhancement (adaptive local histogram)</li>
            <li>Skew detection &amp; correction (Hough line-based)</li>
            <li>Perspective distortion correction (4-corner warp)</li>
            <li>Adaptive binarization (Gaussian local thresholding)</li>
            <li>Morphological cleanup (remove small noise, reconnect lines)</li>
        </ol>

        <hr>
        <p><em>Report generated automatically by step1_preprocessing.py</em></p>
    </body>
    </html>
    """

    report_path.write_text(html_content, encoding="utf-8")
    print_step(f"Saved HTML report: {report_path}")


    return html_content

def run():
    """
    Process all 13 maps in stitching order and print a summary table.
    Called from main.py or run directly: python step1_preprocessing.py
    """
    print("\n" + "="*60)
    print("  STEP 1 — IMAGE PREPROCESSING & ENHANCEMENT")
    print("  Cadastral Panoramic Reconstruction — Masters Thesis")
    print("="*60)

    make_dirs()

    all_metrics = []
    missing = []

    for fname in MAP_FILES:
        fpath = DATASET_DIR / fname
        if not fpath.exists():
            print(f"\n  [Step 1] WARNING: file not found — {fpath}")
            missing.append(fname)
            continue
        metrics = preprocess_map(fpath)
        all_metrics.append(metrics)

    # -----------------------------------------------------------------------
    # SUMMARY REPORT
    # -----------------------------------------------------------------------
    print("\n" + "="*60)
    print("  STEP 1 SUMMARY")
    print("="*60)
    print(f"  Maps processed : {len(all_metrics)}")
    if missing:
        print(f"  Maps missing   : {len(missing)} — {', '.join(missing)}")

    if all_metrics:
        avg_sharpness_gain = np.mean([
            m["sharpness_after"] - m["sharpness_before"]
            for m in all_metrics
        ])
        avg_elapsed = np.mean([m["elapsed"] for m in all_metrics])
        total_elapsed = sum(m["elapsed"] for m in all_metrics)

        print(f"\n  Average sharpness gain : +{avg_sharpness_gain:.1f}")
        print(f"  Average time per map   : {avg_elapsed:.1f}s")
        print(f"  Total processing time  : {total_elapsed:.1f}s")
        print(f"\n  Output directory: {OUTPUT_DIR.resolve()}")
        print(f"  Debug  directory: {DEBUG_DIR.resolve()}")

        print("\n  Per-map quality table:")
        print(f"  {'Map':>4} | {'Sharp_B':>8} | {'Sharp_A':>8} | "
              f"{'Contrast_B':>10} | {'Contrast_A':>10} | {'Time':>6}")
        print("  " + "-"*58)
        for m in all_metrics:
            print(f"  {m['map']:>4} | "
                  f"{m['sharpness_before']:>8.0f} | "
                  f"{m['sharpness_after']:>8.0f} | "
                  f"{m['contrast_before']:>10.1f} | "
                  f"{m['contrast_after']:>10.1f} | "
                  f"{m['elapsed']:>5.1f}s")

    success = len(all_metrics) == len(MAP_FILES)
    status = "SUCCESS" if success else "PARTIAL"
    print(f"\n  [Step 1] Status: {status}")
    print("="*60 + "\n")

    # Generate visualizations and reports
    print("\n" + "="*60)
    print("  GENERATING VISUALIZATIONS & REPORTS")
    print("="*60)
    create_summary_visualization(all_metrics)
    create_before_after_grid()
    generate_html_report(all_metrics, missing)

    print("\n" + "="*60)
    print("  STEP 1 COMPLETE")
    print("="*60)
    print(f"  ✓ Preprocessed {len(all_metrics)} maps")
    print(f"  ✓ Generated quality metrics visualization")
    print(f"  ✓ Generated before/after samples")
    print(f"  ✓ Generated HTML summary report")
    print(f"\n  Open this file in a browser for full report:")
    print(f"    output/reports/step1_summary.html")
    print("="*60 + "\n")

    return success


if __name__ == "__main__":
    ok = run()
    sys.exit(0 if ok else 1)
