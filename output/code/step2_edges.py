"""
=============================================================================
STEP 2: EDGE DETECTION, HOUGH TRANSFORM & SEGMENTATION
=============================================================================
Masters Thesis — AI-Based Panoramic Cadastral Image Reconstruction
Author  : Hussein Chalhoub
Purpose : Detect parcel boundary lines and segment individual land parcels
          from the preprocessed cadastral map images produced by Step 1.

Why this step is essential
--------------------------
Cadastral maps are fundamentally line drawings.  Every parcel boundary is
a drawn line, and every parcel number sits inside a closed polygon formed
by those lines.  Before we can match overlapping maps (Step 4) or detect
parcel numbers (Step 5), we need to:

  1. Know WHERE the boundary lines are        → Canny edge detection
  2. Know WHICH lines are structural features → Hough line transform
  3. Know WHICH closed regions are parcels    → Contour segmentation

This step operates on the CLEANED GRAYSCALE images from Step 1
(output/preprocessed/map_<N>_clean.png).

Pipeline inside this step
--------------------------
  Stage A: Canny edge detection
           → Finds all sharp intensity transitions (ink edges)
  Stage B: Edge cleaning & dilation
           → Connects broken edge fragments, removes noise edges
  Stage C: Probabilistic Hough Line Transform
           → Detects straight line segments (parcel boundaries)
  Stage D: Line classification
           → Groups lines by orientation (horizontal / vertical / diagonal)
  Stage E: Contour-based parcel segmentation
           → Finds closed regions = individual land parcels
  Stage F: Parcel statistics
           → Area, centroid, bounding box for each parcel
  Stage G: Visualisation & save
           → Colour-coded output images for thesis figures

Output files (per map N)
------------------------
  output/edges/map_<N>_edges.png       — raw Canny edge map
  output/edges/map_<N>_edges_clean.png — cleaned / dilated edge map
  output/hough/map_<N>_lines.png       — detected Hough lines overlaid
  output/hough/map_<N>_lines_classified.png  — H/V/D lines colour-coded
  output/segments/map_<N>_segments.png — colour-labelled parcel regions
  output/segments/map_<N>_parcels.png  — parcel bounding boxes overlaid
  output/segments/map_<N>_stats.csv    — per-parcel area / centroid table

=============================================================================
"""

import os
import sys
import time
import csv
import re
import cv2
import numpy as np
from pathlib import Path


# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------

PREPROCESSED_DIR = Path("output/preprocessed")   # Step 1 output
EDGES_DIR        = Path("output/edges")
HOUGH_DIR        = Path("output/hough")
SEGMENTS_DIR     = Path("output/segments")
DEBUG_DIR        = Path("output/debug/step2")

# Map numbers in stitching order
MAP_NUMBERS = [str(n) for n in range(43, 56)]     # ['43','44',...,'55']

# Missing maps listed here are considered expected and do not downgrade
# final status from SUCCESS to PARTIAL.
EXPECTED_MISSING_MAPS = set()

# ── Canny parameters ──────────────────────────────────────────────────────
# threshold1 : lower hysteresis threshold
#              Pixels below this are definitely NOT edges
# threshold2 : upper hysteresis threshold
#              Pixels above this are definitely edges
#              Pixels between the two thresholds are edges only if connected
#              to a definite edge (hysteresis linking)
# apertureSize : Sobel kernel size (3 = standard, 5 for thicker lines)
# L2gradient   : use more accurate L2 norm for gradient magnitude
CANNY_THRESH1      = 30
CANNY_THRESH2      = 90
CANNY_APERTURE     = 3
CANNY_L2GRADIENT   = True

# ── Edge cleaning parameters ──────────────────────────────────────────────
# Dilation connects broken edge fragments caused by faded ink or scan noise
# kernel size 3×3 with 1 iteration is conservative — larger values merge
# adjacent boundary lines which destroys parcel topology
EDGE_DILATE_KERNEL = (3, 3)
EDGE_DILATE_ITER   = 1

# ── Probabilistic Hough Line Transform parameters ─────────────────────────
# rho         : distance resolution (pixels)
# theta       : angle resolution (radians)
# threshold   : minimum number of votes (intersections) for a line
#               Higher = only detect longer / stronger lines
# minLineLength : minimum pixel length of a detected line segment
#               Set relative to image size — we use a fraction below
# maxLineGap  : maximum gap (pixels) between collinear points to join them
HOUGH_RHO           = 1
HOUGH_THETA         = np.pi / 180       # 1-degree resolution
HOUGH_THRESHOLD     = 80
HOUGH_MIN_LINE_LEN  = 60               # overridden per image (see below)
HOUGH_MAX_LINE_GAP  = 15

# ── Line classification angle thresholds ─────────────────────────────────
# Lines within ±HORIZONTAL_TOL degrees of 0° are "horizontal"
# Lines within ±VERTICAL_TOL degrees of 90° are "vertical"
# Everything else is "diagonal"
HORIZONTAL_TOL = 15   # degrees
VERTICAL_TOL   = 15   # degrees

# ── Contour / segmentation parameters ────────────────────────────────────
# MIN_PARCEL_AREA : ignore tiny contours (scanner artifacts, text fragments)
#                   Set in pixels² — typical parcel is several thousand px²
# MAX_PARCEL_AREA : ignore the map border contour (nearly full image area)
#                   Expressed as a fraction of total image area
MIN_PARCEL_AREA       = 500      # px²
MAX_PARCEL_AREA_FRAC  = 0.85    # fraction of image area


# ---------------------------------------------------------------------------
# UTILITY FUNCTIONS
# ---------------------------------------------------------------------------

def print_step(msg: str):
    """Formatted print for thesis pipeline logging."""
    print(f"  [Step 2] {msg}")


def make_dirs():
    for d in [EDGES_DIR, HOUGH_DIR, SEGMENTS_DIR, DEBUG_DIR]:
        d.mkdir(parents=True, exist_ok=True)
    print_step("Output directories ready")


def find_preprocessed_file(map_num: str, kind: str) -> Path | None:
    """Resolve Step 1 output path for either numeric or Arabic-based naming."""
    direct = PREPROCESSED_DIR / f"map_{map_num}_{kind}.png"
    if direct.exists():
        return direct

    arabic = PREPROCESSED_DIR / f"map_الخريطة رقم {map_num}_{kind}.png"
    if arabic.exists():
        return arabic

    # Fallback: tolerate any prefix/suffix as long as the filename contains
    # the map number and the requested output kind.
    candidates = []
    for p in PREPROCESSED_DIR.glob(f"*.png"):
        name = p.name
        if kind.lower() not in name.lower():
            continue
        if map_num in name:
            candidates.append(p)

    if not candidates:
        return None

    # Prefer the shortest match to avoid accidental collisions.
    return sorted(candidates, key=lambda p: (len(p.name), p.name))[0]


def read_gray_unicode(path: Path) -> np.ndarray:
    """Read grayscale image with Windows Unicode filename support."""
    data = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)


def load_preprocessed(map_num: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Load both outputs from Step 1 for a given map number.
    Returns (gray_clean, binary) as uint8 grayscale arrays.

    We use the CLEAN grayscale (not binary) for edge detection because:
      • Canny works on gradient magnitude — it needs gradual intensity
        transitions, which are present in grayscale but lost in binary
      • The binary image is used for contour segmentation where we need
        crisp closed-region boundaries
    """
    clean_path = find_preprocessed_file(map_num, "clean")
    binary_path = find_preprocessed_file(map_num, "binary")

    if clean_path is None:
        raise FileNotFoundError(
            f"Preprocessed image not found for map {map_num} in {PREPROCESSED_DIR}\n"
            f"Run step1_preprocessing.py first."
        )

    gray = read_gray_unicode(clean_path)
    binary = read_gray_unicode(binary_path) if binary_path is not None else None

    if gray is None:
        raise IOError(f"Could not read: {clean_path}")
    if binary is None:
        print_step(f"WARNING: binary image not found for map {map_num} — "
                   f"will compute from grayscale")
        _, binary = cv2.threshold(gray, 0, 255,
                                   cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    h, w = gray.shape
    print_step(f"Loaded map {map_num} — shape: ({h}, {w})")
    return gray, binary


# ---------------------------------------------------------------------------
# STAGE A: CANNY EDGE DETECTION
# ---------------------------------------------------------------------------

def detect_edges_canny(gray: np.ndarray,
                        map_num: str) -> np.ndarray:
    """
    Apply the Canny edge detector to the cleaned grayscale cadastral image.

    How Canny works (3 internal stages):
    ─────────────────────────────────────
    1. Gaussian smoothing: suppress high-frequency noise that would create
       false edge responses.  The Canny function uses a 5×5 Gaussian
       internally when apertureSize=3.

    2. Gradient computation: apply Sobel filters in X and Y directions,
       compute gradient magnitude G = √(Gx²+Gy²) and direction θ.
       With L2gradient=True this uses the exact formula rather than
       |Gx|+|Gy| approximation — more accurate for thin cadastral lines.

    3. Non-maximum suppression: thin the gradient ridges to 1-pixel-wide
       edges by keeping only local maxima along each gradient direction.

    4. Double threshold + hysteresis:
       • Pixels with G > threshold2 → strong edge (kept)
       • Pixels with G < threshold1 → non-edge (discarded)
       • Pixels between → weak edge, kept ONLY if connected to a strong edge
       This is why threshold2/threshold1 ≈ 3:1 is recommended — it links
       faint-but-real boundary continuations to strong anchor pixels.

    Parameter choice for cadastral maps:
    ─────────────────────────────────────
    threshold1=30 : low, because the maps have faint/aged ink lines that
                    produce weak gradients — we want to catch them as weak
                    edges so hysteresis can link them to stronger segments.
    threshold2=90 : 3× ratio.  Keeps only pixels with clearly higher gradient
                    than the paper texture as definite edges.
    apertureSize=3 : standard Sobel kernel — matches the typical 2–4 pixel
                     width of cadastral boundary lines.

    Returns an 8-bit binary edge map: 255 = edge, 0 = non-edge.
    """
    edges = cv2.Canny(
        gray,
        threshold1=CANNY_THRESH1,
        threshold2=CANNY_THRESH2,
        apertureSize=CANNY_APERTURE,
        L2gradient=CANNY_L2GRADIENT
    )

    n_edge_px  = int(np.sum(edges > 0))
    total_px   = edges.size
    edge_ratio = 100 * n_edge_px / total_px

    print_step(f"Canny edges — map {map_num}: "
               f"{n_edge_px:,} edge pixels ({edge_ratio:.2f}% of image)")
    return edges


# ---------------------------------------------------------------------------
# STAGE B: EDGE CLEANING & DILATION
# ---------------------------------------------------------------------------

def clean_edges(edges: np.ndarray) -> np.ndarray:
    """
    Post-process the raw Canny edge map to improve edge connectivity.

    The problem:
    ─────────────
    Cadastral boundary lines are typically 1–4 pixels wide after scanning.
    Faded ink, fold creases, and paper texture can break a single continuous
    boundary line into multiple short segments separated by 1–3 pixel gaps.
    If these gaps are not closed:
      • Hough transform will detect many short segments instead of one
        long boundary line — noisy and harder to work with
      • Contour-based segmentation will produce open contours instead of
        closed polygons — parcels won't be detected as closed regions

    The fix:
    ─────────
    1. Morphological dilation with a 3×3 elliptical kernel:
       Each edge pixel "grows" by 1 pixel in all directions, closing
       1–2 pixel gaps between adjacent edge segments.

    2. Small connected-component removal:
       Any connected region of edge pixels smaller than 8 pixels is
       almost certainly scanner noise or paper texture — remove it.
       Real cadastral lines span dozens to hundreds of pixels.

    Trade-off:
    ──────────
    Dilation can merge two parallel boundary lines that are only 2–3 pixels
    apart.  For these maps this is rare because parcel boundaries are
    typically spaced much further apart.  We use only 1 dilation iteration
    to be conservative.
    """
    # Dilation
    kernel  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                         EDGE_DILATE_KERNEL)
    dilated = cv2.dilate(edges, kernel, iterations=EDGE_DILATE_ITER)

    # Remove tiny connected components (noise)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        dilated, connectivity=8
    )
    # Keep components with area ≥ 8 pixels (label 0 is background)
    clean = np.zeros_like(dilated)
    for label in range(1, num_labels):
        if stats[label, cv2.CC_STAT_AREA] >= 8:
            clean[labels == label] = 255

    removed = int(np.sum(dilated > 0)) - int(np.sum(clean > 0))
    print_step(f"Edge cleaning: removed {removed:,} noise pixels, "
               f"{int(np.sum(clean > 0)):,} edge pixels remain")
    return clean


# ---------------------------------------------------------------------------
# STAGE C: PROBABILISTIC HOUGH LINE TRANSFORM
# ---------------------------------------------------------------------------

def detect_hough_lines(edges_clean: np.ndarray,
                        gray: np.ndarray,
                        map_num: str) -> list[tuple]:
    """
    Apply the Probabilistic Hough Line Transform (HoughLinesP) to detect
    straight line segments in the cleaned edge map.

    Standard Hough vs Probabilistic Hough:
    ────────────────────────────────────────
    Standard HoughLines: votes in (ρ, θ) accumulator for every edge pixel,
    returns infinite lines (ρ, θ) with no start/end points.

    HoughLinesP (probabilistic): randomly samples edge pixels to vote,
    returns line SEGMENTS (x1,y1,x2,y2) with actual endpoints.

    For cadastral maps we use HoughLinesP because:
      ✓ We need endpoints to measure line length and position
      ✓ Parcel boundaries are finite segments, not infinite lines
      ✓ It is significantly faster on large map images
      ✓ maxLineGap parameter explicitly handles broken ink lines

    Parameter strategy:
    ───────────────────
    threshold=80: a line must accumulate at least 80 votes to be detected.
                  This filters out short random alignments in paper texture
                  while retaining true cadastral boundaries.

    minLineLength: set dynamically to 1% of the image's shorter dimension.
                   For a 3000×4000 image this gives 30px — reasonable lower
                   bound for a real parcel boundary vs a text stroke.

    maxLineGap=15: gaps up to 15 pixels between collinear points are bridged.
                   This handles the most common crease-induced breaks in
                   boundary lines.

    Returns a list of (x1, y1, x2, y2) tuples.
    """
    h, w = edges_clean.shape
    min_line_len = max(HOUGH_MIN_LINE_LEN, int(min(h, w) * 0.01))

    raw = cv2.HoughLinesP(
        edges_clean,
        rho=HOUGH_RHO,
        theta=HOUGH_THETA,
        threshold=HOUGH_THRESHOLD,
        minLineLength=min_line_len,
        maxLineGap=HOUGH_MAX_LINE_GAP
    )

    if raw is None:
        print_step(f"Hough lines — map {map_num}: NO lines detected")
        return []

    lines = [tuple(r[0]) for r in raw]   # list of (x1,y1,x2,y2)

    lengths = [np.hypot(x2 - x1, y2 - y1) for x1, y1, x2, y2 in lines]
    avg_len = float(np.mean(lengths)) if lengths else 0.0

    print_step(f"Hough lines — map {map_num}: {len(lines)} lines detected, "
               f"avg length = {avg_len:.1f}px, "
               f"minLineLength used = {min_line_len}px")
    return lines


# ---------------------------------------------------------------------------
# STAGE D: LINE CLASSIFICATION
# ---------------------------------------------------------------------------

def classify_lines(lines: list[tuple]) -> dict[str, list]:
    """
    Group detected line segments by their dominant orientation.

    Categories:
    ────────────
    horizontal  : angle within ±HORIZONTAL_TOL degrees of 0° (or 180°)
    vertical    : angle within ±VERTICAL_TOL degrees of 90°
    diagonal    : everything else

    Why classify?
    ─────────────
    Cadastral maps in Lebanon (and most of the Middle East) follow a
    roughly north-aligned grid.  Most parcel boundaries are either
    horizontal (east-west) or vertical (north-south), with some diagonal
    boundaries following terrain features.

    Classification allows:
      • Generating separate visualisations by orientation (thesis figures)
      • Prioritising horizontal/vertical lines for perspective correction
        in Step 1 (skew detection used these implicitly)
      • Statistical analysis: if most lines are diagonal, the map may
        still be misaligned and need additional correction

    Angle computation:
    ──────────────────
    arctan2(dy, dx) gives the line angle in (-180°, 180°).
    We normalise to (0°, 180°) because line orientation is ambiguous
    (a horizontal line is the same whether it points left or right).
    """
    classified = {"horizontal": [], "vertical": [], "diagonal": []}

    for x1, y1, x2, y2 in lines:
        dx, dy = x2 - x1, y2 - y1
        # angle in degrees, normalised to [0, 180)
        angle = (np.degrees(np.arctan2(abs(dy), abs(dx)))) % 180

        if angle <= HORIZONTAL_TOL or angle >= (180 - HORIZONTAL_TOL):
            classified["horizontal"].append((x1, y1, x2, y2))
        elif abs(angle - 90) <= VERTICAL_TOL:
            classified["vertical"].append((x1, y1, x2, y2))
        else:
            classified["diagonal"].append((x1, y1, x2, y2))

    h = len(classified["horizontal"])
    v = len(classified["vertical"])
    d = len(classified["diagonal"])
    total = h + v + d or 1
    print_step(f"Line classification: horizontal={h} ({100*h/total:.0f}%), "
               f"vertical={v} ({100*v/total:.0f}%), "
               f"diagonal={d} ({100*d/total:.0f}%)")
    return classified


# ---------------------------------------------------------------------------
# STAGE E: CONTOUR-BASED PARCEL SEGMENTATION
# ---------------------------------------------------------------------------

def segment_parcels(binary: np.ndarray,
                    map_num: str) -> tuple[np.ndarray, list[dict]]:
    """
    Detect individual land parcels as closed contour regions.

    Strategy:
    ──────────
    We use the BINARY image (black lines on white background from Step 1)
    rather than the Canny edge map, because:
      • The binary image has closed parcel outlines (the full drawn boundary)
      • The Canny edge map has open fragments (only the gradient transitions)
      • findContours works best on solid closed boundaries

    Approach:
    ──────────
    1. Invert the binary image:
       Our binary has black lines (0) on white background (255).
       findContours in RETR_CCOMP mode looks for white blobs —
       so we invert to get white parcels on black lines.

    2. cv2.findContours with RETR_CCOMP:
       Returns a two-level contour hierarchy:
         Level 0: outer boundaries of white regions (= parcel interiors)
         Level 1: holes inside white regions (= sub-parcels or text cutouts)
       We keep only Level 0 contours with area within our size range.

    3. CHAIN_APPROX_SIMPLE:
       Compresses horizontal, vertical and diagonal segments to just their
       endpoints.  A rectangle is stored as 4 points rather than all the
       perimeter pixels.  Saves memory and speeds up area calculation.

    4. Area filtering:
       MIN_PARCEL_AREA removes text strokes, scan dust, small holes.
       MAX_PARCEL_AREA_FRAC removes the overall map border contour which
       would otherwise be detected as the "biggest parcel".

    Returns:
    ─────────
    segment_map : H×W uint8 image with each parcel filled a unique colour
    parcel_list : list of dicts with keys:
                  id, area_px, cx, cy, bbox (x,y,w,h), contour
    """
    h, w = binary.shape
    max_area = MAX_PARCEL_AREA_FRAC * h * w

    # Invert: parcel interiors become white blobs
    inverted = cv2.bitwise_not(binary)

    # Find contours
    contours, hierarchy = cv2.findContours(
        inverted,
        cv2.RETR_CCOMP,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if contours is None or len(contours) == 0:
        print_step(f"Segmentation — map {map_num}: no contours found")
        return np.zeros((h, w, 3), dtype=np.uint8), []

    parcels = []
    parcel_id = 0

    for i, contour in enumerate(contours):
        # Only take top-level contours (not holes)
        if hierarchy[0][i][3] != -1:  # has a parent → skip (it's a hole)
            continue

        area = cv2.contourArea(contour)
        if area < MIN_PARCEL_AREA or area > max_area:
            continue

        # Moments for centroid
        M = cv2.moments(contour)
        if M["m00"] == 0:
            continue
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])

        # Bounding box
        x, y, bw, bh = cv2.boundingRect(contour)

        parcels.append({
            "id":      parcel_id,
            "area_px": float(area),
            "cx":      cx,
            "cy":      cy,
            "bbox":    (x, y, bw, bh),
            "contour": contour,
        })
        parcel_id += 1

    # Build colour-coded segment map
    segment_map = np.zeros((h, w, 3), dtype=np.uint8)
    rng = np.random.default_rng(seed=42)  # fixed seed → reproducible colours
    for p in parcels:
        colour = tuple(int(c) for c in rng.integers(80, 230, size=3))
        cv2.drawContours(segment_map, [p["contour"]], -1, colour, thickness=-1)

    areas = [p["area_px"] for p in parcels]
    print_step(f"Segmentation — map {map_num}: {len(parcels)} parcels found | "
               f"area range: {min(areas):.0f}–{max(areas):.0f} px² | "
               f"median: {float(np.median(areas)):.0f} px²")
    return segment_map, parcels


# ---------------------------------------------------------------------------
# STAGE F: PARCEL STATISTICS
# ---------------------------------------------------------------------------

def compute_parcel_stats(parcels: list[dict],
                          map_num: str,
                          image_shape: tuple) -> list[dict]:
    """
    Compute summary statistics for each detected parcel and add them
    to the parcel dict.  These are used for:
      • The CSV export (thesis appendix table)
      • Sanity checking: abnormally large/small parcels may indicate
        segmentation errors
      • Cross-referencing with the PDF index (Step 5) to match parcel IDs

    Additional statistics computed here:
    ─────────────────────────────────────
    perimeter_px   : contour perimeter in pixels
    aspect_ratio   : bounding box width / height
    solidity       : contour area / convex hull area
                     Solidity ≈ 1.0 for convex shapes (rectangular parcels)
                     Solidity < 0.8 suggests an L-shaped or irregular parcel
    compactness    : 4π·area / perimeter²  (=1 for a circle, <1 for irregular)
                     Useful to flag unusual shapes for manual review
    """
    h_img, w_img = image_shape

    for p in parcels:
        contour = p["contour"]

        p["perimeter_px"] = float(cv2.arcLength(contour, closed=True))

        # Convex hull for solidity
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        p["solidity"] = (p["area_px"] / hull_area) if hull_area > 0 else 0.0

        # Compactness (circularity)
        peri = p["perimeter_px"]
        p["compactness"] = (
            (4 * np.pi * p["area_px"]) / (peri * peri)
        ) if peri > 0 else 0.0

        # Aspect ratio from bounding box
        _, _, bw, bh = p["bbox"]
        p["aspect_ratio"] = (bw / bh) if bh > 0 else 0.0

        # Normalised area (fraction of image)
        p["area_frac"] = p["area_px"] / (h_img * w_img)

    # Print summary
    solidities  = [p["solidity"]  for p in parcels]
    compacts    = [p["compactness"] for p in parcels]
    print_step(f"Parcel stats — map {map_num}: "
               f"avg solidity={np.mean(solidities):.2f}, "
               f"avg compactness={np.mean(compacts):.2f}")
    return parcels


def save_parcel_csv(parcels: list[dict], map_num: str):
    """
    Write per-parcel statistics to CSV for thesis appendix / analysis.
    Columns: id, area_px, area_frac, cx, cy, bbox_x, bbox_y, bbox_w,
             bbox_h, aspect_ratio, perimeter_px, solidity, compactness
    """
    csv_path = SEGMENTS_DIR / f"map_{map_num}_parcel_stats.csv"
    fieldnames = ["id", "area_px", "area_frac", "cx", "cy",
                  "bbox_x", "bbox_y", "bbox_w", "bbox_h",
                  "aspect_ratio", "perimeter_px", "solidity", "compactness"]

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for p in parcels:
            x, y, bw, bh = p["bbox"]
            writer.writerow({
                "id":           p["id"],
                "area_px":      round(p["area_px"], 1),
                "area_frac":    round(p["area_frac"], 6),
                "cx":           p["cx"],
                "cy":           p["cy"],
                "bbox_x":       x,
                "bbox_y":       y,
                "bbox_w":       bw,
                "bbox_h":       bh,
                "aspect_ratio": round(p["aspect_ratio"], 3),
                "perimeter_px": round(p["perimeter_px"], 1),
                "solidity":     round(p["solidity"], 3),
                "compactness":  round(p["compactness"], 3),
            })
    print_step(f"CSV saved: {csv_path.name}  ({len(parcels)} rows)")


# ---------------------------------------------------------------------------
# STAGE G: VISUALISATION & SAVE
# ---------------------------------------------------------------------------

def draw_hough_lines(gray: np.ndarray,
                      lines: list[tuple],
                      classified: dict[str, list]) -> tuple[np.ndarray,
                                                            np.ndarray]:
    """
    Produce two visualisation images for the Hough line results:

    1. lines_img: all detected lines drawn in red over a BGR version
       of the grayscale map — shows raw detection density.

    2. classified_img: lines colour-coded by orientation:
         horizontal → blue  (most common in these north-aligned maps)
         vertical   → green
         diagonal   → red / orange

    These are intended as thesis Figure 4.x (edge/Hough results).
    """
    bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    # Image 1: all lines in red
    lines_img = bgr.copy()
    for x1, y1, x2, y2 in lines:
        cv2.line(lines_img, (x1, y1), (x2, y2), (0, 0, 220), 1,
                 cv2.LINE_AA)

    # Image 2: classified colours
    classified_img = bgr.copy()
    color_map = {
        "horizontal": (220, 80,  30),   # blue
        "vertical":   (30,  180, 60),   # green
        "diagonal":   (30,  60,  220),  # red
    }
    for category, colour in color_map.items():
        for x1, y1, x2, y2 in classified[category]:
            cv2.line(classified_img, (x1, y1), (x2, y2), colour, 1,
                     cv2.LINE_AA)

    return lines_img, classified_img


def draw_parcel_bboxes(gray: np.ndarray,
                        parcels: list[dict]) -> np.ndarray:
    """
    Draw bounding boxes and centroid markers for each detected parcel,
    and label each with its assigned ID.  Produces a clear overview of
    the segmentation result for thesis figures.
    """
    vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    for p in parcels:
        x, y, bw, bh = p["bbox"]
        # Bounding box in teal
        cv2.rectangle(vis, (x, y), (x + bw, y + bh), (180, 140, 0), 1)
        # Centroid dot in red
        cv2.circle(vis, (p["cx"], p["cy"]), 3, (0, 60, 220), -1)
        # ID label (small, near centroid)
        cv2.putText(vis, str(p["id"]),
                    (p["cx"] + 4, p["cy"] - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 60, 220), 1,
                    cv2.LINE_AA)
    return vis


def save_outputs(map_num: str,
                 gray: np.ndarray,
                 edges_raw: np.ndarray,
                 edges_clean: np.ndarray,
                 lines_img: np.ndarray,
                 classified_img: np.ndarray,
                 segment_map: np.ndarray,
                 parcel_bbox_img: np.ndarray):
    """Save all visualisation outputs for map_num."""
    saves = [
        (EDGES_DIR    / f"map_{map_num}_edges.png",            edges_raw),
        (EDGES_DIR    / f"map_{map_num}_edges_clean.png",      edges_clean),
        (HOUGH_DIR    / f"map_{map_num}_lines.png",            lines_img),
        (HOUGH_DIR    / f"map_{map_num}_lines_classified.png", classified_img),
        (SEGMENTS_DIR / f"map_{map_num}_segments.png",         segment_map),
        (SEGMENTS_DIR / f"map_{map_num}_parcels.png",          parcel_bbox_img),
    ]

    for path, img in saves:
        cv2.imwrite(str(path), img)
        print_step(f"Saved: {path.name}")

    # Debug: 3-panel composite (edges | Hough | segments) for quick review
    h_target = 400
    def rz(img, h):
        scale = h / img.shape[0]
        return cv2.resize(img, (int(img.shape[1] * scale), h))

    e_s  = rz(cv2.cvtColor(edges_clean, cv2.COLOR_GRAY2BGR), h_target)
    l_s  = rz(lines_img,   h_target)
    sg_s = rz(segment_map, h_target)

    min_w = min(e_s.shape[1], l_s.shape[1], sg_s.shape[1])
    divider = np.ones((h_target, 4, 3), dtype=np.uint8) * 150
    debug = np.hstack([e_s[:, :min_w], divider,
                       l_s[:, :min_w], divider,
                       sg_s[:, :min_w]])
    debug_path = DEBUG_DIR / f"map_{map_num}_debug.png"
    cv2.imwrite(str(debug_path), debug)
    print_step(f"Saved (debug): {debug_path.name}")


# ---------------------------------------------------------------------------
# MAIN FUNCTION — process one map end-to-end
# ---------------------------------------------------------------------------

def process_map(map_num: str) -> dict:
    """
    Run the full Stage A–G pipeline on one map.
    Returns a summary dict for the final report.
    """
    print(f"\n{'='*60}")
    print(f"  Processing map {map_num}")
    print(f"{'='*60}")
    t0 = time.time()

    # Load Step 1 output
    gray, binary = load_preprocessed(map_num)

    # A — Canny edges
    edges_raw = detect_edges_canny(gray, map_num)

    # B — Edge cleaning
    edges_clean = clean_edges(edges_raw)

    # C — Hough lines
    lines = detect_hough_lines(edges_clean, gray, map_num)

    # D — Line classification
    classified = classify_lines(lines)

    # E — Parcel segmentation
    segment_map, parcels = segment_parcels(binary, map_num)

    # F — Parcel statistics
    if parcels:
        parcels = compute_parcel_stats(parcels, map_num, gray.shape)
        save_parcel_csv(parcels, map_num)

    # G — Visualisation
    if lines:
        lines_img, classified_img = draw_hough_lines(gray, lines, classified)
    else:
        bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        lines_img = classified_img = bgr.copy()

    parcel_bbox_img = draw_parcel_bboxes(gray, parcels)

    save_outputs(map_num, gray, edges_raw, edges_clean,
                 lines_img, classified_img, segment_map, parcel_bbox_img)

    elapsed = time.time() - t0
    print_step(f"Map {map_num} done in {elapsed:.1f}s")

    return {
        "map":        map_num,
        "edges_px":   int(np.sum(edges_clean > 0)),
        "n_lines":    len(lines),
        "n_h":        len(classified["horizontal"]),
        "n_v":        len(classified["vertical"]),
        "n_d":        len(classified["diagonal"]),
        "n_parcels":  len(parcels),
        "elapsed":    elapsed,
    }


# ---------------------------------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------------------------------

def run():
    """
    Process all 13 maps in stitching order and print a summary table.
    Called by main.py or directly: python step2_edges.py
    """
    print("\n" + "="*60)
    print("  STEP 2 — EDGE DETECTION, HOUGH TRANSFORM & SEGMENTATION")
    print("  Cadastral Panoramic Reconstruction — Masters Thesis")
    print("="*60)

    make_dirs()

    all_results = []
    missing     = []

    for num in MAP_NUMBERS:
        clean_path = find_preprocessed_file(num, "clean")
        if clean_path is None:
            print(f"\n  [Step 2] WARNING: preprocessed image missing "
                  f"for map {num} — skipping")
            missing.append(num)
            continue
        result = process_map(num)
        all_results.append(result)

    # ── Summary ─────────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("  STEP 2 SUMMARY")
    print("="*60)
    print(f"  Maps processed : {len(all_results)}")
    if missing:
        print(f"  Maps skipped   : {', '.join(missing)}")

    if all_results:
        print(f"\n  {'Map':>4} | {'Edges(px)':>10} | {'Lines':>6} | "
              f"{'H':>5} | {'V':>5} | {'D':>5} | {'Parcels':>8} | {'Time':>6}")
        print("  " + "-"*65)
        for r in all_results:
            print(f"  {r['map']:>4} | "
                  f"{r['edges_px']:>10,} | "
                  f"{r['n_lines']:>6} | "
                  f"{r['n_h']:>5} | "
                  f"{r['n_v']:>5} | "
                  f"{r['n_d']:>5} | "
                  f"{r['n_parcels']:>8} | "
                  f"{r['elapsed']:>5.1f}s")

        total_parcels = sum(r["n_parcels"] for r in all_results)
        total_lines   = sum(r["n_lines"]   for r in all_results)
        total_time    = sum(r["elapsed"]   for r in all_results)
        print(f"\n  Total lines detected  : {total_lines:,}")
        print(f"  Total parcels found   : {total_parcels:,}")
        print(f"  Total processing time : {total_time:.1f}s")
        print(f"\n  Outputs:")
        print(f"    Edge maps    → {EDGES_DIR.resolve()}")
        print(f"    Hough lines  → {HOUGH_DIR.resolve()}")
        print(f"    Segments     → {SEGMENTS_DIR.resolve()}")

    unexpected_missing = [m for m in missing if m not in EXPECTED_MISSING_MAPS]
    success = len(unexpected_missing) == 0 and len(all_results) > 0
    status  = "SUCCESS" if success else "PARTIAL"
    if missing:
        print(f"\n  Expected missing maps : {', '.join(sorted(EXPECTED_MISSING_MAPS))}")
    if unexpected_missing:
        print(f"  Unexpected missing    : {', '.join(unexpected_missing)}")
    print(f"\n  [Step 2] Status: {status}")
    print("="*60 + "\n")
    return success


if __name__ == "__main__":
    ok = run()
    sys.exit(0 if ok else 1)
