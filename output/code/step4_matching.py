"""
=============================================================================
STEP 4: TWO-STAGE BORDER-PROFILE & PARCEL-CENTROID REGISTRATION
=============================================================================
Masters Thesis — AI-Based Panoramic Cadastral Image Reconstruction
Author  : Hussein Chalhoub

Why the previous SIFT-based approach failed
--------------------------------------------
SIFT was designed for panoramic PHOTOGRAPHY where two photos share a 20-50%
visual overlap.  Cadastral map sheets work completely differently: each sheet
shows a unique geographic area with NO visual overlap with its neighbours.
Map 45 and map 47 share only a BOUNDARY LINE, not any repeating visual
content.  SIFT finds ~27 ratio-test matches out of 5000 keypoints (0.5%),
which is pure noise — the same as random.

This is actually a thesis finding: standard SIFT stitching fails on
cadastral atlas sheets and a domain-specific approach is required.

The correct approach for cadastral atlas registration
------------------------------------------------------
Adjacent map sheets in a cadastral atlas share this property:
  EVERY parcel boundary line that crosses the right edge of sheet A
  continues immediately from the left edge of sheet B.

The boundary line crossing POSITIONS (y-coordinates along the edge) are
therefore the same on both sheets.  We use these as the registration signal.

Two-stage pipeline
-------------------
STAGE 1 — Border-line profile alignment (runs always)
  For each map, detect where parcel boundary lines cross each of its 4 edges.
  This gives a "crossing profile" — a sparse set of positions along the edge.
  For each pair (A, B):
    - Try all 4 rotations of B (0/90/180/270 degrees)
    - For each rotation, try all 4 edge combinations (right-left, top-bottom, etc.)
    - Cross-correlate A's edge profile with rotated-B's edge profile using a 1D
      histogram cross-correlation — this finds both the best alignment AND the
      y-shift (the vertical offset between the two map sheets)
    - The (rotation, edge pair, shift) with the highest correlation score is used
      to build the final 3x3 homography H

STAGE 2 — Parcel centroid refinement (runs after Step 5)
  OCR in Step 5 detects parcel numbers and their centroids.
  Boundary parcels (e.g. parcel 2580 between maps 45 and 47) appear on BOTH
  map sheets.  Their centroids are the same real-world location on both maps.
  We use these as ground control point pairs to refine the Stage 1 homography.
  Requires output/detected/map_<N>_parcels.json from Step 5.

Output files
-------------
  output/matches/map_<A>_<B>_registration.png  - blue-green overlay preview
  output/homographies/H_<A>_<B>.npy            - 3x3 homography matrix
  output/homographies/homographies.json        - all matrices + metadata
  output/homographies/homography_quality.csv   - per-pair results table

Step 6 usage
-------------
  For each pair, homographies.json stores:
    rotation_b_deg : rotation applied to B before placing next to A
    H              : 3x3 matrix mapping B_original coords -> A frame coords
  Step 6 calls cv2.warpPerspective(img_B, H, canvas_size) directly.

=============================================================================
"""

import sys
import time
import json
import csv
import cv2
import numpy as np
from pathlib import Path


# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------

PREPROCESSED_DIR = Path("output/preprocessed")
DETECTED_DIR     = Path("output/detected")
MATCHES_DIR      = Path("output/matches")
HOMOGRAPHY_DIR   = Path("output/homographies")
DEBUG_DIR        = Path("output/debug/step4")

# ── Adjacency table (from PDF index boundary-parcel column) ───────────────
# Each entry: (map_A, map_B, [boundary_parcel_ids])
ADJACENCY_PAIRS = [
    ("45", "47", [2580]),
    ("45", "46", [2616, 2619]),
    ("47", "48", [2749, 2803, 2814]),
    ("48", "49", [2893]),
    ("49", "50", [3022, 3054]),
    ("50", "51", [3068, 3813, 3866]),
    ("52", "53", [3215]),
    ("52", "54", [3216]),
    ("52", "55", [3217]),
    ("54", "55", [3338, 3339, 3345, 3346]),
]

MAPS_IN_USE = sorted(
    {m for p in ADJACENCY_PAIRS for m in (p[0], p[1])}, key=int
)

# ── Stage 1: Hough line detection parameters ──────────────────────────────
# More permissive than Step 2 — we want to catch short lines near edges
HOUGH_THRESHOLD    = 50     # minimum votes (lower = more lines detected)
HOUGH_MIN_LINE_LEN = 30     # px — shorter than Step 2 to catch boundary stubs
HOUGH_MAX_LINE_GAP = 25     # px — bridge gaps in faded boundary lines

# ── Stage 1: Profile matching parameters ──────────────────────────────────
PROFILE_BINS    = 200   # histogram resolution for cross-correlation
PROFILE_MARGIN  = 50    # px — how far from the edge to accept a line crossing
EDGE_EXCLUDE_PX = 12    # px — ignore crossings this close to image corners
MIN_CROSSINGS   = 3     # minimum crossings to consider a profile reliable

# ── Stage 2: Parcel centroid refinement parameters ────────────────────────
MIN_CONTROL_PAIRS = 1   # minimum matched parcel centroids to run Stage 2
OCR_TOLERANCE     = 2   # fuzzy number matching (±2 on the parcel number)

ROTATIONS = [0, 90, 180, 270]


# ---------------------------------------------------------------------------
# UTILITIES
# ---------------------------------------------------------------------------

def print_step(msg: str):
    print(f"  [Step 4] {msg}")


def make_dirs():
    for d in [MATCHES_DIR, HOMOGRAPHY_DIR, DEBUG_DIR]:
        d.mkdir(parents=True, exist_ok=True)
    print_step("Output directories ready")


def read_image(path: Path, grayscale: bool = True) -> np.ndarray:
    """Windows-safe reader for Arabic-named files."""
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


def load_map_images(map_num: str) -> tuple[np.ndarray, np.ndarray]:
    """Load (grayscale_clean, binary) from Step 1 output."""
    gray   = read_image(PREPROCESSED_DIR / f"map_{map_num}_clean.png",  grayscale=True)
    binary = read_image(PREPROCESSED_DIR / f"map_{map_num}_binary.png", grayscale=True)
    return gray, binary


def rotate_image(img: np.ndarray, angle: int) -> np.ndarray:
    """Rotate image by 0/90/180/270 degrees counter-clockwise."""
    if angle == 0:   return img
    if angle == 90:  return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    if angle == 180: return cv2.rotate(img, cv2.ROTATE_180)
    if angle == 270: return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    raise ValueError(f"Invalid angle: {angle}")


# ---------------------------------------------------------------------------
# STAGE 1A: LINE DETECTION FOR PROFILE
# ---------------------------------------------------------------------------

def detect_lines(binary: np.ndarray) -> list[tuple]:
    """
    Detect line segments in the binary image using Probabilistic Hough.

    The binary image from Step 1 has:
      ink = 0 (black), paper = 255 (white)

    HoughLinesP needs non-zero pixels as "edge" pixels, so we invert first.
    Parameters are more permissive than Step 2 because we specifically want
    to catch short line stubs that terminate AT the map edge — these are the
    boundary crossings we use as registration landmarks.
    """
    H, W = binary.shape
    min_len = max(HOUGH_MIN_LINE_LEN, int(min(H, W) * 0.005))

    # Invert: ink becomes white (255) for Hough detection
    inv = cv2.bitwise_not(binary)

    raw = cv2.HoughLinesP(
        inv,
        rho=1,
        theta=np.pi / 180,
        threshold=HOUGH_THRESHOLD,
        minLineLength=min_len,
        maxLineGap=HOUGH_MAX_LINE_GAP
    )
    if raw is None:
        return []
    return [tuple(r[0]) for r in raw]


# ---------------------------------------------------------------------------
# STAGE 1B: EDGE CROSSING PROFILE
# ---------------------------------------------------------------------------

def compute_edge_profile(lines: list[tuple],
                          shape: tuple[int, int],
                          edge: str) -> list[float]:
    """
    Find where line segments (extended to the edge if needed) cross the
    given map edge.

    For left / right edges:  returns y-coordinates of crossings (sorted).
    For top  / bottom edges: returns x-coordinates of crossings (sorted).

    Key filtering rules:
      1. Lines nearly parallel to the edge are excluded — they are the outer
         map border frame, not parcel boundaries.
      2. Lines whose closest point to the edge is more than PROFILE_MARGIN px
         away are excluded — they don't reach the edge.
      3. Crossings within EDGE_EXCLUDE_PX of the image corners are excluded —
         these are border-frame intersections, not parcel crossings.

    The PROFILE_MARGIN makes this robust to ink lines that fade out slightly
    before reaching the sheet edge (common in scanned cadastral maps).
    """
    H, W = shape
    crossings = []

    if edge in ('left', 'right'):
        x_edge = 0 if edge == 'left' else W - 1

        for x1, y1, x2, y2 in lines:
            dx = x2 - x1
            # Skip lines nearly parallel to left/right edge (nearly vertical)
            if abs(dx) < 5:
                continue
            # Skip lines that ARE the border frame
            if abs(x1 - x_edge) < 3 and abs(x2 - x_edge) < 3:
                continue
            # Reject lines too far from the edge
            x_min_seg = min(x1, x2) - PROFILE_MARGIN
            x_max_seg = max(x1, x2) + PROFILE_MARGIN
            if not (x_min_seg <= x_edge <= x_max_seg):
                continue
            # Extrapolate line to x = x_edge
            t = (x_edge - x1) / dx
            y_cross = y1 + t * (y2 - y1)
            # Accept only within image bounds (excluding corners)
            if EDGE_EXCLUDE_PX < y_cross < H - EDGE_EXCLUDE_PX:
                crossings.append(float(y_cross))

    elif edge in ('top', 'bottom'):
        y_edge = 0 if edge == 'top' else H - 1

        for x1, y1, x2, y2 in lines:
            dy = y2 - y1
            if abs(dy) < 5:
                continue
            if abs(y1 - y_edge) < 3 and abs(y2 - y_edge) < 3:
                continue
            y_min_seg = min(y1, y2) - PROFILE_MARGIN
            y_max_seg = max(y1, y2) + PROFILE_MARGIN
            if not (y_min_seg <= y_edge <= y_max_seg):
                continue
            t = (y_edge - y1) / dy
            x_cross = x1 + t * (x2 - x1)
            if EDGE_EXCLUDE_PX < x_cross < W - EDGE_EXCLUDE_PX:
                crossings.append(float(x_cross))

    return sorted(crossings)


def compute_all_profiles(lines: list[tuple],
                          shape: tuple[int, int]) -> dict[str, list[float]]:
    """Compute crossing profiles for all 4 edges."""
    return {e: compute_edge_profile(lines, shape, e)
            for e in ('left', 'right', 'top', 'bottom')}


# ---------------------------------------------------------------------------
# STAGE 1C: PROFILE CROSS-CORRELATION
# ---------------------------------------------------------------------------

def correlate_profiles(crossings_a: list[float],
                        crossings_b: list[float],
                        length_a: float,
                        length_b: float) -> tuple[float, float]:
    """
    Find the 1D shift that best aligns profile B with profile A.

    Method:
      Build normalized histograms for both profiles, then use 1D
      cross-correlation to find the shift and its confidence score.

    Why histograms, not raw point sets?
      With only 3-20 crossings per edge, point-set matching would be brittle.
      Histograms smooth the distribution so that nearby crossings still
      contribute to the correlation, making it robust to small OCR or line
      detection errors.

    The shift convention:
      y_A ≈ y_B_rotated + shift
      shift > 0 means B is shifted UP relative to A (B's crossings are at
      lower y-values, so we need to add shift to B to match A).

    Returns (shift_pixels, score).
      score is the raw cross-correlation peak; higher is better.
      A score of 0 means no meaningful alignment found.
    """
    if len(crossings_a) < MIN_CROSSINGS or len(crossings_b) < MIN_CROSSINGS:
        return 0.0, 0.0

    # Common normalization range — slightly larger than both profiles
    norm_len = max(length_a, length_b) * 1.1

    hist_a = np.zeros(PROFILE_BINS, dtype=np.float32)
    hist_b = np.zeros(PROFILE_BINS, dtype=np.float32)

    for c in crossings_a:
        idx = int(min(c / norm_len * PROFILE_BINS, PROFILE_BINS - 1))
        hist_a[idx] += 1.0

    for c in crossings_b:
        idx = int(min(c / norm_len * PROFILE_BINS, PROFILE_BINS - 1))
        hist_b[idx] += 1.0

    # Normalize by max so both histograms peak at 1.0
    if hist_a.max() > 0:
        hist_a /= hist_a.max()
    if hist_b.max() > 0:
        hist_b /= hist_b.max()

    # Full cross-correlation
    corr      = np.correlate(hist_a, hist_b, mode='full')
    best_idx  = int(np.argmax(corr))
    best_score = float(corr[best_idx])

    # Convert lag (bins) to pixels
    # lag_bins > 0: B's crossings are at lower positions → shift is positive
    lag_bins     = best_idx - (PROFILE_BINS - 1)
    shift_pixels = lag_bins * (norm_len / PROFILE_BINS)

    # Weight score by number of crossings (more crossings = more reliable)
    n_min = min(len(crossings_a), len(crossings_b))
    weighted_score = best_score * (n_min ** 0.5)  # sqrt prevents domination

    return float(shift_pixels), float(weighted_score)


# ---------------------------------------------------------------------------
# STAGE 1D: HOMOGRAPHY CONSTRUCTION FROM ALIGNMENT
# ---------------------------------------------------------------------------

def build_rotation_matrix(angle: int,
                            H_B: int,
                            W_B: int) -> tuple[np.ndarray, tuple[int, int]]:
    """
    Build the 3x3 homogeneous matrix R that maps B_original pixel coordinates
    to B_rotated pixel coordinates.

    Coordinate formulas (counter-clockwise rotation):
      0°  : (x,y) -> (x, y)                    shape (H_B, W_B)
      90° : (x,y) -> (y, W_B-1-x)              shape (W_B, H_B)
      180°: (x,y) -> (W_B-1-x, H_B-1-y)        shape (H_B, W_B)
      270°: (x,y) -> (H_B-1-y, x)              shape (W_B, H_B)

    Returns (R, (new_H, new_W)).
    """
    if angle == 0:
        return np.eye(3, dtype=np.float64), (H_B, W_B)

    if angle == 90:
        R = np.array([[0, 1, 0], [-1, 0, W_B - 1], [0, 0, 1]], dtype=np.float64)
        return R, (W_B, H_B)

    if angle == 180:
        R = np.array([[-1, 0, W_B - 1], [0, -1, H_B - 1], [0, 0, 1]], dtype=np.float64)
        return R, (H_B, W_B)

    if angle == 270:
        R = np.array([[0, -1, H_B - 1], [1, 0, 0], [0, 0, 1]], dtype=np.float64)
        return R, (W_B, H_B)

    raise ValueError(f"Invalid angle: {angle}")


def build_H_from_alignment(angle: int,
                             edge_a: str,
                             edge_b: str,
                             shift: float,
                             shape_a: tuple[int, int],
                             shape_b: tuple[int, int]) -> np.ndarray:
    """
    Build the full 3x3 homography H = T × R that maps B_original pixel
    coordinates directly to A frame pixel coordinates.

    R = rotation matrix (B_original → B_rotated, computed above)
    T = translation matrix (B_rotated → A frame, determined by edge placement)

    Placement conventions (based on which edges are adjacent):
      right-left  : B is placed to the RIGHT of A.
                    B_rotated's left edge (x_rot=0) aligns with A's x = W_A.
                    shift is the y-offset: y_A = y_rot + shift.
      left-right  : B is placed to the LEFT of A.
                    B_rotated's right edge aligns with A's x = -1.
      bottom-top  : B is placed BELOW A.
                    B_rotated's top edge (y_rot=0) aligns with A's y = H_A.
                    shift is the x-offset.
      top-bottom  : B is placed ABOVE A.

    H maps B_original coordinates:
      H × [x_B, y_B, 1]^T = [x_A, y_A, 1]^T

    Step 6 uses: cv2.warpPerspective(img_B_original, H, canvas_size)
    """
    H_A, W_A = shape_a
    H_B, W_B = shape_b

    R, (H_B_rot, W_B_rot) = build_rotation_matrix(angle, H_B, W_B)

    # Translation: where to place rotated-B in A's coordinate frame
    if edge_a == 'right' and edge_b == 'left':
        dx, dy = float(W_A), float(shift)
    elif edge_a == 'left' and edge_b == 'right':
        dx, dy = float(-W_B_rot), float(shift)
    elif edge_a == 'bottom' and edge_b == 'top':
        dx, dy = float(shift), float(H_A)
    elif edge_a == 'top' and edge_b == 'bottom':
        dx, dy = float(shift), float(-H_B_rot)
    else:
        raise ValueError(f"Incompatible edge pair: {edge_a} - {edge_b}")

    T = np.array([[1, 0, dx], [0, 1, dy], [0, 0, 1]], dtype=np.float64)

    # Combined transformation: H = T × R
    return T @ R


# ---------------------------------------------------------------------------
# STAGE 1 MAIN: FIND BEST ROTATION + PLACEMENT
# ---------------------------------------------------------------------------

OPPOSITE_EDGE = {
    'left': 'right', 'right': 'left',
    'top': 'bottom', 'bottom': 'top'
}


def stage1_alignment(map_a: str, map_b: str) -> dict:
    """
    Find the best rotation of B and the best edge-pair placement by
    cross-correlating parcel-line crossing profiles.

    For every (rotation × opposite-edge-pair) combination (4×4 = 16 total),
    we correlate A's edge profile with rotated-B's edge profile and record
    the shift and score.  The combination with the highest weighted score wins.

    Prints a table of all combinations for transparency.

    Returns a dict with keys:
        rotation, edge_a, edge_b, shift, score, H,
        shape_a, shape_b, n_crossings_a, n_crossings_b
    """
    gray_a, binary_a = load_map_images(map_a)
    gray_b, binary_b = load_map_images(map_b)
    shape_a = binary_a.shape
    shape_b = binary_b.shape

    print_step(f"    Map {map_a}: {shape_a},  Map {map_b}: {shape_b}")

    # Detect lines for map A (fixed reference — only done once)
    lines_a    = detect_lines(binary_a)
    profiles_a = compute_all_profiles(lines_a, shape_a)

    H_A, W_A = shape_a

    print_step(f"    Map {map_a} edge crossings: "
               f"L={len(profiles_a['left'])} "
               f"R={len(profiles_a['right'])} "
               f"T={len(profiles_a['top'])} "
               f"B={len(profiles_a['bottom'])}")

    best = {
        'score': -1.0, 'rotation': 0,
        'edge_a': 'right', 'edge_b': 'left',
        'shift': 0.0, 'H': None,
        'n_crossings_a': 0, 'n_crossings_b': 0,
    }

    for angle in ROTATIONS:
        # Physically rotate binary B for this candidate
        binary_b_rot    = rotate_image(binary_b, angle)
        shape_b_rot     = binary_b_rot.shape
        H_B_rot, W_B_rot = shape_b_rot

        lines_b_rot    = detect_lines(binary_b_rot)
        profiles_b_rot = compute_all_profiles(lines_b_rot, shape_b_rot)

        print_step(f"    rot={angle:>3}deg  crossings: "
                   f"L={len(profiles_b_rot['left'])} "
                   f"R={len(profiles_b_rot['right'])} "
                   f"T={len(profiles_b_rot['top'])} "
                   f"B={len(profiles_b_rot['bottom'])}")

        for edge_a, edge_b in [('right', 'left'), ('left', 'right'),
                                 ('bottom', 'top'), ('top', 'bottom')]:
            # Edge lengths for normalization
            len_a = float(H_A    if edge_a in ('left', 'right') else W_A)
            len_b = float(H_B_rot if edge_b in ('left', 'right') else W_B_rot)

            shift, score = correlate_profiles(
                profiles_a[edge_a],
                profiles_b_rot[edge_b],
                len_a,
                len_b
            )

            n_a = len(profiles_a[edge_a])
            n_b = len(profiles_b_rot[edge_b])

            print_step(f"      {edge_a:>6}->{edge_b:<6} | "
                       f"A={n_a:>3} B={n_b:>3} crossings | "
                       f"shift={shift:>8.1f}px | score={score:.3f}")

            if score > best['score']:
                try:
                    H = build_H_from_alignment(
                        angle, edge_a, edge_b, shift, shape_a, shape_b
                    )
                    best.update({
                        'score':         score,
                        'rotation':      angle,
                        'edge_a':        edge_a,
                        'edge_b':        edge_b,
                        'shift':         shift,
                        'H':             H,
                        'n_crossings_a': n_a,
                        'n_crossings_b': n_b,
                    })
                except ValueError:
                    pass

    best['shape_a'] = shape_a
    best['shape_b'] = shape_b

    print_step(f"    WINNER: rotation={best['rotation']}deg | "
               f"{best['edge_a']}->{best['edge_b']} | "
               f"shift={best['shift']:.1f}px | score={best['score']:.3f}")

    return best


# ---------------------------------------------------------------------------
# STAGE 2: PARCEL CENTROID REFINEMENT (runs after Step 5)
# ---------------------------------------------------------------------------

def load_parcels_json(map_num: str) -> list[dict] | None:
    """Load Step 5 parcel detection JSON. Returns None if Step 5 not run."""
    path = DETECTED_DIR / f"map_{map_num}_parcels.json"
    if not path.exists():
        return None
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data if isinstance(data, list) else data.get('parcels', [])


def find_parcel_centroid(parcels: list[dict],
                          target_id: int) -> tuple[float, float] | None:
    """
    Find the centroid of the parcel whose OCR-detected number is closest
    to target_id (within OCR_TOLERANCE).
    Returns (cx, cy) in image pixels, or None if not found.
    """
    for p in parcels:
        raw = str(p.get('ocr_text', p.get('detected_number',
                         p.get('number', ''))))
        try:
            if abs(int(raw.strip()) - target_id) <= OCR_TOLERANCE:
                return (float(p.get('cx', p.get('centroid_x', 0))),
                        float(p.get('cy', p.get('centroid_y', 0))))
        except (ValueError, TypeError):
            continue
    return None


def stage2_refinement(map_a: str,
                       map_b: str,
                       boundary_parcels: list[int],
                       H_rough: np.ndarray) -> tuple[np.ndarray, str]:
    """
    Refine H_rough using manually annotated parcel centroids as ground
    control points.

    A boundary parcel (e.g. 2580 between maps 45 and 47) has its centre
    marked on BOTH map sheets by the annotation tool.  These pixel positions
    are the same real-world location on both maps and therefore give us
    exact correspondence pairs.

    Transformation chosen based on number of control pairs available:
      1 pair  -> pure translation (most constrained, most robust)
      2-3 pairs -> affine transformation (rotation + scale + translation)
      4+ pairs  -> full homography with RANSAC (most flexible)

    This tiered approach means Stage 2 activates for ALL pairs that have
    at least 1 boundary parcel annotated on both maps, not just pairs
    with 4+ parcels.

    Returns (H_refined, status_string).
    """
    parcels_a = load_parcels_json(map_a)
    parcels_b = load_parcels_json(map_b)

    if parcels_a is None or parcels_b is None:
        return H_rough, "SKIP (Step 5 not run yet)"

    pts_a, pts_b, matched = [], [], []
    for pid in boundary_parcels:
        c_a = find_parcel_centroid(parcels_a, pid)
        c_b = find_parcel_centroid(parcels_b, pid)
        if c_a and c_b:
            pts_a.append(c_a)
            pts_b.append(c_b)
            matched.append(pid)

    n = len(pts_a)

    if n < MIN_CONTROL_PAIRS:
        return H_rough, (f"SKIP (only {n}/{len(boundary_parcels)} "
                         f"parcels annotated in both maps)")

    pa = np.float32(pts_a)
    pb = np.float32(pts_b)

    if n == 1:
        # Pure translation: shift B so its centroid lands on A's centroid.
        # H_rough already encodes the rotation and edge placement from Stage 1.
        # We compute the correction as the difference between where H_rough
        # maps B's centroid and where it should actually land (A's centroid).
        pb_h = np.array([pb[0, 0], pb[0, 1], 1.0], dtype=np.float64)
        predicted = H_rough @ pb_h
        predicted /= predicted[2]
        dx = float(pa[0, 0] - predicted[0])
        dy = float(pa[0, 1] - predicted[1])
        # Correction matrix: pure translation
        T_corr = np.array([[1, 0, dx],
                            [0, 1, dy],
                            [0, 0, 1]], dtype=np.float64)
        H_refined = T_corr @ H_rough
        status = (f"OK (translation): 1 pair, dx={dx:.0f}px dy={dy:.0f}px "
                  f"(parcel: {matched})")

    else:
        # Average translation over all available control points.
        # Affine and homography with few points (2-4) often produces unstable
        # rotation/scale because tiny clicking errors get amplified.  Pure
        # translation is more robust because it preserves the rotation and
        # scale already correctly identified by Stage 1, and only corrects
        # the position offset.
        dxs, dys = [], []
        for i in range(n):
            pb_h = np.array([pb[i, 0], pb[i, 1], 1.0], dtype=np.float64)
            predicted = H_rough @ pb_h
            predicted /= predicted[2]
            dxs.append(float(pa[i, 0] - predicted[0]))
            dys.append(float(pa[i, 1] - predicted[1]))
        # Median is more robust than mean to occasional clicking errors
        dx = float(np.median(dxs))
        dy = float(np.median(dys))
        T_corr = np.array([[1, 0, dx], [0, 1, dy], [0, 0, 1]],
                           dtype=np.float64)
        H_refined = T_corr @ H_rough
        status = (f"OK (translation, {n} points): "
                  f"dx={dx:.0f}px dy={dy:.0f}px "
                  f"(parcels: {matched})")

    print_step(f"  Stage 2: {status}")
    return H_refined, status




# ---------------------------------------------------------------------------
# VISUALISATION
# ---------------------------------------------------------------------------

def draw_registration_preview(gray_a: np.ndarray,
                                gray_b_original: np.ndarray,
                                H: np.ndarray,
                                rotation_b: int,
                                edge_a: str,
                                edge_b: str,
                                map_a: str,
                                map_b: str,
                                stage: str) -> np.ndarray:
    """
    Side-by-side registration preview where BOTH maps are rendered in
    identical natural grayscale so the parcel boundary lines can be visually
    traced from one sheet onto the other.

    Design decisions:
      - Both maps use the same grayscale colour (no tinting).  This is the
        most important request — tinted maps make it hard to see if parcel
        boundary lines actually continue across the join.
      - Map A and Map B are distinguished by a thin coloured BORDER drawn
        around each one (not by tinting the content):
          BLUE border  = Map A
          ORANGE border = Map B
      - Where the maps overlap, A is drawn on top so its content is visible.
      - The shared edge of A is drawn in RED, the shared edge of B in GREEN.
        When red and green overlap, the alignment is correct.
      - The combined canvas is downscaled if it exceeds 6000px on the longest
        side, so the saved PNG is manageable in size while preserving detail.
    """
    H_A, W_A = gray_a.shape
    H_B, W_B = gray_b_original.shape

    # Compute canvas that fits both maps after H is applied to B
    corners_b = np.float32([[0, 0], [W_B, 0],
                             [W_B, H_B], [0, H_B]]).reshape(-1, 1, 2)
    corners_b_in_a = cv2.perspectiveTransform(corners_b, H).reshape(-1, 2)

    all_x = np.concatenate([[0, W_A], corners_b_in_a[:, 0]])
    all_y = np.concatenate([[0, H_A], corners_b_in_a[:, 1]])

    x_min, x_max = int(np.floor(all_x.min())), int(np.ceil(all_x.max()))
    y_min, y_max = int(np.floor(all_y.min())), int(np.ceil(all_y.max()))

    dx = -x_min
    dy = -y_min
    canvas_w = x_max - x_min
    canvas_h = y_max - y_min

    # Cap canvas so output stays manageable. Use a uniform downscale factor
    # if needed so both maps stay aligned.
    MAX_DIM = 6000
    scale_factor = 1.0
    if max(canvas_w, canvas_h) > MAX_DIM:
        scale_factor = MAX_DIM / max(canvas_w, canvas_h)
        canvas_w = int(canvas_w * scale_factor)
        canvas_h = int(canvas_h * scale_factor)
        dx = int(dx * scale_factor)
        dy = int(dy * scale_factor)

    # Build the per-map placement transforms in canvas pixels
    S = np.array([[scale_factor, 0, 0],
                  [0, scale_factor, 0],
                  [0, 0, 1]], dtype=np.float64)
    T_off = np.array([[1, 0, dx], [0, 1, dy], [0, 0, 1]], dtype=np.float64)
    M_A = T_off @ S
    M_B = T_off @ S @ H

    # Render map A and map B into the canvas as grayscale (no tint)
    placed_a = cv2.warpPerspective(
        gray_a, M_A, (canvas_w, canvas_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT, borderValue=255
    )
    placed_b = cv2.warpPerspective(
        gray_b_original, M_B, (canvas_w, canvas_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT, borderValue=255
    )

    # Coverage masks tell us where each map actually placed pixels
    mask_a = cv2.warpPerspective(
        np.ones((H_A, W_A), dtype=np.uint8) * 255, M_A,
        (canvas_w, canvas_h), borderValue=0
    ) > 0
    mask_b = cv2.warpPerspective(
        np.ones((H_B, W_B), dtype=np.uint8) * 255, M_B,
        (canvas_w, canvas_h), borderValue=0
    ) > 0

    # Compose: white background -> place A first, then B on top in overlap
    # zone.  B is drawn on top so that the joining/incoming map's parcel
    # lines are fully visible at the boundary, instead of being hidden by
    # A's overlapping content.  This is the correct visual choice when the
    # purpose is to verify that B's parcels connect properly to A.
    canvas_gray = np.ones((canvas_h, canvas_w), dtype=np.uint8) * 255
    canvas_gray[mask_a] = placed_a[mask_a]
    canvas_gray[mask_b] = placed_b[mask_b]

    # Convert to BGR so we can draw coloured overlays
    canvas = cv2.cvtColor(canvas_gray, cv2.COLOR_GRAY2BGR)

    # Draw coloured borders around each map outline so you can tell them apart
    # without colouring the actual content
    BLUE_BORDER   = (255, 80, 0)     # Map A border
    ORANGE_BORDER = (0, 140, 255)    # Map B border
    border_thickness = max(2, int(8 * scale_factor))

    def draw_outline(img, shape, M, color, thickness):
        h, w = shape
        corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
        warped = cv2.perspectiveTransform(corners, M).reshape(-1, 2).astype(np.int32)
        cv2.polylines(img, [warped], isClosed=True, color=color,
                       thickness=thickness, lineType=cv2.LINE_AA)

    draw_outline(canvas, (H_A, W_A), M_A, BLUE_BORDER,   border_thickness)
    draw_outline(canvas, (H_B, W_B), M_B, ORANGE_BORDER, border_thickness)

    # Highlight the SHARED edge of each map - if the registration is correct
    # these two lines will overlap perfectly along the join
    line_thickness = max(3, int(10 * scale_factor))

    def draw_edge_line(img, shape, M, edge, color, thickness):
        h, w = shape
        if edge == 'right':
            pts = np.float32([[w, 0], [w, h]]).reshape(-1, 1, 2)
        elif edge == 'left':
            pts = np.float32([[0, 0], [0, h]]).reshape(-1, 1, 2)
        elif edge == 'bottom':
            pts = np.float32([[0, h], [w, h]]).reshape(-1, 1, 2)
        elif edge == 'top':
            pts = np.float32([[0, 0], [w, 0]]).reshape(-1, 1, 2)
        else:
            return
        warped = cv2.perspectiveTransform(pts, M).reshape(-1, 2)
        p1 = (int(warped[0, 0]), int(warped[0, 1]))
        p2 = (int(warped[1, 0]), int(warped[1, 1]))
        cv2.line(img, p1, p2, color, thickness, cv2.LINE_AA)

    draw_edge_line(canvas, (H_A, W_A), M_A, edge_a, (0,   0, 255), line_thickness)  # RED  = A shared edge
    draw_edge_line(canvas, (H_B, W_B), M_B, edge_b, (0, 200,   0), line_thickness)  # GREEN = B shared edge

    # Top-left labels
    label1 = (f"[{stage}]  Map {map_a} (BLUE border)  +  Map {map_b} (ORANGE border) | "
              f"B rotated {rotation_b}deg | {edge_a} -> {edge_b}")
    label2 = "Both maps in natural grayscale.  RED line = A shared edge,  GREEN line = B shared edge.  Overlap means correct alignment."

    font_scale = max(0.6, min(canvas_w, canvas_h) / 1800)
    for txt, y_pos in [(label1, 40), (label2, 80)]:
        cv2.putText(canvas, txt, (12, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                    (255, 255, 255), 4, cv2.LINE_AA)
        cv2.putText(canvas, txt, (12, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                    (0, 0, 0), 1, cv2.LINE_AA)

    return canvas


# ---------------------------------------------------------------------------
# SAVE HOMOGRAPHY
# ---------------------------------------------------------------------------

def save_homography(map_a: str, map_b: str,
                    H: np.ndarray,
                    rotation_deg: int,
                    boundary_parcels: list[int],
                    shape_a: tuple,
                    shape_b: tuple,
                    stage1: dict,
                    stage2_status: str):
    """Save homography matrix (.npy) and metadata (.json)."""
    np.save(str(HOMOGRAPHY_DIR / f"H_{map_a}_{map_b}.npy"), H)

    json_path = HOMOGRAPHY_DIR / "homographies.json"
    all_data = {}
    if json_path.exists():
        with open(json_path, 'r', encoding='utf-8') as f:
            all_data = json.load(f)

    all_data[f"{map_a}_{map_b}"] = {
        "map_a":            map_a,
        "map_b":            map_b,
        "rotation_b_deg":   int(rotation_deg),
        "edge_a":           stage1.get('edge_a', ''),
        "edge_b":           stage1.get('edge_b', ''),
        "shift_px":         float(stage1.get('shift', 0)),
        "stage1_score":     float(stage1.get('score', 0)),
        "n_crossings_a":    int(stage1.get('n_crossings_a', 0)),
        "n_crossings_b":    int(stage1.get('n_crossings_b', 0)),
        "stage2_status":    stage2_status,
        "shape_a":          list(shape_a),
        "shape_b":          list(shape_b),
        "boundary_parcels": boundary_parcels,
        "H":                H.tolist(),
    }

    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(all_data, f, indent=2, ensure_ascii=False)


def save_quality_csv(all_results: list[dict]):
    csv_path = HOMOGRAPHY_DIR / "homography_quality.csv"
    fields = ["pair", "map_a", "map_b", "rotation_deg", "edge_a", "edge_b",
              "shift_px", "stage1_score", "n_crossings_a", "n_crossings_b",
              "stage2_status", "final_stage", "boundary_parcels", "elapsed"]

    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        w.writeheader()
        for r in all_results:
            row = {k: r.get(k, '') for k in fields}
            if isinstance(row.get('boundary_parcels'), list):
                row['boundary_parcels'] = ';'.join(str(x) for x in row['boundary_parcels'])
            w.writerow(row)
    print_step(f"Quality CSV saved: {csv_path.name}")


# ---------------------------------------------------------------------------
# MAIN: PROCESS ONE PAIR
# ---------------------------------------------------------------------------

def process_pair(map_a: str,
                 map_b: str,
                 boundary_parcels: list[int]) -> dict:
    """Run the full two-stage pipeline for one adjacency pair."""
    print(f"\n  {'-'*60}")
    print(f"  Pair: map {map_a} <-> map {map_b}   "
          f"(boundary parcels: {boundary_parcels})")
    print(f"  {'-'*60}")
    t0 = time.time()

    # ── Stage 1 ────────────────────────────────────────────────────────────
    print_step(f"  Stage 1: border-line profile alignment")
    try:
        s1 = stage1_alignment(map_a, map_b)
    except FileNotFoundError as e:
        print_step(f"  SKIP: {e}")
        return {"pair": f"{map_a}_{map_b}", "status": "SKIP",
                "map_a": map_a, "map_b": map_b, "elapsed": 0.0}

    if s1['H'] is None:
        print_step(f"  FAIL: no valid alignment found in Stage 1")
        return {"pair": f"{map_a}_{map_b}", "status": "FAIL",
                "map_a": map_a, "map_b": map_b, "elapsed": time.time() - t0}

    H = s1['H']
    rotation = s1['rotation']

    # ── Stage 2 ────────────────────────────────────────────────────────────
    print_step(f"  Stage 2: parcel centroid refinement")
    H_final, stage2_status = stage2_refinement(
        map_a, map_b, boundary_parcels, H
    )
    is_stage2 = stage2_status.startswith("OK")
    final_stage = "Stage2" if is_stage2 else "Stage1"

    # ── Save ───────────────────────────────────────────────────────────────
    save_homography(
        map_a, map_b, H_final,
        rotation_deg    = rotation,
        boundary_parcels = boundary_parcels,
        shape_a         = s1['shape_a'],
        shape_b         = s1['shape_b'],
        stage1          = s1,
        stage2_status   = stage2_status,
    )
    print_step(f"  Saved: H_{map_a}_{map_b}.npy + homographies.json")

    # ── Visualisation ───────────────────────────────────────────────────────
    try:
        gray_a, _ = load_map_images(map_a)
        gray_b, _ = load_map_images(map_b)
        overlay = draw_registration_preview(
            gray_a, gray_b, H_final, rotation,
            s1['edge_a'], s1['edge_b'],
            map_a, map_b, final_stage
        )
        save_image(MATCHES_DIR / f"map_{map_a}_{map_b}_registration.png", overlay)
        print_step(f"  Saved: map_{map_a}_{map_b}_registration.png")
    except Exception as e:
        print_step(f"  WARN: visualisation skipped — {e}")

    elapsed = time.time() - t0
    print_step(f"  [{final_stage}] Done {elapsed:.1f}s | "
               f"rot={rotation}deg | {s1['edge_a']}->{s1['edge_b']} | "
               f"shift={s1['shift']:.0f}px | score={s1['score']:.3f}")

    return {
        "pair":             f"{map_a}_{map_b}",
        "map_a":            map_a,
        "map_b":            map_b,
        "boundary_parcels": boundary_parcels,
        "rotation_deg":     rotation,
        "edge_a":           s1['edge_a'],
        "edge_b":           s1['edge_b'],
        "shift_px":         s1['shift'],
        "stage1_score":     s1['score'],
        "n_crossings_a":    s1.get('n_crossings_a', 0),
        "n_crossings_b":    s1.get('n_crossings_b', 0),
        "stage2_status":    stage2_status,
        "final_stage":      final_stage,
        "status":           final_stage,
        "elapsed":          round(elapsed, 2),
    }


# ---------------------------------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------------------------------

def run():
    print("\n" + "=" * 70)
    print("  STEP 4 — TWO-STAGE BORDER-PROFILE & PARCEL-CENTROID REGISTRATION")
    print("  Cadastral Panoramic Reconstruction — Masters Thesis")
    print("=" * 70)

    make_dirs()

    # Clear old homographies.json
    json_path = HOMOGRAPHY_DIR / "homographies.json"
    if json_path.exists():
        json_path.unlink()
        print_step("Cleared previous homographies.json")

    # Check Stage 2 readiness
    stage2_ready = any(
        (DETECTED_DIR / f"map_{p[0]}_parcels.json").exists()
        for p in ADJACENCY_PAIRS
    )

    print_step(f"Adjacency pairs  : {len(ADJACENCY_PAIRS)}")
    print_step(f"Maps involved    : {', '.join(MAPS_IN_USE)}")
    print_step(f"Stage 1          : border-line cross-correlation (always runs)")
    print_step(f"Stage 2          : parcel centroid refinement "
               f"({'READY' if stage2_ready else 'NOT READY — run Step 5 first'})")

    all_results = []
    for map_a, map_b, boundary_parcels in ADJACENCY_PAIRS:
        result = process_pair(map_a, map_b, boundary_parcels)
        all_results.append(result)

    save_quality_csv(all_results)

    # ── Summary table ────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  STEP 4 SUMMARY")
    print("=" * 70)
    print(f"\n  {'Pair':>7} | {'Rot':>4} | {'Edges':>11} | "
          f"{'nA':>4} | {'nB':>4} | {'Shift':>8} | "
          f"{'Score':>6} | {'Stage':>7} | {'Time':>6}")
    print("  " + "-" * 78)

    n_s1 = n_s2 = n_fail = n_skip = 0
    for r in all_results:
        st = r.get('status', 'SKIP')
        if   st == 'Stage2': n_s2   += 1
        elif st == 'Stage1': n_s1   += 1
        elif st == 'FAIL':   n_fail += 1
        else:                n_skip += 1

        edges = f"{r.get('edge_a','?')}->{r.get('edge_b','?')}"
        print(f"  {r['pair']:>7} | "
              f"{r.get('rotation_deg', 0):>3}  | "
              f"{edges:>11} | "
              f"{r.get('n_crossings_a', 0):>4} | "
              f"{r.get('n_crossings_b', 0):>4} | "
              f"{r.get('shift_px', 0):>7.0f}px | "
              f"{r.get('stage1_score', 0):>6.3f} | "
              f"{st:>7} | "
              f"{r.get('elapsed', 0):>5.1f}s")

    total_time = sum(r.get('elapsed', 0) for r in all_results)
    print(f"\n  Stage 1 only : {n_s1}")
    print(f"  Stage 2 used : {n_s2}")
    print(f"  FAIL         : {n_fail}")
    print(f"  SKIP         : {n_skip}")
    print(f"  Total time   : {total_time:.1f}s")
    print(f"\n  Homographies : {HOMOGRAPHY_DIR.resolve()}")
    print(f"  Overlays     : {MATCHES_DIR.resolve()}")

    if not stage2_ready:
        print(f"\n  NOTE: Run Step 5 (OCR), then re-run Step 4 to activate Stage 2.")
        print(f"  Stage 2 uses detected parcel numbers as ground control points")
        print(f"  to refine the Stage 1 border-profile homographies.")

    print(f"\n  → Next: Run Step 5 (OCR parcel detection)")
    print(f"  → Then: Re-run Step 4 for Stage 2 refinement")
    print(f"  → Then: Run Step 6 for final panoramic reconstruction")

    print(f"\n  [Step 4] Status: "
          f"{'SUCCESS' if n_fail == 0 and n_skip == 0 else 'PARTIAL'}")
    print("=" * 70 + "\n")
    return n_fail == 0 and n_skip == 0


if __name__ == "__main__":
    ok = run()
    sys.exit(0 if ok else 1)