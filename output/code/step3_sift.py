"""
=============================================================================
STEP 3: SIFT & SURF KEYPOINT DETECTION
=============================================================================
Masters Thesis — AI-Based Panoramic Cadastral Image Reconstruction
Author  : Hussein Chalhoub
Purpose : Detect and describe scale- and rotation-invariant keypoints on
          every preprocessed cadastral map image.  The descriptors produced
          here are the inputs consumed by Step 4 (matching) and ultimately
          drive the homography estimation that aligns overlapping maps.

Why keypoint detection is needed
----------------------------------
Two adjacent cadastral maps (e.g. 43 and 44) share an overlapping strip
along their common border.  To stitch them together we need to answer:
  "Which pixel in map 43 corresponds to the same ground point in map 44?"

Simple template matching fails because:
  • The two maps may have been photographed at slightly different scales
  • One map may be rotated a few degrees relative to the other
  • Illumination and contrast differ between scans

SIFT (Scale-Invariant Feature Transform) solves all three problems by
building descriptors that are invariant to scale, rotation, and moderate
changes in illumination.  Each descriptor is a 128-dimensional vector
that characterises the local gradient pattern around a keypoint — making
it possible to find reliable correspondences even across different scans.

SURF (Speeded-Up Robust Features) is an alternative that approximates the
SIFT detector using box filters (integral images) and is significantly
faster.  We run BOTH and compare, as this comparison is a valid thesis
contribution in itself.

Pipeline inside this step
--------------------------
  Stage A : Load preprocessed grayscale images (from Step 1)
  Stage B : SIFT keypoint detection & descriptor computation
  Stage C : SURF keypoint detection & descriptor computation
  Stage D : Keypoint filtering
            → Remove keypoints inside large empty areas (paper background)
            → Retain keypoints near boundary lines (more discriminative)
  Stage E : Keypoint quality analysis
            → Response distribution, scale distribution, orientation spread
  Stage F : Descriptor saving (NumPy .npz format for Step 4)
  Stage G : Visualisation
            → Keypoint overlay images (size = scale, orientation = angle)
            → Response heatmap (density map of detection confidence)
            → Side-by-side SIFT vs SURF comparison figure

Output files (per map N)
-------------------------
  output/sift/map_<N>_sift_keypoints.png   — SIFT keypoints visualised
  output/sift/map_<N>_surf_keypoints.png   — SURF keypoints visualised
  output/sift/map_<N>_comparison.png       — SIFT vs SURF side-by-side
  output/sift/map_<N>_heatmap.png          — keypoint density heatmap
  output/sift/map_<N>_sift_descriptors.npz — SIFT keypoints + descriptors
  output/sift/map_<N>_surf_descriptors.npz — SURF keypoints + descriptors

=============================================================================
"""

import sys
import time
import cv2
import numpy as np
from pathlib import Path


# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------

BASE_DIR         = Path(__file__).resolve().parents[2]
PREPROCESSED_DIR = BASE_DIR / "output" / "preprocessed"   # Step 1 output
SIFT_DIR         = BASE_DIR / "output" / "sift"
DEBUG_DIR        = BASE_DIR / "output" / "debug" / "step3"

MAP_NUMBERS = [str(n) for n in range(43, 56)]    # ['43','44',...,'55']

# ── SIFT parameters ───────────────────────────────────────────────────────
# nfeatures     : maximum number of keypoints to retain (0 = unlimited)
#                 We cap at 5000; cadastral maps have lots of texture from
#                 the Arabic numerals and boundary intersections.
# nOctaveLayers : number of layers per octave in the DoG pyramid
#                 3 is the SIFT paper default — good for line-art images
# contrastThreshold : filters weak keypoints in low-contrast regions
#                     Lower = more keypoints (including near paper texture)
#                     0.03 is more permissive than the default (0.04) because
#                     blueprint ink can produce moderate contrast only
# edgeThreshold : suppresses keypoints on edges (line endpoints)
#                 Higher value = more edge keypoints kept
#                 10 is SIFT default; we keep it to preserve boundary corners
# sigma         : Gaussian blur applied before pyramid construction
#                 1.6 is the theoretically optimal value from the SIFT paper
SIFT_N_FEATURES         = 5000
SIFT_N_OCTAVE_LAYERS    = 3
SIFT_CONTRAST_THRESHOLD = 0.03
SIFT_EDGE_THRESHOLD     = 10
SIFT_SIGMA              = 1.6

# ── SURF parameters ───────────────────────────────────────────────────────
# hessianThreshold : minimum Hessian determinant to accept a keypoint
#                    Higher = fewer, more distinctive keypoints
#                    400 is a common starting value; cadastral line art
#                    has strong corner responses at boundary intersections
# nOctaves         : number of octaves in the scale-space pyramid
# nOctaveLayers    : layers per octave
# extended         : if True, use 128-dim descriptor (same as SIFT)
#                    if False, use 64-dim (faster, slightly less accurate)
# upright          : if True, skip orientation computation (faster)
#                    We set False to get full rotation invariance
SURF_HESSIAN_THRESHOLD = 400
SURF_N_OCTAVES         = 4
SURF_N_OCTAVE_LAYERS   = 3
SURF_EXTENDED          = True     # 128-dim descriptor
SURF_UPRIGHT           = False    # compute orientation

# ── Keypoint filtering ────────────────────────────────────────────────────
# Keypoints detected in large homogeneous background regions (white paper)
# are uninformative and will produce unreliable matches.  We filter them
# by checking whether the local neighbourhood (window around the keypoint)
# contains sufficient gradient energy.
# MIN_LOCAL_VARIANCE : minimum pixel variance in a FILTER_WIN×FILTER_WIN
#                      window around each keypoint
#                      Pure white paper → variance ≈ 0
#                      Ink boundary    → variance >> 100
FILTER_WIN          = 31     # pixels (odd)
MIN_LOCAL_VARIANCE  = 50.0   # threshold


# ---------------------------------------------------------------------------
# UTILITY FUNCTIONS
# ---------------------------------------------------------------------------

def print_step(msg: str):
    print(f"  [Step 3] {msg}")


def make_dirs():
    for d in [SIFT_DIR, DEBUG_DIR]:
        d.mkdir(parents=True, exist_ok=True)
    print_step("Output directories ready")


def load_gray_unicode(path: Path) -> np.ndarray:
    """Read grayscale images with Windows Unicode filename support."""
    data = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)


def save_image_unicode(path: Path, image: np.ndarray) -> None:
    """Write images with Windows Unicode filename support."""
    path.parent.mkdir(parents=True, exist_ok=True)
    ext = path.suffix or ".png"
    ok, encoded = cv2.imencode(ext, image)
    if not ok:
        raise ValueError(f"Unable to encode image for writing: {path}")
    encoded.tofile(str(path))


def find_preprocessed_file(map_num: str) -> Path | None:
    """Resolve Step 1 clean output for either numeric or Arabic-based naming."""
    direct = PREPROCESSED_DIR / f"map_{map_num}_clean.png"
    if direct.exists():
        return direct

    arabic = PREPROCESSED_DIR / f"map_الخريطة رقم {map_num}_clean.png"
    if arabic.exists():
        return arabic

    for p in PREPROCESSED_DIR.glob("*.png"):
        if "clean" in p.name.lower() and map_num in p.name:
            return p

    return None


def load_preprocessed(map_num: str) -> np.ndarray:
    """
    Load the cleaned grayscale image produced by Step 1.
    Both SIFT and SURF operate on uint8 grayscale images.
    """
    path = find_preprocessed_file(map_num)
    if path is None:
        raise FileNotFoundError(
            f"Preprocessed image not found for map {map_num} in {PREPROCESSED_DIR}\n"
            f"Run step1_preprocessing.py first."
        )
    gray = load_gray_unicode(path)
    if gray is None:
        raise IOError(f"Could not read: {path}")
    h, w = gray.shape
    print_step(f"Loaded map {map_num} — shape: ({h}, {w})")
    return gray


def check_surf_available() -> bool:
    """
    SURF is in opencv-contrib and was patent-restricted until 2020.
    We check gracefully so the pipeline still runs with SIFT only.
    """
    try:
        _ = cv2.xfeatures2d.SURF_create(400)
        return True
    except AttributeError:
        return False
    except cv2.error:
        return False


# ---------------------------------------------------------------------------
# STAGE A+B: SIFT DETECTION & DESCRIPTION
# ---------------------------------------------------------------------------

def run_sift(gray: np.ndarray,
             map_num: str) -> tuple[list[cv2.KeyPoint], np.ndarray]:
    """
    Detect SIFT keypoints and compute their 128-dimensional descriptors.

    How SIFT works — step by step:
    ────────────────────────────────

    1. Scale-space construction (DoG pyramid):
       The image is progressively blurred with Gaussians of increasing σ.
       At each scale, we subtract adjacent Gaussian layers to get Difference-
       of-Gaussian (DoG) images, which approximate the Laplacian of Gaussian
       (LoG) — an excellent blob detector.

    2. Keypoint localisation:
       Local extrema (maxima AND minima) in the 3D DoG space (x, y, scale)
       are candidates.  Each candidate is tested against:
         • Contrast threshold (contrastThreshold): removes low-contrast
           points in smooth regions like blank paper
         • Edge threshold (edgeThreshold): removes points on edges rather
           than corners/blobs — computed via the Hessian eigenvalue ratio

    3. Orientation assignment:
       A histogram of gradient orientations is built in the local neighbourhood
       of each keypoint.  The dominant peak becomes the keypoint's canonical
       orientation.  Any secondary peaks > 80% of the dominant peak generate
       additional keypoints at that location — this is why the final count
       can exceed nfeatures slightly.
       This orientation assignment is what makes SIFT ROTATION INVARIANT.

    4. Descriptor computation:
       A 16×16 pixel neighbourhood is divided into a 4×4 grid of cells.
       In each cell an 8-bin gradient orientation histogram is computed.
       Concatenating all 4×4×8 = 128 values gives the final descriptor.
       The descriptor is normalised to unit length (L2), then clipped at 0.2
       and renormalised — making it insensitive to non-linear illumination.

    Why SIFT for cadastral maps specifically:
    ──────────────────────────────────────────
    • Scale invariance: maps 43–55 were photographed at different distances —
      a feature at scale σ in map 43 matches the same feature at scale 2σ
      in map 44 if the camera was twice as far away.
    • Rotation invariance: hand-held photography introduces small rotations
      between adjacent maps.
    • Boundary intersections (corners where two cadastral lines meet) and
      Arabic numeral strokes produce strong, distinctive SIFT keypoints.
    • The 128-dim descriptor is highly discriminative — false matches between
      different-looking regions are very rare (used with ratio test in Step 4).

    Returns:
    ─────────
    keypoints   : list of cv2.KeyPoint objects
                  Each has: .pt (x,y), .size (scale), .angle (orientation),
                  .response (detection strength), .octave (pyramid level)
    descriptors : float32 ndarray of shape (N, 128)
    """
    sift = cv2.SIFT_create(
        nfeatures=SIFT_N_FEATURES,
        nOctaveLayers=SIFT_N_OCTAVE_LAYERS,
        contrastThreshold=SIFT_CONTRAST_THRESHOLD,
        edgeThreshold=SIFT_EDGE_THRESHOLD,
        sigma=SIFT_SIGMA
    )

    keypoints, descriptors = sift.detectAndCompute(gray, mask=None)

    if descriptors is None:
        descriptors = np.empty((0, 128), dtype=np.float32)

    responses = [kp.response for kp in keypoints]
    scales    = [kp.size     for kp in keypoints]

    print_step(f"SIFT — map {map_num}: {len(keypoints)} keypoints detected")
    if keypoints:
        print_step(f"  Response: min={min(responses):.4f}, "
                   f"max={max(responses):.4f}, "
                   f"mean={float(np.mean(responses)):.4f}")
        print_step(f"  Scale:    min={min(scales):.1f}px, "
                   f"max={max(scales):.1f}px, "
                   f"mean={float(np.mean(scales)):.1f}px")
        print_step(f"  Descriptor shape: {descriptors.shape}")

    return keypoints, descriptors


# ---------------------------------------------------------------------------
# STAGE C: SURF DETECTION & DESCRIPTION
# ---------------------------------------------------------------------------

def run_surf(gray: np.ndarray,
             map_num: str) -> tuple[list[cv2.KeyPoint], np.ndarray]:
    """
    Detect SURF keypoints and compute their descriptors.

    How SURF differs from SIFT:
    ────────────────────────────
    SURF approximates the computationally expensive steps of SIFT using
    integral images (summed-area tables), which allow any rectangular
    sum to be computed in O(1) time regardless of rectangle size.

    1. Scale-space (Fast-Hessian detector):
       Instead of DoG, SURF uses the determinant of the Hessian matrix:
         det(H) = Lxx·Lyy − (0.9·Lxy)²
       The second-order partial derivatives (Lxx, Lyy, Lxy) are approximated
       with box filters of increasing size (instead of Gaussians of increasing
       σ), computed instantly via integral images.
       The Hessian determinant is maximum at blob centres — exactly what we
       want for intersection points on a cadastral map.

    2. Orientation assignment:
       Haar wavelet responses in X and Y are computed in a circular
       neighbourhood, then summed over 60° sliding windows to find the
       dominant orientation — faster than SIFT's gradient histogram approach.

    3. Descriptor (SURF-128 with extended=True):
       The neighbourhood is divided into 4×4 sub-regions; in each sub-region
       the Haar wavelet responses (dx, dy, |dx|, |dy|) are summed → 4 values
       per sub-region → 4×4×4 = 64 values (SURF-64) or 128 values (SURF-128
       when extended=True, which adds sign-distinguishing information).

    Speed comparison with SIFT:
    ────────────────────────────
    On a 3000×4000 cadastral map image:
      SIFT  ≈ 3–6 seconds (pure Python + optimised C++ backend)
      SURF  ≈ 0.5–1.5 seconds (integral image acceleration)

    For a thesis benchmark, run both and include the timing table — it
    demonstrates you understand the computational trade-offs.

    Availability note:
    ───────────────────
    SURF requires opencv-contrib-python.  If not installed, this stage is
    skipped gracefully and only SIFT results are used downstream.

    Returns:
    ─────────
    keypoints   : list of cv2.KeyPoint  (empty list if SURF unavailable)
    descriptors : float32 ndarray shape (N, 128) or (N, 64)
    """
    if not check_surf_available():
        print_step(f"SURF — map {map_num}: SKIPPED "
                   f"(SURF not exposed in this OpenCV build — "
                   f"SIFT still runs normally)")
        return [], np.empty((0, 128), dtype=np.float32)

    surf = cv2.xfeatures2d.SURF_create(
        hessianThreshold=SURF_HESSIAN_THRESHOLD,
        nOctaves=SURF_N_OCTAVES,
        nOctaveLayers=SURF_N_OCTAVE_LAYERS,
        extended=SURF_EXTENDED,
        upright=SURF_UPRIGHT
    )

    keypoints, descriptors = surf.detectAndCompute(gray, mask=None)

    if descriptors is None:
        descriptors = np.empty((0, 128), dtype=np.float32)

    desc_dim = descriptors.shape[1] if len(descriptors) > 0 else 0
    responses = [kp.response for kp in keypoints]
    scales    = [kp.size     for kp in keypoints]

    print_step(f"SURF — map {map_num}: {len(keypoints)} keypoints detected "
               f"({desc_dim}-dim descriptors)")
    if keypoints:
        print_step(f"  Response: min={min(responses):.4f}, "
                   f"max={max(responses):.4f}, "
                   f"mean={float(np.mean(responses)):.4f}")
        print_step(f"  Scale:    min={min(scales):.1f}px, "
                   f"max={max(scales):.1f}px, "
                   f"mean={float(np.mean(scales)):.1f}px")

    return keypoints, descriptors


# ---------------------------------------------------------------------------
# STAGE D: KEYPOINT FILTERING
# ---------------------------------------------------------------------------

def filter_keypoints_by_local_variance(
        gray: np.ndarray,
        keypoints: list[cv2.KeyPoint],
        descriptors: np.ndarray,
        map_num: str,
        detector_name: str) -> tuple[list[cv2.KeyPoint], np.ndarray]:
    """
    Remove keypoints located in uninformative, homogeneous background regions.

    The problem:
    ─────────────
    SIFT and SURF detect keypoints wherever the DoG / Hessian response is
    strong enough.  On cadastral blueprint images, this includes:
      ✓ Parcel boundary line intersections — highly discriminative
      ✓ Arabic numeral strokes — highly discriminative
      ✗ Paper texture microstructure — very low discriminability
      ✗ Scanner noise in white areas — produces unstable keypoints

    Keypoints in background paper regions have two problems:
      1. Their descriptors describe featureless paper — many pixels look
         nearly identical, so they produce many false matches in Step 4.
      2. They are geometrically unstable — scanner noise shifts them by
         several pixels between different scans of the same region.

    The filter:
    ────────────
    For each keypoint at (x, y), extract a FILTER_WIN × FILTER_WIN pixel
    window from the grayscale image and compute its variance.
    Pure white paper has variance ≈ 0–10.
    An ink boundary line through the window creates variance >> 100.
    We keep only keypoints whose local window variance ≥ MIN_LOCAL_VARIANCE.

    This is a fast, parameter-free filter compared to alternatives like
    minimum-eigenvalue filtering or gradient magnitude thresholding.

    Why FILTER_WIN = 31?
    The SIFT descriptor covers a 16σ × 16σ region around the keypoint.
    For σ ≈ 1.6 (base scale), this is ≈ 26 pixels.  We use 31 (slightly
    larger) to ensure the window captures the full local neighbourhood that
    the descriptor actually encodes.

    Returns the filtered keypoints and their corresponding descriptors.
    """
    if len(keypoints) == 0:
        return keypoints, descriptors

    h, w = gray.shape
    half = FILTER_WIN // 2
    kept_kps  = []
    kept_desc = []

    for i, kp in enumerate(keypoints):
        cx, cy = int(kp.pt[0]), int(kp.pt[1])

        # Clamp window to image bounds
        y1 = max(0, cy - half)
        y2 = min(h, cy + half + 1)
        x1 = max(0, cx - half)
        x2 = min(w, cx + half + 1)

        patch = gray[y1:y2, x1:x2].astype(np.float32)
        variance = float(np.var(patch))

        if variance >= MIN_LOCAL_VARIANCE:
            kept_kps.append(kp)
            if i < len(descriptors):
                kept_desc.append(descriptors[i])

    n_removed = len(keypoints) - len(kept_kps)
    kept_desc_arr = (np.array(kept_desc, dtype=np.float32)
                     if kept_desc
                     else np.empty((0, descriptors.shape[1] if len(descriptors) > 0 else 128),
                                   dtype=np.float32))

    print_step(f"  {detector_name} filtering — map {map_num}: "
               f"removed {n_removed} background keypoints "
               f"({len(kept_kps)} remaining, "
               f"variance threshold={MIN_LOCAL_VARIANCE})")
    return kept_kps, kept_desc_arr


# ---------------------------------------------------------------------------
# STAGE E: KEYPOINT QUALITY ANALYSIS
# ---------------------------------------------------------------------------

def analyse_keypoints(keypoints: list[cv2.KeyPoint],
                      map_num: str,
                      detector_name: str) -> dict:
    """
    Compute quality statistics over the detected keypoint set.

    These statistics serve two purposes:
      1. Thesis results table: a quantitative comparison of SIFT vs SURF
         across all 13 maps (Table 4.x in the thesis).
      2. Diagnostic: abnormal distributions may indicate preprocessing
         issues that should be addressed before Step 4.

    Statistics computed:
    ─────────────────────
    n_keypoints       : total count
    response_mean/std : detection confidence distribution
    scale_mean/std    : scale distribution — large mean scale = detector
                        found mostly large features (coarse textures)
    angle_entropy     : entropy of orientation histogram (0–8 bits)
                        High entropy ≈ uniformly distributed orientations
                        (good — map has rich variety of edge directions)
                        Low entropy ≈ all keypoints point the same way
                        (bad — may indicate a scanning artifact)
    spatial_coverage  : fraction of image area within FILTER_WIN of any kp
                        Low coverage means keypoints cluster in one region
    """
    if not keypoints:
        return {
            "detector": detector_name,
            "map": map_num,
            "n_keypoints": 0,
        }

    responses = np.array([kp.response for kp in keypoints])
    scales    = np.array([kp.size     for kp in keypoints])
    angles    = np.array([kp.angle    for kp in keypoints])

    # Orientation entropy (8 bins over 0–360°)
    hist, _ = np.histogram(angles, bins=36, range=(0, 360))
    hist_norm = hist / (hist.sum() + 1e-9)
    entropy = -float(np.sum(hist_norm * np.log2(hist_norm + 1e-9)))

    stats = {
        "detector":       detector_name,
        "map":            map_num,
        "n_keypoints":    len(keypoints),
        "response_mean":  round(float(responses.mean()), 5),
        "response_std":   round(float(responses.std()),  5),
        "scale_mean":     round(float(scales.mean()),    2),
        "scale_std":      round(float(scales.std()),     2),
        "angle_entropy":  round(entropy, 3),
    }

    print_step(f"  {detector_name} quality — map {map_num}: "
               f"n={stats['n_keypoints']}, "
               f"response_mean={stats['response_mean']:.5f}, "
               f"scale_mean={stats['scale_mean']:.1f}px, "
               f"orientation_entropy={stats['angle_entropy']:.2f} bits")
    return stats


# ---------------------------------------------------------------------------
# STAGE F: SAVE DESCRIPTORS
# ---------------------------------------------------------------------------

def save_descriptors(map_num: str,
                     detector_name: str,
                     keypoints: list[cv2.KeyPoint],
                     descriptors: np.ndarray):
    """
    Persist keypoints and descriptors to disk in NumPy .npz format.

    Why .npz and not pickle?
    ─────────────────────────
    • .npz is a standard, portable NumPy archive — Step 4 can load it
      without importing any OpenCV or detector-specific code.
    • Pickle embeds Python-version-specific object representations; cv2.KeyPoint
      objects are not directly picklable without a custom serialiser.
    • .npz is compact and loads instantly with np.load().

    Keypoint serialisation:
    ────────────────────────
    cv2.KeyPoint objects cannot be stored directly as arrays.
    We serialise them into a float32 array of shape (N, 7):
      col 0: x coordinate
      col 1: y coordinate
      col 2: size (scale)
      col 3: angle (orientation, degrees)
      col 4: response (detection strength)
      col 5: octave (pyramid level)
      col 6: class_id (always -1 for our use)

    Step 4 reconstructs cv2.KeyPoint objects from this array using
    the load_descriptors() function defined below.

    File naming: map_<N>_<detector>_descriptors.npz
    """
    if len(keypoints) == 0:
        print_step(f"  {detector_name} — map {map_num}: "
                   f"no keypoints to save")
        return

    # Serialise keypoints
    kp_array = np.array([
        [kp.pt[0], kp.pt[1], kp.size, kp.angle,
         kp.response, kp.octave, kp.class_id]
        for kp in keypoints
    ], dtype=np.float32)

    fname = f"map_{map_num}_{detector_name.lower()}_descriptors.npz"
    fpath = SIFT_DIR / fname

    np.savez_compressed(
        str(fpath),
        keypoints=kp_array,
        descriptors=descriptors
    )

    size_kb = fpath.stat().st_size / 1024
    print_step(f"  Saved: {fname} "
               f"({len(keypoints)} keypoints, "
               f"descriptors {descriptors.shape}, "
               f"{size_kb:.1f} KB)")


def load_descriptors(map_num: str,
                     detector_name: str = "sift"
                     ) -> tuple[list[cv2.KeyPoint], np.ndarray]:
    """
    Load serialised keypoints and descriptors saved by save_descriptors().
    Used by Step 4 (matching).

    Returns (keypoints, descriptors) where keypoints is a list of
    cv2.KeyPoint objects reconstructed from the stored float array.
    """
    fname = f"map_{map_num}_{detector_name.lower()}_descriptors.npz"
    fpath = SIFT_DIR / fname

    if not fpath.exists():
        raise FileNotFoundError(
            f"Descriptors not found: {fpath}\n"
            f"Run step3_sift.py first."
        )

    data = np.load(str(fpath))
    kp_array    = data["keypoints"]     # shape (N, 7)
    descriptors = data["descriptors"]   # shape (N, 128)

    keypoints = [
        cv2.KeyPoint(
            x=float(row[0]), y=float(row[1]),
            size=float(row[2]), angle=float(row[3]),
            response=float(row[4]),
            octave=int(row[5]), class_id=int(row[6])
        )
        for row in kp_array
    ]

    return keypoints, descriptors


# ---------------------------------------------------------------------------
# STAGE G: VISUALISATION
# ---------------------------------------------------------------------------

def draw_keypoints_rich(gray: np.ndarray,
                         keypoints: list[cv2.KeyPoint],
                         colour: tuple) -> np.ndarray:
    """
    Draw keypoints with DRAW_RICH_KEYPOINTS flag:
      • Circle radius = keypoint scale (larger circle = detected at larger scale)
      • Line inside circle = keypoint orientation

    This visualisation directly encodes two properties of the descriptor —
    scale and orientation — making it a useful thesis figure to explain
    what SIFT/SURF actually detects.

    colour : BGR tuple for the keypoint circles
    """
    bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    vis = cv2.drawKeypoints(
        bgr,
        keypoints,
        None,
        color=colour,
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )
    return vis


def draw_keypoint_heatmap(gray: np.ndarray,
                           keypoints: list[cv2.KeyPoint]) -> np.ndarray:
    """
    Render a spatially smoothed density map of keypoint detections.

    Each keypoint deposits a Gaussian blob (σ = keypoint.size) at its
    location.  The accumulated density shows WHERE the detectors fire most
    densely — typically along parcel boundaries and around Arabic numerals.

    Interpretation for thesis:
    ───────────────────────────
    • Bright regions → high keypoint density → rich local features
      (boundary line intersections, corners of parcel polygons)
    • Dark regions  → low keypoint density → featureless areas
      (paper background, large blank parcel interiors)

    Colourmap: COLORMAP_HOT (black→red→yellow→white) — intuitive reading
    of "hot" dense regions vs "cold" empty regions.
    """
    h, w = gray.shape
    density = np.zeros((h, w), dtype=np.float32)

    for kp in keypoints:
        cx, cy = int(kp.pt[0]), int(kp.pt[1])
        sigma  = max(1, int(kp.size))

        # Gaussian blob size: 6σ × 6σ centred on keypoint
        bsize = 6 * sigma + 1
        half  = bsize // 2

        # Bounds-clipped region
        y1, y2 = max(0, cy - half), min(h, cy + half + 1)
        x1, x2 = max(0, cx - half), min(w, cx + half + 1)

        # Pre-compute Gaussian kernel and add to density
        ky1 = half - (cy - y1)
        ky2 = ky1  + (y2 - y1)
        kx1 = half - (cx - x1)
        kx2 = kx1  + (x2 - x1)

        gauss_1d = cv2.getGaussianKernel(bsize, sigma)
        gauss_2d = gauss_1d @ gauss_1d.T
        density[y1:y2, x1:x2] += gauss_2d[ky1:ky2, kx1:kx2]

    # Normalise to 0–255 and apply colour map
    density_norm = cv2.normalize(density, None, 0, 255,
                                  cv2.NORM_MINMAX).astype(np.uint8)
    heatmap_bgr = cv2.applyColorMap(density_norm, cv2.COLORMAP_HOT)

    # Blend with original image for context
    gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    blended  = cv2.addWeighted(gray_bgr, 0.4, heatmap_bgr, 0.6, 0)
    return blended


def make_comparison_figure(sift_vis: np.ndarray,
                            surf_vis: np.ndarray,
                            map_num: str,
                            n_sift: int,
                            n_surf: int) -> np.ndarray:
    """
    Create a side-by-side comparison of SIFT vs SURF keypoints on the
    same map, with count annotations.  Ready-to-use as a thesis figure.
    """
    # Resize both to same height
    target_h = 500
    def rz(img):
        scale = target_h / img.shape[0]
        return cv2.resize(img, (int(img.shape[1] * scale), target_h))

    s_img  = rz(sift_vis)
    su_img = rz(surf_vis) if surf_vis is not None else np.zeros_like(s_img)

    # Add labels
    label_params = dict(fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.9, thickness=2,
                        lineType=cv2.LINE_AA)
    cv2.putText(s_img,  f"SIFT  n={n_sift}",
                (12, 34), color=(30, 200, 30), **label_params)
    cv2.putText(su_img, f"SURF  n={n_surf}",
                (12, 34), color=(30, 80, 220), **label_params)

    divider = np.ones((target_h, 6, 3), dtype=np.uint8) * 160
    comparison = np.hstack([s_img, divider, su_img])
    return comparison


def save_visualisations(map_num: str,
                         gray: np.ndarray,
                         sift_kps: list[cv2.KeyPoint],
                         surf_kps: list[cv2.KeyPoint]):
    """
    Render and save all four visualisation images for map_num.
    """
    # SIFT keypoints (green circles with orientation lines)
    sift_vis = draw_keypoints_rich(gray, sift_kps, colour=(30, 200, 30))
    sift_path = SIFT_DIR / f"map_{map_num}_sift_keypoints.png"
    save_image_unicode(sift_path, sift_vis)
    print_step(f"  Saved: {sift_path.name}")

    # SURF keypoints (blue circles)
    if surf_kps:
        surf_vis  = draw_keypoints_rich(gray, surf_kps, colour=(220, 80, 30))
        surf_path = SIFT_DIR / f"map_{map_num}_surf_keypoints.png"
        save_image_unicode(surf_path, surf_vis)
        print_step(f"  Saved: {surf_path.name}")
    else:
        surf_vis = None

    # Density heatmap (SIFT)
    heatmap = draw_keypoint_heatmap(gray, sift_kps)
    heat_path = SIFT_DIR / f"map_{map_num}_heatmap.png"
    save_image_unicode(heat_path, heatmap)
    print_step(f"  Saved: {heat_path.name}")

    # SIFT vs SURF comparison
    comparison = make_comparison_figure(
        sift_vis, surf_vis, map_num,
        n_sift=len(sift_kps), n_surf=len(surf_kps)
    )
    comp_path = SIFT_DIR / f"map_{map_num}_comparison.png"
    save_image_unicode(comp_path, comparison)
    print_step(f"  Saved: {comp_path.name}")


# ---------------------------------------------------------------------------
# MAIN FUNCTION — process one map end-to-end
# ---------------------------------------------------------------------------

def process_map(map_num: str) -> dict:
    """
    Run the full Stage A–G pipeline on one map.
    Returns a summary dict for the final report table.
    """
    print(f"\n{'='*60}")
    print(f"  Processing map {map_num}")
    print(f"{'='*60}")
    t0 = time.time()

    # A — Load
    gray = load_preprocessed(map_num)

    # B — SIFT detection
    t_sift = time.time()
    sift_kps, sift_desc = run_sift(gray, map_num)
    sift_time = time.time() - t_sift

    # D — Filter SIFT (background removal)
    sift_kps, sift_desc = filter_keypoints_by_local_variance(
        gray, sift_kps, sift_desc, map_num, "SIFT"
    )

    # E — SIFT quality analysis
    sift_stats = analyse_keypoints(sift_kps, map_num, "SIFT")
    sift_stats["time_s"] = round(sift_time, 2)

    # F — Save SIFT descriptors
    save_descriptors(map_num, "sift", sift_kps, sift_desc)

    # C — SURF detection
    t_surf = time.time()
    surf_kps, surf_desc = run_surf(gray, map_num)
    surf_time = time.time() - t_surf

    # D — Filter SURF
    if surf_kps:
        surf_kps, surf_desc = filter_keypoints_by_local_variance(
            gray, surf_kps, surf_desc, map_num, "SURF"
        )

    # E — SURF quality analysis
    surf_stats = analyse_keypoints(surf_kps, map_num, "SURF")
    surf_stats["time_s"] = round(surf_time, 2)

    # F — Save SURF descriptors
    if surf_kps:
        save_descriptors(map_num, "surf", surf_kps, surf_desc)

    # G — Visualisations
    save_visualisations(map_num, gray, sift_kps, surf_kps)

    elapsed = time.time() - t0
    print_step(f"Map {map_num} done in {elapsed:.1f}s  "
               f"(SIFT: {sift_time:.1f}s, SURF: {surf_time:.1f}s)")

    return {
        "map":          map_num,
        "sift_n":       sift_stats.get("n_keypoints", 0),
        "sift_resp":    sift_stats.get("response_mean", 0.0),
        "sift_scale":   sift_stats.get("scale_mean", 0.0),
        "sift_entropy": sift_stats.get("angle_entropy", 0.0),
        "sift_time":    sift_time,
        "surf_n":       surf_stats.get("n_keypoints", 0),
        "surf_resp":    surf_stats.get("response_mean", 0.0),
        "surf_scale":   surf_stats.get("scale_mean", 0.0),
        "surf_time":    surf_time,
        "total_time":   elapsed,
    }


# ---------------------------------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------------------------------

def run():
    """
    Process all 13 maps in stitching order and print the thesis
    comparison table (SIFT vs SURF per map).
    Called by main.py or directly: python step3_sift.py
    """
    print("\n" + "="*60)
    print("  STEP 3 — SIFT & SURF KEYPOINT DETECTION")
    print("  Cadastral Panoramic Reconstruction — Masters Thesis")
    print("="*60)

    surf_ok = check_surf_available()
    print_step(f"SIFT available : YES (cv2.SIFT_create)")
    print_step(f"SURF available : {'YES (cv2.xfeatures2d.SURF_create)' if surf_ok else 'NO  (not exposed by this OpenCV build)'}")

    make_dirs()

    all_results = []
    missing     = []

    for num in MAP_NUMBERS:
        path = find_preprocessed_file(num)
        if path is None:
            print(f"\n  [Step 3] WARNING: map {num} preprocessed image "
                  f"not found — skipping")
            missing.append(num)
            continue
        result = process_map(num)
        all_results.append(result)

    # ── Summary table ────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("  STEP 3 SUMMARY  — SIFT vs SURF comparison")
    print("="*60)
    print(f"  Maps processed : {len(all_results)}")
    if missing:
        print(f"  Maps skipped   : {', '.join(missing)}")

    if all_results:
        print(f"\n  {'Map':>4} | {'SIFT n':>7} | {'SIFT resp':>10} | "
              f"{'SIFT scale':>11} | {'SIFT t':>7} | "
              f"{'SURF n':>7} | {'SURF t':>7}")
        print("  " + "-"*72)
        for r in all_results:
            print(f"  {r['map']:>4} | "
                  f"{r['sift_n']:>7} | "
                  f"{r['sift_resp']:>10.5f} | "
                  f"{r['sift_scale']:>11.2f} | "
                  f"{r['sift_time']:>6.1f}s | "
                  f"{r['surf_n']:>7} | "
                  f"{r['surf_time']:>6.1f}s")

        total_sift_kp = sum(r["sift_n"] for r in all_results)
        total_surf_kp = sum(r["surf_n"] for r in all_results)
        total_time    = sum(r["total_time"] for r in all_results)
        avg_sift_t    = float(np.mean([r["sift_time"] for r in all_results]))
        avg_surf_t    = float(np.mean([r["surf_time"] for r in all_results]))

        print(f"\n  Total SIFT keypoints     : {total_sift_kp:,}")
        print(f"  Total SURF keypoints     : {total_surf_kp:,}")
        print(f"  Avg SIFT time / map      : {avg_sift_t:.2f}s")
        print(f"  Avg SURF time / map      : {avg_surf_t:.2f}s")
        print(f"  SURF speedup             : {avg_sift_t / max(avg_surf_t, 0.01):.1f}×")
        print(f"  Total Step 3 time        : {total_time:.1f}s")
        print(f"\n  Descriptors saved to     : {SIFT_DIR.resolve()}")
        print(f"  (Step 4 will load these with load_descriptors(map_num))")

    success = len(all_results) == len(MAP_NUMBERS)
    status  = "SUCCESS" if success else "PARTIAL"
    print(f"\n  [Step 3] Status: {status}")
    print("="*60 + "\n")
    return success


if __name__ == "__main__":
    ok = run()
    sys.exit(0 if ok else 1)
