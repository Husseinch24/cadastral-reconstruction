"""
=============================================================================
EDGE ORIENTATION FINDER v2 — Fully Automatic, Zero Manual Input
=============================================================================
Masters Thesis — AI-Based Panoramic Cadastral Image Reconstruction
Author  : Hussein Chalhoub

Purpose
--------
Given two maps KNOWN to be adjacent, automatically determines:
  1. Which edge of A touches which edge of B
  2. The exact rotation of B (coarse 0/90/180/270° + fine ±5°)
  3. The alignment offset along the shared boundary
  4. A confidence score and error for every combination tried

Why this works without parcel numbers
--------------------------------------
Parcel boundary lines that exit map A's edge CONTINUE into map B's edge.
This creates a characteristic crossing pattern along the shared boundary.
The wrong edge combinations give near-random correlation — the correct
one gives a sharp peak. No labels, no clicks, no parcel numbers needed.

Pipeline
---------
  Stage 0  — Load + auto-crop white scan margins from both maps
  Stage 1  — Coarse search: 4 edge-pairs × 4 rotations = 16 combinations
             Each scored with:
               • 1D phase correlation (FFT) on the boundary crossing profile
               • 2D NCC at the best phase-correlation offset
               • Ink density similarity (both edges must have similar line density)
  Stage 2  — Fine search: ±5° in 0.5° steps around the best coarse orientation
  Stage 3  — Full ranked table printed + JSON saved + visualisation PNGs

Scoring formula
----------------
  combined = 0.50 × phase_ncc + 0.30 × ncc_2d + 0.20 × density_sim
  error    = 1.0 − combined          (LOWER = BETTER, 0.0 = perfect)

Usage
------
  python new_pipeline/src/edge_orientation_finder.py --pair 47_48
  python new_pipeline/src/edge_orientation_finder.py --pair 47_48 --visualise
  python new_pipeline/src/edge_orientation_finder.py --all --visualise

Output
-------
  new_pipeline/data/orientation_results/
    pair_<A>_<B>_scores.json   — all scores for every combination
    pair_<A>_<B>_grid.png      — 4×4 grid of all 16 coarse orientations
    pair_<A>_<B>_best.png      — stitch preview of the winning alignment
    orientation_summary.json   — best result per pair (all pairs)
=============================================================================
"""

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

# Force UTF-8 output so Unicode box-drawing / arrow chars render correctly
# on Windows terminals that default to cp1252.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------

PREPROCESSED_DIR = Path("output/preprocessed")
RESULTS_DIR      = Path("new_pipeline/data/orientation_results")

ADJACENT_PAIRS = [
    ("45", "47"), ("45", "46"), ("47", "48"), ("48", "49"),
    ("49", "50"), ("50", "51"), ("52", "53"), ("52", "54"),
    ("52", "55"), ("54", "55"),
]

ALL_MAPS = ["43", "44", "45", "46", "47", "48", "49", "50", "51", "52", "53", "54", "55"]

STRIP_WIDTH      = 200      # px — strip extracted from each edge for comparison
COARSE_ROTATIONS = [0, 90, 180, 270]
FINE_RANGE_DEG   = 5.0      # ±5° fine search around best coarse rotation
FINE_STEP_DEG    = 0.5      # 0.5° increments in fine search

VALID_EDGE_PAIRS = [
    ("right",  "left"),    # A on left,   B on right
    ("left",   "right"),   # A on right,  B on left
    ("bottom", "top"),     # A on top,    B on bottom
    ("top",    "bottom"),  # A on bottom, B on top
]

FONT = cv2.FONT_HERSHEY_SIMPLEX


# ---------------------------------------------------------------------------
# IMAGE I/O
# ---------------------------------------------------------------------------

def read_image(path: Path) -> np.ndarray:
    data = np.fromfile(str(path), dtype=np.uint8)
    img  = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Cannot read: {path}")
    return img


def save_image(path: Path, img: np.ndarray):
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, enc = cv2.imencode(path.suffix, img)
    if ok:
        enc.tofile(str(path))

def auto_crop(img: np.ndarray,
              ink_threshold: int = 200,
              min_ink_frac: float = 0.001,
              padding: int = 30) -> np.ndarray:
    """Remove white scan margins. Returns cropped image."""
    h, w   = img.shape
    is_ink = img < ink_threshold
    rows   = np.where(is_ink.mean(axis=1) >= min_ink_frac)[0]
    cols   = np.where(is_ink.mean(axis=0) >= min_ink_frac)[0]
    if not len(rows) or not len(cols):
        return img
    y0 = max(0, int(rows[0])  - padding)
    y1 = min(h, int(rows[-1]) + padding + 1)
    x0 = max(0, int(cols[0]) - padding)
    x1 = min(w, int(cols[-1]) + padding + 1)
    return img[y0:y1, x0:x1]


def rotate_image(img: np.ndarray, angle_deg: float) -> np.ndarray:
    """Rotate by any angle. Uses fast cv2.rotate for 90° multiples."""
    if angle_deg == 0:   return img
    if angle_deg == 90:  return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    if angle_deg == 180: return cv2.rotate(img, cv2.ROTATE_180)
    if angle_deg == 270: return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    h, w = img.shape
    cx, cy = w / 2.0, h / 2.0
    M = cv2.getRotationMatrix2D((cx, cy), angle_deg, 1.0)
    cos_a, sin_a = abs(M[0, 0]), abs(M[0, 1])
    new_w = int(w * cos_a + h * sin_a)
    new_h = int(w * sin_a + h * cos_a)
    M[0, 2] += (new_w - w) / 2.0
    M[1, 2] += (new_h - h) / 2.0
    return cv2.warpAffine(img, M, (new_w, new_h),
                          flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=255)


def extract_strip(img: np.ndarray, edge: str, width: int = STRIP_WIDTH) -> np.ndarray:
    """Extract a pixel strip from one edge of the image."""
    h, w  = img.shape
    width = min(width, w // 4, h // 4)
    if edge == 'right':  return img[:, w - width:].copy()
    if edge == 'left':   return img[:, :width].copy()
    if edge == 'bottom': return img[h - width:, :].copy()
    if edge == 'top':    return img[:width, :].copy()
    raise ValueError(f"Unknown edge: {edge}")


# ---------------------------------------------------------------------------
# SCORING
# ---------------------------------------------------------------------------

def boundary_profile(strip: np.ndarray, edge: str) -> np.ndarray:
    """
    Project an edge strip to a 1D signal representing ink at each position
    along the boundary.

    For vertical edges (left/right): average each row → length = image height.
    For horizontal edges (top/bottom): average each column → length = image width.

    Inverted so ink (dark pixels) becomes a positive peak.
    """
    if edge in ('left', 'right'):
        profile = strip.mean(axis=1).astype(np.float32)
    else:
        profile = strip.mean(axis=0).astype(np.float32)
    return 255.0 - profile   # invert: ink = positive


def phase_correlate_1d(sig_a: np.ndarray, sig_b: np.ndarray) -> tuple[float, int]:
    """
    FFT-based 1D alignment: use cross-correlation to find the best offset, then
    return the 1D NCC at that offset as the score (proper bounded value in [0, 1]).
    """
    sa = sig_a - sig_a.mean()
    sb = sig_b - sig_b.mean()
    n  = int(2 ** np.ceil(np.log2(len(sa) + len(sb))))

    Fa = np.fft.rfft(sa, n=n)
    Fb = np.fft.rfft(sb, n=n)
    # Standard cross-correlation (no whitening) — maximises at the true offset
    corr = np.fft.irfft(Fa * np.conj(Fb), n=n)

    peak_idx = int(np.argmax(corr))
    offset   = peak_idx if peak_idx <= n // 2 else peak_idx - n

    # Score: 1D NCC at the found offset (properly bounded)
    a0 = max(0, offset);  b0 = max(0, -offset)
    L  = min(len(sa) - a0, len(sb) - b0)
    if L < 20:
        return 0.0, int(offset)
    sa_s = sa[a0:a0+L];  sb_s = sb[b0:b0+L]
    std_a, std_b = sa_s.std(), sb_s.std()
    if std_a < 1e-6 or std_b < 1e-6:
        return 0.0, int(offset)
    ncc_val = float(np.dot(sa_s / std_a, sb_s / std_b)) / L
    return max(0.0, min(1.0, (ncc_val + 1.0) / 2.0)), int(offset)


def ncc_2d_at_offset(strip_a: np.ndarray, strip_b: np.ndarray,
                     offset: int, edge: str) -> float:
    """
    2D NCC between two edge strips with the given offset applied.
    The offset shifts along the boundary direction (y for vertical edges,
    x for horizontal edges). Returns a score in [0, 1].
    """
    if edge in ('left', 'right'):
        # Align widths
        w = min(strip_a.shape[1], strip_b.shape[1])
        sa = strip_a[:, :w].astype(np.float32)
        sb = strip_b[:, :w].astype(np.float32)
        ha, hb = sa.shape[0], sb.shape[0]
        a0 = max(0,  offset); b0 = max(0, -offset)
        L  = min(ha - a0, hb - b0)
        if L < 20:
            return 0.0
        sa = sa[a0:a0+L]; sb = sb[b0:b0+L]
    else:
        # Align heights
        h = min(strip_a.shape[0], strip_b.shape[0])
        sa = strip_a[:h, :].astype(np.float32)
        sb = strip_b[:h, :].astype(np.float32)
        wa, wb = sa.shape[1], sb.shape[1]
        a0 = max(0,  offset); b0 = max(0, -offset)
        L  = min(wa - a0, wb - b0)
        if L < 20:
            return 0.0
        sa = sa[:, a0:a0+L]; sb = sb[:, b0:b0+L]

    sa -= sa.mean(); sb -= sb.mean()
    std_a, std_b = sa.std(), sb.std()
    if std_a < 1e-6 or std_b < 1e-6:
        return 0.0
    ncc = float(np.mean((sa / std_a) * (sb / std_b)))
    return max(0.0, (ncc + 1.0) / 2.0)


def ink_density(strip: np.ndarray, threshold: int = 180) -> float:
    """Fraction of pixels darker than threshold (= ink density)."""
    return float((strip < threshold).sum()) / max(strip.size, 1)


def score_one(img_a: np.ndarray, edge_a: str,
              img_b_rot: np.ndarray, edge_b: str,
              coarse_rot: float, fine_rot: float = 0.0) -> dict:
    """
    Score one edge-pair + rotation combination.
    Only pure rotations (0/90/180/270°) are applied — no mirroring/flipping.

    For rotations that reverse the traversal direction along the shared boundary
    (notably 180°), the boundary profile of B is read in the wrong order.
    We therefore try BOTH the forward and reversed profile and keep whichever
    gives the higher 1D NCC, then use that same orientation for the 2D NCC.
    """
    strip_a = extract_strip(img_a,     edge_a)
    strip_b = extract_strip(img_b_rot, edge_b)

    prof_a = boundary_profile(strip_a, edge_a)
    prof_b = boundary_profile(strip_b, edge_b)

    # Try forward and reversed B profile; rotation can flip traversal direction
    phase_f, off_f = phase_correlate_1d(prof_a, prof_b)
    phase_r, off_r = phase_correlate_1d(prof_a, prof_b[::-1])

    if phase_r > phase_f:
        phase_score, best_offset = phase_r, off_r
        # Reverse the 2D strip along the boundary axis for consistent NCC
        strip_b_cmp = strip_b[::-1, :] if edge_b in ('left', 'right') else strip_b[:, ::-1]
    else:
        phase_score, best_offset = phase_f, off_f
        strip_b_cmp = strip_b

    ncc      = ncc_2d_at_offset(strip_a, strip_b_cmp, best_offset, edge_a)
    dens_a   = ink_density(strip_a)
    dens_b   = ink_density(strip_b)
    dens_sim = 1.0 - abs(dens_a - dens_b)

    # NCC2D is the most reliable discriminator for cadastral maps;
    # phase provides the offset; density is a light tiebreaker.
    combined  = 0.80 * ncc + 0.10 * phase_score + 0.10 * dens_sim
    total_rot = coarse_rot + fine_rot

    return {
        "edge_a":         edge_a,
        "edge_b":         edge_b,
        "coarse_rot":     coarse_rot,
        "fine_rot":       round(fine_rot, 2),
        "total_rot":      round(total_rot, 2),
        "phase_score":    round(phase_score, 4),
        "ncc_2d":         round(ncc, 4),
        "density_a":      round(dens_a, 4),
        "density_b":      round(dens_b, 4),
        "density_sim":    round(dens_sim, 4),
        "combined_score": round(combined, 4),
        "error":          round(1.0 - combined, 4),
        "best_offset_px": int(best_offset),
        "label":          f"A:{edge_a} ↔ B:{edge_b}  rot={total_rot:.1f}°",
    }


# ---------------------------------------------------------------------------
# MAIN ANALYSIS
# ---------------------------------------------------------------------------

def analyse_pair(map_a: str, map_b: str, visualise: bool = False, steps: bool = False) -> dict:
    print(f"\n{'='*72}")
    print(f"  PAIR {map_a}_{map_b}")
    print(f"{'='*72}")

    img_a = auto_crop(read_image(PREPROCESSED_DIR / f"map_{map_a}_clean.png"))
    img_b = auto_crop(read_image(PREPROCESSED_DIR / f"map_{map_b}_clean.png"))

    print(f"  Map {map_a} cropped: {img_a.shape[1]}×{img_a.shape[0]} px")
    print(f"  Map {map_b} cropped: {img_b.shape[1]}×{img_b.shape[0]} px")

    # ------------------------------------------------------------------
    # STAGE 1 — Coarse search: 16 combinations
    # ------------------------------------------------------------------
    print(f"\n  STAGE 1 — Coarse rotation search (16 combinations)\n")
    print(f"  {'Orientation':<52}  {'Score':>7}  {'Error':>7}  "
          f"{'Phase':>7}  {'NCC2D':>7}  {'DenSim':>7}  {'Offset':>8}")
    print(f"  {'-'*100}")

    coarse_results = []
    for coarse_rot in COARSE_ROTATIONS:
        img_b_rot = rotate_image(img_b, coarse_rot)
        for edge_a, edge_b in VALID_EDGE_PAIRS:
            r = score_one(img_a, edge_a, img_b_rot, edge_b, coarse_rot)
            coarse_results.append(r)
            print(f"  {r['label']:<52}  "
                  f"{r['combined_score']:>7.4f}  "
                  f"{r['error']:>7.4f}  "
                  f"{r['phase_score']:>7.4f}  "
                  f"{r['ncc_2d']:>7.4f}  "
                  f"{r['density_sim']:>7.4f}  "
                  f"{r['best_offset_px']:>+7d}px")

    coarse_results.sort(key=lambda x: x["error"])
    best_coarse  = coarse_results[0]
    runner_up    = coarse_results[1]

    # ------------------------------------------------------------------
    # STAGE 2 — Fine rotation around best coarse
    # ------------------------------------------------------------------
    print(f"\n  STAGE 2 — Fine rotation ±{FINE_RANGE_DEG}° "
          f"(step {FINE_STEP_DEG}°) around rot={best_coarse['coarse_rot']}° ...")

    fine_angles = np.arange(-FINE_RANGE_DEG,
                             FINE_RANGE_DEG + FINE_STEP_DEG / 2,
                             FINE_STEP_DEG)
    fine_angles = fine_angles[fine_angles != 0.0]

    fine_results = []
    edge_a    = best_coarse["edge_a"]
    edge_b    = best_coarse["edge_b"]
    base_rot  = best_coarse["coarse_rot"]

    for fine_rot in fine_angles:
        img_b_rot = rotate_image(img_b, base_rot + fine_rot)
        r = score_one(img_a, edge_a, img_b_rot, edge_b, base_rot, fine_rot)
        fine_results.append(r)

    all_fine = sorted([best_coarse] + fine_results, key=lambda x: x["error"])
    best = all_fine[0]

    print(f"  Best fine rotation: {best['total_rot']:.1f}°  "
          f"score={best['combined_score']:.4f}  error={best['error']:.4f}")

    # ------------------------------------------------------------------
    # FINAL RANKED TABLE
    # ------------------------------------------------------------------
    confidence_gap = runner_up["error"] - coarse_results[0]["error"]

    print(f"\n  {'='*72}")
    print(f"  FINAL RANKED TABLE --- Coarse orientations (best to worst)")
    print(f"  {'='*72}")
    print(f"  {'#':>3}  {'Orientation':<52}  {'Score':>7}  {'Error':>7}  {'Offset':>8}")
    print(f"  {'-'*85}")
    for i, r in enumerate(coarse_results):
        tag = "  ← BEST" if i == 0 else ("  ← 2nd" if i == 1 else "")
        print(f"  {i+1:>3}  {r['label']:<52}  "
              f"{r['combined_score']:>7.4f}  "
              f"{r['error']:>7.4f}  "
              f"{r['best_offset_px']:>+7d}px{tag}")

    print(f"\n  WINNER  : {best['label']}")
    print(f"  | Phase correlation : {best['phase_score']:.4f}")
    print(f"  | 2D NCC            : {best['ncc_2d']:.4f}")
    print(f"  | Ink density sim   : {best['density_sim']:.4f}")
    print(f"  | Combined score    : {best['combined_score']:.4f}")
    print(f"  | Error             : {best['error']:.4f}")
    print(f"  | Offset            : {best['best_offset_px']:+d} px")
    print(f"  | Confidence gap    : {confidence_gap:.4f}  "
          f"(vs runner-up: {runner_up['label']})")

    # ------------------------------------------------------------------
    # SAVE
    # ------------------------------------------------------------------
    output = {
        "pair":             f"{map_a}_{map_b}",
        "map_a":            map_a,
        "map_b":            map_b,
        "best":             best,
        "runner_up":        runner_up,
        "confidence_gap":   round(confidence_gap, 4),
        "coarse_results":   coarse_results,
        "fine_results":     all_fine,
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    json_path = RESULTS_DIR / f"pair_{map_a}_{map_b}_scores.json"

    def _np_clean(obj):
        if isinstance(obj, dict):
            return {k: _np_clean(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_np_clean(v) for v in obj]
        if isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, (np.integer,)):
            return int(obj)
        return obj

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(_np_clean(output), f, indent=2)
    print(f"\n  Saved JSON: {json_path.name}")

    if visualise:
        _save_grid(map_a, map_b, img_a, img_b, coarse_results)
        _save_best_preview(map_a, map_b, img_a, img_b, best)

    if steps:
        _save_step_images(map_a, map_b, img_a, img_b, best)

    return output


# ---------------------------------------------------------------------------
# VISUALISATION
# ---------------------------------------------------------------------------

def _thumbnail(img: np.ndarray, tw: int, th: int, edge: str) -> np.ndarray:
    """Scale to thumbnail and draw a coloured line on the matched edge."""
    scale = min(tw / img.shape[1], th / img.shape[0])
    rw = int(img.shape[1] * scale)
    rh = int(img.shape[0] * scale)
    t  = cv2.resize(img, (rw, rh), interpolation=cv2.INTER_AREA)
    t  = cv2.cvtColor(t, cv2.COLOR_GRAY2BGR)
    lc = (0, 80, 255); lw = 3
    if edge == 'left':   cv2.line(t, (0, 0), (0, rh), lc, lw)
    if edge == 'right':  cv2.line(t, (rw-1, 0), (rw-1, rh), lc, lw)
    if edge == 'top':    cv2.line(t, (0, 0), (rw, 0), lc, lw)
    if edge == 'bottom': cv2.line(t, (0, rh-1), (rw, rh-1), lc, lw)
    canvas = np.full((th, tw, 3), 35, dtype=np.uint8)
    yo = (th - rh) // 2; xo = (tw - rw) // 2
    canvas[yo:yo+rh, xo:xo+rw] = t
    return canvas


def _save_grid(map_a: str, map_b: str,
               img_a: np.ndarray, img_b: np.ndarray,
               coarse_results: list):
    """4×4 grid showing all 16 coarse orientations with scores."""
    TW, TH   = 260, 170
    PAD      = 10
    LABEL_H  = 72
    TITLE_H  = 52
    COLS = 4; ROWS = 4

    cell_w = 2 * TW + PAD * 3
    cell_h = TH + LABEL_H + PAD
    grid_w = COLS * (cell_w + PAD) + PAD
    grid_h = TITLE_H + ROWS * (cell_h + PAD) + PAD

    grid = np.full((grid_h, grid_w, 3), 22, dtype=np.uint8)

    cv2.putText(grid,
                f"Map {map_a} vs Map {map_b} — All 16 coarse orientations  (green = BEST)",
                (PAD, 36), FONT, 0.68, (210, 210, 210), 2, cv2.LINE_AA)

    for idx, r in enumerate(coarse_results):
        col = idx % COLS; row = idx // COLS
        x0  = PAD + col * (cell_w + PAD)
        y0  = TITLE_H + PAD + row * (cell_h + PAD)

        img_b_rot = rotate_image(img_b, r["coarse_rot"])
        ta = _thumbnail(img_a,     TW, TH, r["edge_a"])
        tb = _thumbnail(img_b_rot, TW, TH, r["edge_b"])

        # Place A and B in geometrically correct order:
        # A:right→B is on A's right.  A:left→B is on A's left.  Always show [A | B].
        # For A:left the maps are swapped visually to avoid confusion.
        if r["edge_a"] in ("left", "top"):
            # B is spatially "before" A — show [B | A]
            grid[y0:y0+TH, x0:x0+TW] = tb
            x1 = x0 + TW + PAD
            grid[y0:y0+TH, x1:x1+TW] = ta
        else:
            grid[y0:y0+TH, x0:x0+TW] = ta
            x1 = x0 + TW + PAD
            grid[y0:y0+TH, x1:x1+TW] = tb

        is_best = (idx == 0)
        tc = (70, 210, 70) if is_best else (150, 150, 150)
        lx = x0; ly = y0 + TH + 18

        short_label = f"A:{r['edge_a']} ↔ B:{r['edge_b']}  rot={r['coarse_rot']}°"
        cv2.putText(grid, short_label,                              (lx, ly),    FONT, 0.40, tc, 1, cv2.LINE_AA)
        cv2.putText(grid, f"score={r['combined_score']:.4f}  error={r['error']:.4f}", (lx, ly+20), FONT, 0.40, tc, 1, cv2.LINE_AA)
        cv2.putText(grid, f"phase={r['phase_score']:.3f}  ncc2d={r['ncc_2d']:.3f}", (lx, ly+40), FONT, 0.40, tc, 1, cv2.LINE_AA)
        cv2.putText(grid, f"offset={r['best_offset_px']:+d}px  dens_sim={r['density_sim']:.3f}", (lx, ly+58), FONT, 0.40, tc, 1, cv2.LINE_AA)

        if is_best:
            cv2.rectangle(grid,
                          (x0-3, y0-3),
                          (x0 + cell_w - PAD + 2, y0 + cell_h - 1),
                          (60, 190, 60), 2)

    out = RESULTS_DIR / f"pair_{map_a}_{map_b}_grid.png"
    save_image(out, grid)
    print(f"  Saved grid: {out.name}")


def _trim_contact_edge(img: np.ndarray, edge: str,
                       ink_threshold: int = 200,
                       min_ink_frac: float = 0.003) -> np.ndarray:
    """
    Remove white scan border from the side of img that will contact the other map.
    Scans inward from that edge until it finds a row/col with enough ink,
    then returns the image trimmed to that point.
    """
    binary = (img < ink_threshold).astype(np.float32)
    h, w = img.shape

    if edge == 'bottom':
        density = binary.mean(axis=1)          # per-row ink density
        for i in range(h - 1, h // 2, -1):
            if density[i] >= min_ink_frac:
                return img[:i + 1]
    elif edge == 'top':
        density = binary.mean(axis=1)
        for i in range(0, h // 2):
            if density[i] >= min_ink_frac:
                return img[i:]
    elif edge == 'right':
        density = binary.mean(axis=0)          # per-col ink density
        for i in range(w - 1, w // 2, -1):
            if density[i] >= min_ink_frac:
                return img[:, :i + 1]
    elif edge == 'left':
        density = binary.mean(axis=0)
        for i in range(0, w // 2):
            if density[i] >= min_ink_frac:
                return img[:, i:]
    return img


def _save_best_preview(map_a: str, map_b: str,
                       img_a: np.ndarray, img_b: np.ndarray,
                       best: dict):
    """
    Stitch preview: maps placed flush against each other with no gap.
    Both images are trimmed at their contact edge to remove white scan
    borders, then rescaled to match along the shared boundary axis.
    """
    TARGET = 900   # max dimension for display
    img_b_rot = rotate_image(img_b, best["total_rot"])

    edge_a = best["edge_a"]
    edge_b = best["edge_b"]

    # Trim white scan borders from the contact edges before compositing
    img_a_trim     = _trim_contact_edge(img_a,     edge_a)
    img_b_rot_trim = _trim_contact_edge(img_b_rot, edge_b)

    scale = min(TARGET / img_a_trim.shape[1],     TARGET / img_a_trim.shape[0],
                TARGET / img_b_rot_trim.shape[1], TARGET / img_b_rot_trim.shape[0],
                0.3)

    da = cv2.resize(img_a_trim,     (int(img_a_trim.shape[1]*scale),
                                      int(img_a_trim.shape[0]*scale)),
                    interpolation=cv2.INTER_AREA)
    db = cv2.resize(img_b_rot_trim, (int(img_b_rot_trim.shape[1]*scale),
                                      int(img_b_rot_trim.shape[0]*scale)),
                    interpolation=cv2.INTER_AREA)

    # Match the shared boundary axis so both halves are the same size
    if edge_a in ('right', 'left'):
        target_h = min(da.shape[0], db.shape[0])
        if da.shape[0] != target_h:
            da = cv2.resize(da, (int(da.shape[1]*target_h/da.shape[0]), target_h),
                            interpolation=cv2.INTER_AREA)
        if db.shape[0] != target_h:
            db = cv2.resize(db, (int(db.shape[1]*target_h/db.shape[0]), target_h),
                            interpolation=cv2.INTER_AREA)
    else:
        target_w = min(da.shape[1], db.shape[1])
        if da.shape[1] != target_w:
            da = cv2.resize(da, (target_w, int(da.shape[0]*target_w/da.shape[1])),
                            interpolation=cv2.INTER_AREA)
        if db.shape[1] != target_w:
            db = cv2.resize(db, (target_w, int(db.shape[0]*target_w/db.shape[1])),
                            interpolation=cv2.INTER_AREA)

    da_bgr = cv2.cvtColor(da, cv2.COLOR_GRAY2BGR)
    db_bgr = cv2.cvtColor(db, cv2.COLOR_GRAY2BGR)
    da_f   = da_bgr.astype(np.float32)
    db_f   = db_bgr.astype(np.float32)

    ha, wa = da.shape;  hb, wb = db.shape

    # ------------------------------------------------------------------
    # VERSION 1 — Split: maps side-by-side with a thin orange seam
    # ------------------------------------------------------------------
    GAP = 3
    if edge_a in ('right', 'left'):
        split = np.empty((ha, wa + GAP + wb, 3), dtype=np.uint8)
        if edge_a == 'right':
            split[:, :wa]       = da_bgr
            split[:, wa:wa+GAP] = (255, 140, 0)
            split[:, wa+GAP:]   = db_bgr
        else:
            split[:, :wb]       = db_bgr
            split[:, wb:wb+GAP] = (255, 140, 0)
            split[:, wb+GAP:]   = da_bgr
    else:
        split = np.empty((ha + GAP + hb, wa, 3), dtype=np.uint8)
        if edge_a == 'bottom':
            split[:ha]              = da_bgr
            split[ha:ha+GAP]        = (255, 140, 0)
            split[ha+GAP:ha+GAP+hb] = db_bgr
        else:
            split[:hb]              = db_bgr
            split[hb:hb+GAP]        = (255, 140, 0)
            split[hb+GAP:hb+GAP+ha] = da_bgr

    # ------------------------------------------------------------------
    # VERSION 2 — Blend: overlap zone with linear transparency gradient
    # ------------------------------------------------------------------
    if edge_a in ('right', 'left'):
        OVR = max(20, int(min(wa, wb) * 0.12))
    else:
        OVR = max(20, int(min(ha, hb) * 0.12))

    ramp = np.linspace(0.0, 1.0, OVR, dtype=np.float32)

    if edge_a in ('right', 'left'):
        blend = np.zeros((ha, wa + wb - OVR, 3), dtype=np.float32)
        if edge_a == 'right':
            blend[:, :wa-OVR]      = da_f[:, :wa-OVR]
            alpha = ramp[np.newaxis, :, np.newaxis]
            blend[:, wa-OVR:wa]    = (1-alpha)*da_f[:, wa-OVR:wa] + alpha*db_f[:, :OVR]
            blend[:, wa:]          = db_f[:, OVR:]
        else:
            blend[:, :wb-OVR]      = db_f[:, :wb-OVR]
            alpha = ramp[np.newaxis, :, np.newaxis]
            blend[:, wb-OVR:wb]    = (1-alpha)*db_f[:, wb-OVR:wb] + alpha*da_f[:, :OVR]
            blend[:, wb:]          = da_f[:, OVR:]
    else:
        blend = np.zeros((ha + hb - OVR, wa, 3), dtype=np.float32)
        if edge_a == 'bottom':
            blend[:ha-OVR]         = da_f[:ha-OVR]
            alpha = ramp[:, np.newaxis, np.newaxis]
            blend[ha-OVR:ha]       = (1-alpha)*da_f[ha-OVR:ha] + alpha*db_f[:OVR]
            blend[ha:]             = db_f[OVR:]
        else:
            blend[:hb-OVR]         = db_f[:hb-OVR]
            alpha = ramp[:, np.newaxis, np.newaxis]
            blend[hb-OVR:hb]       = (1-alpha)*db_f[hb-OVR:hb] + alpha*da_f[:OVR]
            blend[hb:]             = da_f[OVR:]

    blend = np.clip(blend, 0, 255).astype(np.uint8)

    # ------------------------------------------------------------------
    # Build a shared header (same text for both files)
    # ------------------------------------------------------------------
    def _make_header(w):
        h = np.full((86, w, 3), 22, dtype=np.uint8)
        cv2.putText(h,
                    f"BEST MATCH — Map {map_a} ({best['edge_a']}) vs "
                    f"Map {map_b} ({best['edge_b']})   rot={best['total_rot']:.1f}",
                    (12, 32), FONT, 0.68, (70, 210, 70), 2, cv2.LINE_AA)
        cv2.putText(h,
                    f"score={best['combined_score']:.4f}   error={best['error']:.4f}   "
                    f"phase={best['phase_score']:.4f}   ncc2d={best['ncc_2d']:.4f}   "
                    f"offset={best['best_offset_px']:+d}px",
                    (12, 60), FONT, 0.52, (170, 170, 170), 1, cv2.LINE_AA)
        return h

    # ------------------------------------------------------------------
    # FILE 1 — pair_<a>_<b>_best.png   (split, orange seam line)
    # ------------------------------------------------------------------
    out_split = RESULTS_DIR / f"pair_{map_a}_{map_b}_best.png"
    save_image(out_split, np.vstack([_make_header(split.shape[1]), split]))
    print(f"  Saved split preview : {out_split.name}")

    # ------------------------------------------------------------------
    # FILE 2 — pair_<a>_<b>_bestwithoutspacing.png  (blend, transparent)
    # ------------------------------------------------------------------
    out_blend = RESULTS_DIR / f"pair_{map_a}_{map_b}_bestwithoutspacing.png"
    save_image(out_blend, np.vstack([_make_header(blend.shape[1]), blend]))
    print(f"  Saved blend preview : {out_blend.name}")


# ---------------------------------------------------------------------------
# STEP-BY-STEP DEBUG OUTPUT  (image-based — no abstract charts)
# ---------------------------------------------------------------------------

def _save_step_images(map_a: str, map_b: str,
                      img_a: np.ndarray, img_b: np.ndarray,
                      best: dict):
    """
    Save one PNG per pipeline stage showing the ACTUAL map/strip images
    at each stage with annotations drawn directly on them.

      step0_autocrop.png  — raw map with crop boundary drawn → cropped result
      step1_strips.png    — both maps with strip region highlighted + zoomed strip
      step2_profile.png   — strip + ink-intensity bars drawn alongside each row/col
      step3_align.png     — BEFORE/AFTER: strips with parcel-line markers showing alignment
      step4_ncc2d.png     — aligned strips with ink pixels coloured to show texture match
      step5_density.png   — strips with ink pixels highlighted + density labels
      step6_score.png     — both maps with winning contact edge drawn + score
    """
    edge_a  = best["edge_a"]
    edge_b  = best["edge_b"]
    offset  = best["best_offset_px"]
    PAD     = 14
    LC      = (175, 175, 175)
    P_COLS  = [(0, 60, 255), (0, 210, 255), (50, 220, 50), (255, 130, 0), (200, 0, 230)]

    img_b_rot = rotate_image(img_b, best["total_rot"])
    strip_a   = extract_strip(img_a,     edge_a)
    strip_b   = extract_strip(img_b_rot, edge_b)
    is_vert   = edge_a in ('left', 'right')

    # ---- display window of the strip (first DISP px along long axis) ----
    DISP = 460
    if is_vert:
        h_w = min(DISP, strip_a.shape[0])
        sa_win = strip_a[:h_w, :]
        sb_win = strip_b[:h_w, :]
    else:
        w_w = min(DISP, strip_a.shape[1])
        sa_win = strip_a[:, :w_w]
        sb_win = strip_b[:, :w_w]

    # ---- shared helpers ----
    def _hstack(*imgs, gap=PAD):
        mh = max(i.shape[0] for i in imgs)
        parts = []
        for k, img in enumerate(imgs):
            if img.shape[0] < mh:
                p = np.full((mh - img.shape[0], img.shape[1], 3), 22, dtype=np.uint8)
                img = np.vstack([img, p])
            parts.append(img)
            if k < len(imgs) - 1:
                parts.append(np.full((mh, gap, 3), 22, dtype=np.uint8))
        return np.hstack(parts)

    def _vstack(*imgs, gap=PAD):
        mw = max(i.shape[1] for i in imgs)
        parts = []
        for k, img in enumerate(imgs):
            if img.shape[1] < mw:
                p = np.full((img.shape[0], mw - img.shape[1], 3), 22, dtype=np.uint8)
                img = np.hstack([img, p])
            parts.append(img)
            if k < len(imgs) - 1:
                parts.append(np.full((gap, mw, 3), 22, dtype=np.uint8))
        return np.vstack(parts)

    def _top_bar(img_bgr, text, bg=(35, 55, 35)):
        bar = np.full((24, img_bgr.shape[1], 3), bg, dtype=np.uint8)
        cv2.putText(bar, text, (4, 17), FONT, 0.38, (215, 215, 215), 1, cv2.LINE_AA)
        return np.vstack([bar, img_bgr])

    def _hdr(text, w, sub=""):
        h = 58 if sub else 40
        c = np.full((h, w, 3), 22, dtype=np.uint8)
        cv2.putText(c, text, (8, 28), FONT, 0.60, (70, 210, 70), 2, cv2.LINE_AA)
        if sub:
            cv2.putText(c, sub, (8, 50), FONT, 0.41, LC, 1, cv2.LINE_AA)
        return c

    def _pad_h(img, h):
        if img.shape[0] >= h:
            return img
        return np.vstack([img, np.full((h - img.shape[0], img.shape[1], 3), 22, dtype=np.uint8)])

    def _peaks(prof, n=5, min_dist=25):
        result = []
        for idx in np.argsort(prof)[::-1]:
            idx = int(idx)
            if all(abs(idx - p) >= min_dist for p in result):
                result.append(idx)
            if len(result) >= n:
                break
        return sorted(result)

    # =================================================================
    # STEP 0 — Auto-crop: raw image with crop box drawn → cropped result
    # =================================================================
    raw_a = read_image(PREPROCESSED_DIR / f"map_{map_a}_clean.png")
    raw_b = read_image(PREPROCESSED_DIR / f"map_{map_b}_clean.png")

    def _crop_panel(raw, cropped, label, max_dim=310):
        """Raw thumbnail with green crop rectangle, arrow, cropped thumbnail."""
        s  = max_dim / max(raw.shape)
        rt = cv2.resize(raw,     (int(raw.shape[1]*s),     int(raw.shape[0]*s)),     interpolation=cv2.INTER_AREA)
        ct = cv2.resize(cropped, (int(cropped.shape[1]*s), int(cropped.shape[0]*s)), interpolation=cv2.INTER_AREA)
        rt_bgr = cv2.cvtColor(rt, cv2.COLOR_GRAY2BGR)
        ct_bgr = cv2.cvtColor(ct, cv2.COLOR_GRAY2BGR)
        # draw green rectangle showing what gets kept
        ink = raw < 200
        rows = np.where(ink.mean(axis=1) >= 0.001)[0]
        cols = np.where(ink.mean(axis=0) >= 0.001)[0]
        if len(rows) and len(cols):
            y0 = int(max(0, rows[0]  - 30) * s);  y1 = int(min(raw.shape[0], rows[-1] + 30) * s)
            x0 = int(max(0, cols[0]  - 30) * s);  x1 = int(min(raw.shape[1], cols[-1] + 30) * s)
            cv2.rectangle(rt_bgr, (x0, y0), (x1, y1), (0, 220, 0), 2)
        # arrow between raw and cropped
        mh   = max(rt_bgr.shape[0], ct_bgr.shape[0])
        arr  = np.full((mh, 36, 3), 22, dtype=np.uint8)
        cy   = mh // 2
        cv2.arrowedLine(arr, (4, cy), (32, cy), (70, 210, 70), 2, tipLength=0.5)
        rt_bgr = _pad_h(rt_bgr, mh);  ct_bgr = _pad_h(ct_bgr, mh)
        panel  = np.hstack([rt_bgr, arr, ct_bgr])
        return _top_bar(panel, label)

    pa = _crop_panel(raw_a, img_a, f"Map {map_a}: {raw_a.shape[1]}x{raw_a.shape[0]} -> {img_a.shape[1]}x{img_a.shape[0]} px")
    pb = _crop_panel(raw_b, img_b, f"Map {map_b}: {raw_b.shape[1]}x{raw_b.shape[0]} -> {img_b.shape[1]}x{img_b.shape[0]} px")
    body = _hstack(pa, pb, gap=28)
    out0 = _vstack(_hdr("STEP 0  Auto-crop: white scanner borders removed (green box = region kept)", body.shape[1]), body, gap=4)
    save_image(RESULTS_DIR / f"pair_{map_a}_{map_b}_step0_autocrop.png", out0)
    print(f"  Saved step 0 : pair_{map_a}_{map_b}_step0_autocrop.png")

    # =================================================================
    # STEP 1 — Strip extraction: map thumbnail + strip highlighted + zoom
    # =================================================================
    MAX_DIM = 300

    def _map_strip_panel(img, edge, sc, box_col, strip_win, map_lbl, strip_lbl):
        """Map thumbnail with strip box drawn + zoomed strip alongside."""
        t    = cv2.resize(img, (int(img.shape[1]*sc), int(img.shape[0]*sc)), interpolation=cv2.INTER_AREA)
        t_b  = cv2.cvtColor(t, cv2.COLOR_GRAY2BGR)
        sw   = max(2, int(STRIP_WIDTH * sc))
        h, w = t_b.shape[:2]
        if edge == 'right':  cv2.rectangle(t_b, (w-sw, 0),    (w-1, h-1),  box_col, 2)
        elif edge == 'left': cv2.rectangle(t_b, (0, 0),        (sw-1, h-1), box_col, 2)
        elif edge == 'top':  cv2.rectangle(t_b, (0, 0),        (w-1, sw-1), box_col, 2)
        else:                cv2.rectangle(t_b, (0, h-sw),     (w-1, h-1),  box_col, 2)
        t_b  = _top_bar(t_b, map_lbl)
        # zoomed strip (long axis capped at 350 px, keep 200 px width)
        sw_h, sw_w = strip_win.shape
        if is_vert:
            z_h = min(350, sw_h); z = strip_win[:z_h, :]
        else:
            z_w = min(350, sw_w); z = strip_win[:, :z_w]
        z_b = cv2.cvtColor(z, cv2.COLOR_GRAY2BGR)
        z_b = _top_bar(z_b, strip_lbl)
        return _hstack(t_b, z_b, gap=10)

    sc_a = MAX_DIM / max(img_a.shape)
    sc_b = MAX_DIM / max(img_b_rot.shape)
    panel_a = _map_strip_panel(img_a,     edge_a, sc_a, (0, 220, 0),   sa_win,
                                f"Map {map_a}  (green box = {edge_a} strip)",
                                f"Strip zoom  {edge_a}  {strip_a.shape[1]}x{strip_a.shape[0]}px")
    panel_b = _map_strip_panel(img_b_rot, edge_b, sc_b, (0, 110, 255), sb_win,
                                f"Map {map_b} rot {best['total_rot']:.1f}deg  (blue box = {edge_b} strip)",
                                f"Strip zoom  {edge_b}  {strip_b.shape[1]}x{strip_b.shape[0]}px")
    body = _hstack(panel_a, panel_b, gap=28)
    sub  = f"200 px slice extracted from each contact edge — this region is compared between the two maps"
    out1 = _vstack(_hdr(f"STEP 1  Strip extraction: A:{edge_a}  B:{edge_b}  rot={best['total_rot']:.1f}deg", body.shape[1], sub), body, gap=4)
    save_image(RESULTS_DIR / f"pair_{map_a}_{map_b}_step1_strips.png", out1)
    print(f"  Saved step 1 : pair_{map_a}_{map_b}_step1_strips.png")

    # =================================================================
    # STEP 2 — Boundary profile: ink-intensity bars drawn next to the strip
    # Each row/col gets a bar whose length = average ink at that position.
    # =================================================================
    prof_a = boundary_profile(sa_win, edge_a)
    prof_b = boundary_profile(sb_win, edge_b)

    def _strip_with_bars(strip_win, prof, edge, map_lbl, bar_col):
        """Strip image + per-row/col ink bars drawn as a side panel."""
        s_bgr = cv2.cvtColor(strip_win, cv2.COLOR_GRAY2BGR)
        norm  = prof / (prof.max() + 1e-6)
        BAR_W = 90
        if edge in ('left', 'right'):
            h, w = s_bgr.shape[:2]
            bar  = np.full((h, BAR_W, 3), 35, dtype=np.uint8)
            for r in range(h):
                length = int(norm[r] * (BAR_W - 6))
                if length > 0:
                    cv2.line(bar, (0, r), (length, r), bar_col, 1)
            # label bar axis
            cv2.putText(bar, "ink", (2, 12), FONT, 0.32, (140, 140, 140), 1, cv2.LINE_AA)
            combined = np.hstack([s_bgr, bar])
        else:
            h, w = s_bgr.shape[:2]
            bar  = np.full((BAR_W, w, 3), 35, dtype=np.uint8)
            for c_idx in range(min(w, len(norm))):
                length = int(norm[c_idx] * (BAR_W - 6))
                if length > 0:
                    cv2.line(bar, (c_idx, BAR_W - 1), (c_idx, BAR_W - 1 - length), bar_col, 1)
            cv2.putText(bar, "ink", (2, BAR_W - 4), FONT, 0.32, (140, 140, 140), 1, cv2.LINE_AA)
            combined = np.vstack([s_bgr, bar])
        return _top_bar(combined, map_lbl)

    pa2 = _strip_with_bars(sa_win, prof_a, edge_a,
                            f"Map {map_a}  {edge_a} strip  |  green bars = ink per row", (80, 220, 80))
    pb2 = _strip_with_bars(sb_win, prof_b, edge_b,
                            f"Map {map_b}  {edge_b} strip  |  blue bars = ink per row", (80, 150, 255))
    body = _hstack(pa2, pb2, gap=28)
    sub  = "Each bar = average ink in that row (inverted). Tall bar = parcel line crossing. Bars must look similar for a correct pair."
    out2 = _vstack(_hdr("STEP 2  Marginal projection: compress each strip row to one ink value", body.shape[1], sub), body, gap=4)
    save_image(RESULTS_DIR / f"pair_{map_a}_{map_b}_step2_profile.png", out2)
    print(f"  Saved step 2 : pair_{map_a}_{map_b}_step2_profile.png")

    # =================================================================
    # STEP 3 — Alignment: BEFORE (lines don't match) vs AFTER (they do)
    # Colored lines mark parcel crossings. Same color = same parcel line.
    # =================================================================
    peaks_a = _peaks(prof_a)

    def _draw_peak_lines(strip_win, peaks, edge, colors, shift=0):
        """Coloured lines at parcel crossing positions on a BGR strip copy."""
        s = cv2.cvtColor(strip_win, cv2.COLOR_GRAY2BGR)
        h, w = s.shape[:2]
        for ci, pk in enumerate(peaks):
            col = colors[ci % len(colors)]
            pos = pk + shift
            if edge in ('left', 'right'):
                if 0 <= pos < h:
                    cv2.line(s, (0, pos), (w - 1, pos), col, 2)
            else:
                if 0 <= pos < w:
                    cv2.line(s, (pos, 0), (pos, h - 1), col, 2)
        return s

    # BEFORE: lines in strip_a at peaks, lines in strip_b shifted by -offset
    # (meaning they don't sit at the same visual position → misaligned)
    sa_bf = _draw_peak_lines(sa_win, peaks_a, edge_a, P_COLS, shift=0)
    sb_bf = _draw_peak_lines(sb_win, peaks_a, edge_b, P_COLS, shift=-offset)
    sa_bf = _top_bar(sa_bf, f"Map {map_a}  ({edge_a})  parcel lines marked")
    sb_bf = _top_bar(sb_bf, f"Map {map_b}  ({edge_b})  same colours at WRONG position (offset not applied)", bg=(70, 30, 30))
    before_row = _hstack(sa_bf, sb_bf, gap=18)
    lbl_before = np.full((26, before_row.shape[1], 3), (60, 28, 28), dtype=np.uint8)
    cv2.putText(lbl_before, f"BEFORE alignment  —  offset = {offset:+d} px not yet applied  (lines do not align across maps)",
                (8, 18), FONT, 0.46, (200, 100, 100), 1, cv2.LINE_AA)
    before_block = np.vstack([lbl_before, before_row])

    # AFTER: use the aligned portions of both full strips
    if offset >= 0:
        av = min(DISP, strip_a.shape[0] - offset) if is_vert else min(DISP, strip_a.shape[1] - offset)
        av = max(av, 20)
        sa_af_win = (strip_a[offset:offset+av, :] if is_vert else strip_a[:, offset:offset+av])
        sb_af_win = (strip_b[:av, :]              if is_vert else strip_b[:, :av])
    else:
        av = min(DISP, strip_b.shape[0] + offset) if is_vert else min(DISP, strip_b.shape[1] + offset)
        av = max(av, 20)
        sa_af_win = (strip_a[:av, :]              if is_vert else strip_a[:, :av])
        sb_af_win = (strip_b[-offset:-offset+av, :] if is_vert else strip_b[:, -offset:-offset+av])

    # safety fallback
    if sa_af_win.shape[0] < 10 or sa_af_win.shape[1] < 10 or \
       sb_af_win.shape[0] < 10 or sb_af_win.shape[1] < 10:
        sa_af_win, sb_af_win = sa_win, sb_win

    peaks_af = _peaks(boundary_profile(sa_af_win, edge_a))
    sa_af = _draw_peak_lines(sa_af_win, peaks_af, edge_a, P_COLS, shift=0)
    sb_af = _draw_peak_lines(sb_af_win, peaks_af, edge_b, P_COLS, shift=0)
    sa_af = _top_bar(sa_af, f"Map {map_a}  ({edge_a})  parcel lines marked", bg=(28, 60, 28))
    sb_af = _top_bar(sb_af, f"Map {map_b}  ({edge_b})  same colours NOW ALIGNED  (offset applied)", bg=(28, 40, 65))
    after_row = _hstack(sa_af, sb_af, gap=18)
    lbl_after = np.full((26, after_row.shape[1], 3), (28, 55, 28), dtype=np.uint8)
    cv2.putText(lbl_after, f"AFTER alignment  —  offset = {offset:+d} px applied  (lines now align — parcel boundaries continue across maps)",
                (8, 18), FONT, 0.46, (100, 200, 100), 1, cv2.LINE_AA)
    after_block = np.vstack([lbl_after, after_row])

    body = _vstack(before_block, after_block, gap=18)
    sub  = "Coloured lines = parcel boundary crossings found in Map A. Same colours on Map B show where those lines appear."
    out3 = _vstack(_hdr("STEP 3  Phase correlation: find pixel shift so parcel lines align between both maps", body.shape[1], sub), body, gap=4)
    save_image(RESULTS_DIR / f"pair_{map_a}_{map_b}_step3_align.png", out3)
    print(f"  Saved step 3 : pair_{map_a}_{map_b}_step3_align.png")

    # =================================================================
    # STEP 4 — 2D NCC: aligned strips with ink pixels coloured
    # Green = Map A ink,  Blue = Map B ink.  They should overlap at seam.
    # =================================================================
    INK_T = 180

    def _ink_highlight(strip_win, ink_col):
        b = cv2.cvtColor(strip_win, cv2.COLOR_GRAY2BGR)
        b[strip_win < INK_T] = ink_col
        return b

    sa_ncc = _ink_highlight(sa_af_win, (50, 210, 50))
    sb_ncc = _ink_highlight(sb_af_win, (60, 110, 230))
    sa_ncc = _top_bar(sa_ncc, f"Map {map_a}  ({edge_a})  ink pixels = green", bg=(28, 60, 28))
    sb_ncc = _top_bar(sb_ncc, f"Map {map_b}  ({edge_b})  ink pixels = blue  (offset applied)", bg=(28, 40, 65))
    seam   = np.full((max(sa_ncc.shape[0], sb_ncc.shape[0]), 6, 3), (255, 140, 0), dtype=np.uint8)
    sa_ncc = _pad_h(sa_ncc, seam.shape[0]);  sb_ncc = _pad_h(sb_ncc, seam.shape[0])
    body   = np.hstack([sa_ncc, seam, sb_ncc])
    score_bar = np.full((30, body.shape[1], 3), 30, dtype=np.uint8)
    cv2.putText(score_bar,
                f"ncc_2d = {best['ncc_2d']:.4f}  (0 = no match, 1 = perfect match)   orange line = shared boundary   offset = {offset:+d} px",
                (8, 21), FONT, 0.48, LC, 1, cv2.LINE_AA)
    body = np.vstack([body, score_bar])
    sub  = "Coloured ink pixels must appear at the same positions on both sides of the orange seam for a high NCC score."
    out4 = _vstack(_hdr("STEP 4  2D NCC: full texture comparison — ink patterns must match across the seam", body.shape[1], sub), body, gap=4)
    save_image(RESULTS_DIR / f"pair_{map_a}_{map_b}_step4_ncc2d.png", out4)
    print(f"  Saved step 4 : pair_{map_a}_{map_b}_step4_ncc2d.png")

    # =================================================================
    # STEP 5 — Ink density: full strip window with all ink highlighted
    # =================================================================
    dens_a   = ink_density(strip_a)
    dens_b   = ink_density(strip_b)
    dens_sim = 1.0 - abs(dens_a - dens_b)

    da = _ink_highlight(sa_win, (50, 210, 50))
    db = _ink_highlight(sb_win, (60, 110, 230))
    da = _top_bar(da, f"Map {map_a}  {edge_a}  —  ink density = {dens_a*100:.2f}%  (green pixels)", bg=(28, 60, 28))
    db = _top_bar(db, f"Map {map_b}  {edge_b}  —  ink density = {dens_b*100:.2f}%  (blue pixels)", bg=(28, 40, 65))
    body = _hstack(da, db, gap=28)
    score_bar = np.full((30, body.shape[1], 3), 30, dtype=np.uint8)
    cv2.putText(score_bar,
                f"density_sim = 1 - |{dens_a:.4f} - {dens_b:.4f}| = {dens_sim:.4f}   (weight 0.10 in combined score)",
                (8, 21), FONT, 0.48, LC, 1, cv2.LINE_AA)
    body = np.vstack([body, score_bar])
    sub  = "Both edges should carry the same fraction of ink — similar line density confirms they are truly adjacent."
    out5 = _vstack(_hdr("STEP 5  Ink density: both contact edges must carry the same density of parcel lines", body.shape[1], sub), body, gap=4)
    save_image(RESULTS_DIR / f"pair_{map_a}_{map_b}_step5_density.png", out5)
    print(f"  Saved step 5 : pair_{map_a}_{map_b}_step5_density.png")

    # =================================================================
    # STEP 6 — Final score: both maps with winning edge + score breakdown
    # =================================================================
    def _map_with_edge(img, edge, max_dim=330, col=(70, 210, 70), thick=4):
        s = max_dim / max(img.shape)
        t = cv2.resize(img, (int(img.shape[1]*s), int(img.shape[0]*s)), interpolation=cv2.INTER_AREA)
        b = cv2.cvtColor(t, cv2.COLOR_GRAY2BGR)
        h, w = b.shape[:2]
        if edge == 'right':  cv2.line(b, (w-1, 0), (w-1, h), col, thick)
        elif edge == 'left': cv2.line(b, (0, 0),   (0, h),   col, thick)
        elif edge == 'top':  cv2.line(b, (0, 0),   (w, 0),   col, thick)
        else:                cv2.line(b, (0, h-1), (w, h-1), col, thick)
        return b

    ma = _map_with_edge(img_a,     edge_a, col=(70, 210, 70))
    mb = _map_with_edge(img_b_rot, edge_b, col=(70, 130, 255))
    ma = _top_bar(ma, f"Map {map_a}   contact edge: {edge_a}  (green line)")
    mb = _top_bar(mb, f"Map {map_b}   contact edge: {edge_b}  rot={best['total_rot']:.1f}deg  (blue line)")
    maps_row = _hstack(ma, mb, gap=28)

    score_h = 200
    sp = np.full((score_h, maps_row.shape[1], 3), 28, dtype=np.uint8)
    slines = [
        (f"Phase score  {best['phase_score']:.4f}  x 0.10  =  {best['phase_score']*0.10:.4f}", (100, 220, 100)),
        (f"2D NCC       {best['ncc_2d']:.4f}  x 0.80  =  {best['ncc_2d']*0.80:.4f}   <- primary", (100, 220, 100)),
        (f"Density sim  {best['density_sim']:.4f}  x 0.10  =  {best['density_sim']*0.10:.4f}", (100, 220, 100)),
        ("", (0, 0, 0)),
        (f"Combined score = {best['combined_score']:.4f}   Error = {best['error']:.4f}   Offset = {offset:+d} px", (70, 210, 70)),
    ]
    for i, (txt, col) in enumerate(slines):
        if txt:
            cv2.putText(sp, txt, (20, 38 + i * 34), FONT, 0.54, col, 1, cv2.LINE_AA)
    body = np.vstack([maps_row, sp])
    out6 = _vstack(_hdr(f"STEP 6  Winner: Map {map_a} ({edge_a}) <-> Map {map_b} ({edge_b})  rot={best['total_rot']:.1f}deg", body.shape[1]), body, gap=4)
    save_image(RESULTS_DIR / f"pair_{map_a}_{map_b}_step6_score.png", out6)
    print(f"  Saved step 6 : pair_{map_a}_{map_b}_step6_score.png")

    print(f"\n  All step images -> {RESULTS_DIR.resolve()}")


# ---------------------------------------------------------------------------
# RUN ALL PAIRS
# ---------------------------------------------------------------------------

def run_all(visualise: bool = False):
    print("\n" + "=" * 72)
    print("  EDGE ORIENTATION FINDER — ALL PAIRS")
    print("=" * 72)
    t0 = time.time()
    summary = {}

    for map_a, map_b in ADJACENT_PAIRS:
        result = analyse_pair(map_a, map_b, visualise=visualise)
        best   = result["best"]
        summary[f"{map_a}_{map_b}"] = {
            "edge_a":          best["edge_a"],
            "edge_b":          best["edge_b"],
            "total_rotation":  best["total_rot"],
            "combined_score":  best["combined_score"],
            "error":           best["error"],
            "offset_px":       best["best_offset_px"],
            "confidence_gap":  result["confidence_gap"],
        }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_DIR / "orientation_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 90)
    print("  SUMMARY — BEST ORIENTATION PER PAIR")
    print("=" * 90)
    print(f"  {'Pair':<10}  {'A edge':>7}  {'B edge':>7}  {'Rotation':>10}  "
          f"{'Score':>7}  {'Error':>7}  {'Gap':>7}  {'Offset':>8}")
    print(f"  {'-'*82}")
    for k, s in summary.items():
        print(f"  {k:<10}  {s['edge_a']:>7}  {s['edge_b']:>7}  "
              f"{s['total_rotation']:>9.1f}°  "
              f"{s['combined_score']:>7.4f}  "
              f"{s['error']:>7.4f}  "
              f"{s['confidence_gap']:>7.4f}  "
              f"{s['offset_px']:>+7d}px")
    print(f"\n  Total time : {time.time()-t0:.1f}s")
    print(f"  Results    : {RESULTS_DIR.resolve()}")
    print("=" * 90 + "\n")


# ---------------------------------------------------------------------------
# FULL DISCOVERY — all 78 unique pairs, best neighbour per map
# ---------------------------------------------------------------------------

def discover_all(visualise: bool = False):
    """
    Try every map against every other map (78 unique pairs for 13 maps).
    After scoring all pairs, rank the top candidates for each map and
    output an adjacency discovery report.
    """
    from itertools import combinations

    print("\n" + "=" * 72)
    print("  EDGE ORIENTATION FINDER — FULL DISCOVERY (ALL 78 PAIRS)")
    print(f"  Maps: {', '.join(ALL_MAPS)}")
    print("=" * 72)
    t0 = time.time()

    pairs = list(combinations(ALL_MAPS, 2))
    all_scores = {}

    for i, (map_a, map_b) in enumerate(pairs):
        print(f"\n[{i+1}/{len(pairs)}] ", end="", flush=True)
        try:
            result = analyse_pair(map_a, map_b, visualise=visualise)
            best   = result["best"]
            all_scores[f"{map_a}_{map_b}"] = {
                "map_a":          map_a,
                "map_b":          map_b,
                "edge_a":         best["edge_a"],
                "edge_b":         best["edge_b"],
                "total_rotation": best["total_rot"],
                "combined_score": best["combined_score"],
                "error":          best["error"],
                "offset_px":      best["best_offset_px"],
                "confidence_gap": result["confidence_gap"],
            }
        except Exception as e:
            print(f"  ERROR: {e}")
            all_scores[f"{map_a}_{map_b}"] = {"map_a": map_a, "map_b": map_b, "error_msg": str(e)}

    # ------------------------------------------------------------------
    # Save raw scores for all 78 pairs
    # ------------------------------------------------------------------
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_DIR / "discovery_all_scores.json", "w", encoding="utf-8") as f:
        json.dump(all_scores, f, indent=2)

    # ------------------------------------------------------------------
    # For each map, rank all 12 candidates by combined score
    # ------------------------------------------------------------------
    adjacency = {}
    for map_id in ALL_MAPS:
        candidates = []
        for key, s in all_scores.items():
            if "error_msg" in s:
                continue
            a, b = s["map_a"], s["map_b"]
            if a == map_id or b == map_id:
                other    = b if a == map_id else a
                my_edge  = s["edge_a"] if a == map_id else s["edge_b"]
                candidates.append({
                    "map":      other,
                    "score":    s["combined_score"],
                    "error":    s["error"],
                    "edge":     my_edge,
                    "rotation": s["total_rotation"],
                    "gap":      s["confidence_gap"],
                    "pair_key": key,
                })
        candidates.sort(key=lambda x: x["score"], reverse=True)
        adjacency[map_id] = candidates

    with open(RESULTS_DIR / "discovery_adjacency.json", "w", encoding="utf-8") as f:
        json.dump(adjacency, f, indent=2)

    # ------------------------------------------------------------------
    # Print adjacency table
    # ------------------------------------------------------------------
    print("\n\n" + "=" * 90)
    print("  ADJACENCY DISCOVERY — TOP 3 CANDIDATES PER MAP")
    print("=" * 90)
    print(f"  {'Map':<6}  {'Rank':<5}  {'Best match':<12}  {'Score':>7}  "
          f"{'Error':>7}  {'Edge':>7}  {'Rotation':>10}  {'Gap':>7}")
    print(f"  {'-'*82}")
    for map_id in ALL_MAPS:
        cands = adjacency[map_id]
        for rank, c in enumerate(cands[:3], 1):
            prefix = f"  {map_id:<6}" if rank == 1 else f"  {'':6}"
            print(f"{prefix}  #{rank:<4}  Map {c['map']:<8}  "
                  f"{c['score']:>7.4f}  {c['error']:>7.4f}  "
                  f"{c['edge']:>7}  {c['rotation']:>9.1f}°  "
                  f"{c['gap']:>7.4f}")
        print(f"  {'-'*82}")

    print(f"\n  Total time : {time.time()-t0:.1f}s")
    print(f"  Results    : {RESULTS_DIR.resolve()}")
    print(f"  Saved      : discovery_all_scores.json  +  discovery_adjacency.json")
    print("=" * 90 + "\n")


# ---------------------------------------------------------------------------
# REPORT — read saved discovery_adjacency.json and print best match per map
# ---------------------------------------------------------------------------

def print_report(top_n: int = 3):
    """
    Read the saved discovery_adjacency.json and print a clean ranked table.
    Call this after --discover has finished.
    """
    adj_path = RESULTS_DIR / "discovery_adjacency.json"
    if not adj_path.exists():
        print(f"ERROR: {adj_path} not found. Run --discover first.")
        return

    with open(adj_path, encoding="utf-8") as f:
        adjacency = json.load(f)

    print("\n" + "=" * 70)
    print("  BEST MATCH PER MAP  (from discovery_adjacency.json)")
    print("=" * 70)
    print(f"  {'Map':<6}  {'#':<3}  {'Best match':<12}  {'Score':>7}  "
          f"{'Error':>7}  {'Edge':>7}  {'Rotation':>10}")
    print(f"  {'-'*65}")

    for map_id, candidates in adjacency.items():
        for rank, c in enumerate(candidates[:top_n], 1):
            tag    = "  <-- BEST" if rank == 1 else ""
            prefix = f"  {map_id:<6}" if rank == 1 else f"  {'':6}"
            print(f"{prefix}  {rank:<3}  Map {c['map']:<8}  "
                  f"{c['score']:>7.4f}  {c['error']:>7.4f}  "
                  f"{c['edge']:>7}  {c['rotation']:>9.1f}{tag}")
        print(f"  {'-'*65}")

    print()


# ---------------------------------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fully automatic edge orientation finder — no manual input"
    )
    parser.add_argument("--pair",      type=str,       default=None,
                        help="Single pair, e.g. --pair 47_48")
    parser.add_argument("--all",       action="store_true",
                        help="Run on all 10 known adjacent pairs")
    parser.add_argument("--discover",  action="store_true",
                        help="Try all 78 pairs (13 maps x 12 others) and find best adjacency")
    parser.add_argument("--report",    action="store_true",
                        help="Print best-match-per-map from a finished --discover run")
    parser.add_argument("--top",       type=int,       default=3,
                        help="How many candidates to show per map in --report (default: 3)")
    parser.add_argument("--visualise", action="store_true",
                        help="Save grid PNG + stitch preview PNG")
    parser.add_argument("--steps",     action="store_true",
                        help="Save one debug PNG per pipeline stage (step0..step6)")
    parser.add_argument("--output",    type=str,       default=None,
                        help="Output folder (default: new_pipeline/data/orientation_results)")
    args = parser.parse_args()

    if args.output:
        RESULTS_DIR = Path(args.output)

    if args.pair:
        a, b = args.pair.split("_")
        analyse_pair(a, b, visualise=args.visualise, steps=args.steps)
    elif args.all:
        run_all(visualise=args.visualise)
    elif args.discover:
        discover_all(visualise=args.visualise)
    elif args.report:
        print_report(top_n=args.top)
    else:
        print("Usage:")
        print("  python new_pipeline/src/edge_orientation_finder.py --pair 47_48")
        print("  python new_pipeline/src/edge_orientation_finder.py --pair 47_48 --visualise")
        print("  python new_pipeline/src/edge_orientation_finder.py --all --visualise")
        print("  python new_pipeline/src/edge_orientation_finder.py --discover --visualise --output my_folder")
        print("  python new_pipeline/src/edge_orientation_finder.py --report --output my_folder")
        print("  python new_pipeline/src/edge_orientation_finder.py --report --top 5 --output my_folder")