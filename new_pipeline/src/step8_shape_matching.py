"""
=============================================================================
MILESTONE 6 — SHAPE-BASED PARCEL MATCHING (NO OCR REQUIRED)
=============================================================================
Masters Thesis - AI-Based Panoramic Cadastral Image Reconstruction
Author : Hussein Chalhoub

Purpose
--------
Automatically discover the boundary parcels shared between adjacent
cadastral atlas sheets, by matching parcel polygons by SHAPE alone.

Why shape matching, not OCR?
  - Each parcel polygon is unique within a map (user-confirmed property)
  - The same physical boundary parcel appears on BOTH adjacent sheets
    with the same shape (same survey, just continued onto another sheet)
  - Two parcels with identical area / perimeter / outline are almost
    certainly the same physical parcel that straddles the sheet boundary
  - This works regardless of handwriting legibility, ink fade, or scan
    quality — purely structural / geometric

Adjacency comes from metadata, NOT from manual control points
-------------------------------------------------------------
The 10 adjacent map pairs are derived from the CSV ground truth (same
parcel number listed on two maps -> those maps are adjacent). We use the
edge orientation (left/right/top/bottom) and relative rotation between
sheets that the user already recorded in `output/homographies/
homographies.json` — this is sheet-level metadata, not per-parcel
control-point clicking. The matching itself is fully automatic.

Algorithm (per adjacent pair A, B)
-----------------------------------
  1. Load Mask R-CNN polygons for each map (from step4)
  2. Filter to parcels within BOUNDARY_BAND_PX of the connecting edge
  3. Build a cost matrix where cost[i, j] = shape-distance(A[i], B[j])
       cost = cv2.matchShapes(...)  +  alpha * |log(area_a / area_b)|
  4. Hungarian (linear_sum_assignment) for optimal one-to-one matching
  5. Drop matches whose cost > MAX_MATCH_COST or area ratio is way off
  6. RANSAC homography fit to drop geometrically inconsistent matches
  7. Validation: compute residual vs the manual homography in
     homographies.json — small residuals = correct match

Output
-------
  new_pipeline/data/control_points/
    pair_<A>_<B>_matches.json     - confirmed shape matches per pair
    pair_<A>_<B>_overlay.png      - side-by-side visualisation

Usage
------
  .\\venv_thesis\\Scripts\\Activate.ps1

  # Test on a single pair
  python new_pipeline/src/step8_shape_matching.py --pair 45_47

  # Run on all 10 adjacent pairs
  python new_pipeline/src/step8_shape_matching.py --all
=============================================================================
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment


# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------

PREDICTIONS_DIR  = Path("new_pipeline/data/predictions")
HOMOGRAPHIES_PATH = Path("output/homographies/homographies.json")
PREPROCESSED_DIR  = Path("output/preprocessed")
OUTPUT_DIR        = Path("new_pipeline/data/control_points")

# 10 adjacent pairs derived from the CSV ground truth
ADJACENT_PAIRS = [
    ("45", "47"), ("45", "46"), ("47", "48"), ("48", "49"),
    ("49", "50"), ("50", "51"), ("52", "53"), ("52", "54"),
    ("52", "55"), ("54", "55"),
]

# CSV ground truth (for VALIDATION only — never used as algorithm input)
BOUNDARY_PARCELS_CSV = {
    2580: ["45", "47"],  2616: ["45", "46"],  2619: ["45", "46"],
    2749: ["47", "48"],  2803: ["47", "48"],  2814: ["47", "48"],
    2893: ["48", "49"],  3022: ["49", "50"],  3054: ["49", "50"],
    3068: ["50", "51"],  3215: ["52", "53"],  3216: ["52", "54"],
    3217: ["52", "55"],  3338: ["54", "55"],  3339: ["54", "55"],
    3345: ["54", "55"],  3346: ["54", "55"],  3813: ["50", "51"],
    3866: ["50", "51"],
}

# How close (in px) to the connecting edge must a parcel be to count as a
# boundary candidate. Maps are 6000-9000 px wide; 1000 px is ~13% inset.
BOUNDARY_BAND_PX = 1200

# Cost threshold for accepting a match. cv2.matchShapes(I1) returns 0 for
# identical contours; values <0.3 are usually true matches, >0.8 random.
MAX_MATCH_COST = 0.6

# Reject candidate pairs whose mask areas differ by more than this ratio
# (boundary parcels are the same physical land -> same area)
MAX_AREA_RATIO = 2.0

# Weight for area-ratio penalty in the cost function
AREA_PENALTY_WEIGHT = 0.3

# Drop tiny parcels (Mask R-CNN noise, road slivers)
MIN_PARCEL_AREA_PX = 800

# RANSAC reprojection threshold (px). Above this, a candidate is geometric
# noise — pinned by the manual homography validation.
RANSAC_THRESHOLD_PX = 300.0


# ---------------------------------------------------------------------------
# UTILITIES
# ---------------------------------------------------------------------------

def load_parcels(map_num: str) -> list[dict]:
    path = PREDICTIONS_DIR / f"map_{map_num}_parcels.json"
    if not path.exists():
        raise FileNotFoundError(f"Parcel JSON missing: {path}")
    with open(path, "r", encoding="utf-8") as f:
        parcels = json.load(f)
    # Keep only parcels with a usable polygon
    out = []
    for p in parcels:
        poly = p.get("polygon") or []
        if len(poly) < 3:
            continue
        if p.get("mask_area_px", 0) < MIN_PARCEL_AREA_PX:
            continue
        out.append(p)
    return out


def load_homography_meta() -> dict:
    if not HOMOGRAPHIES_PATH.exists():
        return {}
    with open(HOMOGRAPHIES_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def parcel_on_edge(parcel: dict, edge: str,
                   shape_hw: tuple[int, int],
                   band_px: int = BOUNDARY_BAND_PX) -> bool:
    """True if the parcel's centroid is within `band_px` of the named edge."""
    h, w = shape_hw
    cx, cy = parcel["cx"], parcel["cy"]
    if edge == "left":
        return cx <= band_px
    if edge == "right":
        return cx >= w - band_px
    if edge == "top":
        return cy <= band_px
    if edge == "bottom":
        return cy >= h - band_px
    return True


def polygon_to_contour(poly: list[list[int]]) -> np.ndarray:
    return np.array(poly, dtype=np.int32).reshape(-1, 1, 2)


def shape_cost(poly_a: list[list[int]], poly_b: list[list[int]],
               area_a: float, area_b: float) -> float:
    """
    Combined shape-distance: cv2.matchShapes (rotation/scale-invariant via
    Hu moments) + a log-area-ratio penalty so matched parcels also have
    similar physical size.
    """
    c_a = polygon_to_contour(poly_a)
    c_b = polygon_to_contour(poly_b)
    try:
        s = cv2.matchShapes(c_a, c_b, cv2.CONTOURS_MATCH_I1, 0.0)
    except cv2.error:
        return 1e9
    if not np.isfinite(s):
        return 1e9
    a_max = max(area_a, area_b)
    a_min = max(min(area_a, area_b), 1.0)
    area_pen = abs(np.log(a_max / a_min))
    return float(s) + AREA_PENALTY_WEIGHT * area_pen


# ---------------------------------------------------------------------------
# CORE MATCHING
# ---------------------------------------------------------------------------

def match_pair(map_a: str, map_b: str,
               homog_meta: dict | None) -> dict:
    """
    Run shape-based matching on one adjacent map pair.

    Returns dict with: matches, n_candidates_a, n_candidates_b,
    edge_a, edge_b, validation_residuals, n_inliers.
    """
    parcels_a = load_parcels(map_a)
    parcels_b = load_parcels(map_b)

    # Sheet metadata (edges, dimensions, rotation, manual H for validation)
    edge_a   = "right"
    edge_b   = "left"
    shape_a  = (6000, 8000)
    shape_b  = (6000, 8000)
    H_manual = None
    csv_pids: list[int] = []

    if homog_meta:
        edge_a = homog_meta.get("edge_a", edge_a)
        edge_b = homog_meta.get("edge_b", edge_b)
        sa     = homog_meta.get("shape_a")
        sb     = homog_meta.get("shape_b")
        if sa and len(sa) == 2:
            shape_a = (int(sa[0]), int(sa[1]))
        if sb and len(sb) == 2:
            shape_b = (int(sb[0]), int(sb[1]))
        if "H" in homog_meta:
            H_manual = np.array(homog_meta["H"], dtype=np.float64)
        csv_pids = list(homog_meta.get("boundary_parcels") or [])

    # Filter to boundary candidates near the connecting edges
    cands_a = [p for p in parcels_a
               if parcel_on_edge(p, edge_a, shape_a)]
    cands_b = [p for p in parcels_b
               if parcel_on_edge(p, edge_b, shape_b)]

    # Build cost matrix
    n_a = len(cands_a)
    n_b = len(cands_b)
    cost = np.full((n_a, n_b), 1e6, dtype=np.float32)
    for i, pa in enumerate(cands_a):
        area_a = float(pa.get("mask_area_px") or pa.get("area_px") or 1.0)
        for j, pb in enumerate(cands_b):
            area_b = float(pb.get("mask_area_px") or pb.get("area_px") or 1.0)
            ratio = max(area_a, area_b) / max(min(area_a, area_b), 1.0)
            if ratio > MAX_AREA_RATIO:
                continue
            cost[i, j] = shape_cost(pa["polygon"], pb["polygon"],
                                    area_a, area_b)

    # Hungarian assignment (works on rectangular cost matrices too)
    matches: list[dict] = []
    if n_a > 0 and n_b > 0:
        try:
            rows, cols = linear_sum_assignment(cost)
        except ValueError:
            rows, cols = np.array([]), np.array([])
        for r, c in zip(rows, cols):
            if cost[r, c] < MAX_MATCH_COST:
                pa, pb = cands_a[int(r)], cands_b[int(c)]
                matches.append({
                    "parcel_a": int(pa["parcel_id"]),
                    "parcel_b": int(pb["parcel_id"]),
                    "cx_a":     float(pa["cx"]),
                    "cy_a":     float(pa["cy"]),
                    "cx_b":     float(pb["cx"]),
                    "cy_b":     float(pb["cy"]),
                    "area_a":   float(pa.get("mask_area_px", 0)),
                    "area_b":   float(pb.get("mask_area_px", 0)),
                    "cost":     float(cost[r, c]),
                })

    # RANSAC geometric verification (only when we have enough points)
    inliers_mask = None
    H_auto = None
    if len(matches) >= 4:
        pts_a = np.array([[m["cx_a"], m["cy_a"]] for m in matches],
                          dtype=np.float32)
        pts_b = np.array([[m["cx_b"], m["cy_b"]] for m in matches],
                          dtype=np.float32)
        H_auto, mask = cv2.findHomography(
            pts_a, pts_b, cv2.RANSAC,
            ransacReprojThreshold=RANSAC_THRESHOLD_PX,
            maxIters=2000, confidence=0.99,
        )
        if mask is not None:
            inliers_mask = mask.flatten().astype(bool)
            matches = [m for m, ok in zip(matches, inliers_mask) if ok]

    # Sort surviving matches by cost (best first)
    matches.sort(key=lambda m: m["cost"])

    # ── Validation against the manual homography ────────────────────────
    val_residuals: list[float] = []
    if H_manual is not None and matches:
        pts_a = np.array(
            [[m["cx_a"], m["cy_a"]] for m in matches],
            dtype=np.float64,
        ).reshape(-1, 1, 2)
        pts_b_actual = np.array(
            [[m["cx_b"], m["cy_b"]] for m in matches],
            dtype=np.float64,
        ).reshape(-1, 1, 2)
        try:
            pts_b_pred = cv2.perspectiveTransform(pts_a, H_manual)
            res = np.linalg.norm(
                (pts_b_pred - pts_b_actual).reshape(-1, 2),
                axis=1,
            )
            val_residuals = res.tolist()
            for m, r in zip(matches, val_residuals):
                m["residual_vs_manual_H_px"] = round(float(r), 1)
        except cv2.error:
            pass

    return {
        "pair":              f"{map_a}_{map_b}",
        "edge_a":            edge_a,
        "edge_b":            edge_b,
        "n_candidates_a":    n_a,
        "n_candidates_b":    n_b,
        "csv_expected":      sorted(csv_pids),
        "n_matches":         len(matches),
        "matches":           matches,
        "H_auto":            H_auto.tolist() if H_auto is not None else None,
        "validation_mean_residual_px":
            float(np.mean(val_residuals)) if val_residuals else None,
        "validation_median_residual_px":
            float(np.median(val_residuals)) if val_residuals else None,
    }


# ---------------------------------------------------------------------------
# VISUALISATION
# ---------------------------------------------------------------------------

def visualise_pair(result: dict, out_path: Path):
    a, b = result["pair"].split("_")

    def load_scaled(m, target_w=1500):
        p = PREPROCESSED_DIR / f"map_{m}_clean.png"
        data = np.fromfile(str(p), dtype=np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if img is None:
            return None, 1.0
        scale = target_w / max(img.shape[:2])
        return cv2.resize(
            img,
            (int(img.shape[1] * scale), int(img.shape[0] * scale)),
        ), scale

    img_a, scale_a = load_scaled(a)
    img_b, scale_b = load_scaled(b)
    if img_a is None or img_b is None:
        return

    h = max(img_a.shape[0], img_b.shape[0])
    gap = 30
    canvas = np.ones((h + 60, img_a.shape[1] + gap + img_b.shape[1], 3),
                     dtype=np.uint8) * 255
    canvas[60:60 + img_a.shape[0], :img_a.shape[1]] = img_a
    canvas[60:60 + img_b.shape[0], img_a.shape[1] + gap:] = img_b
    offset_b = img_a.shape[1] + gap

    colors = [
        (0, 200, 0), (200, 0, 0), (0, 0, 200), (200, 130, 0),
        (130, 0, 200), (0, 200, 200), (200, 0, 130), (60, 130, 0),
    ]

    for k, m in enumerate(result["matches"]):
        col = colors[k % len(colors)]
        pa = (int(m["cx_a"] * scale_a),
              int(m["cy_a"] * scale_a) + 60)
        pb = (int(m["cx_b"] * scale_b) + offset_b,
              int(m["cy_b"] * scale_b) + 60)
        cv2.circle(canvas, pa, 10, col, -1)
        cv2.circle(canvas, pb, 10, col, -1)
        cv2.line(canvas, pa, pb, col, 2)
        label = f"{m['parcel_a']}<->{m['parcel_b']}"
        cv2.putText(canvas, label, (pa[0] + 12, pa[1] + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 2, cv2.LINE_AA)

    # Header
    cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 55), (30, 30, 30), -1)
    res_str = ""
    if result.get("validation_median_residual_px") is not None:
        res_str = (f"  |  median residual vs manual H: "
                   f"{result['validation_median_residual_px']:.0f} px")
    header = (f"Pair {result['pair']}  |  {result['n_matches']} shape "
              f"matches  (CSV expected {len(result['csv_expected'])})"
              f"{res_str}")
    cv2.putText(canvas, header, (10, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (255, 255, 255), 2, cv2.LINE_AA)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    ok, enc = cv2.imencode(".png", canvas)
    if ok:
        enc.tofile(str(out_path))


# ---------------------------------------------------------------------------
# RUN MODES
# ---------------------------------------------------------------------------

def save_result(result: dict) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    p = OUTPUT_DIR / f"pair_{result['pair']}_matches.json"
    with open(p, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    return p


def print_pair_summary(r: dict):
    print(f"\n  Pair {r['pair']}  edges {r['edge_a']}->{r['edge_b']}")
    print(f"    Candidates near boundary : {r['n_candidates_a']} on "
          f"{r['pair'].split('_')[0]}, "
          f"{r['n_candidates_b']} on {r['pair'].split('_')[1]}")
    print(f"    Confirmed shape matches  : {r['n_matches']}")
    if r["csv_expected"]:
        print(f"    CSV expected boundaries  : {r['csv_expected']}")
    if r["validation_median_residual_px"] is not None:
        print(f"    Validation vs manual H   : "
              f"median {r['validation_median_residual_px']:.0f} px, "
              f"mean {r['validation_mean_residual_px']:.0f} px")
        print(f"    (small residual = matches consistent with the user's "
              f"existing manual homography)")
    if r["matches"]:
        print(f"    Top matches by cost:")
        for m in r["matches"][:5]:
            ra = m.get("residual_vs_manual_H_px")
            ra_str = f"  resid={ra:.0f}px" if ra is not None else ""
            print(f"      parcel_a={m['parcel_a']:>4} <-> "
                  f"parcel_b={m['parcel_b']:>4}  "
                  f"cost={m['cost']:.3f}  "
                  f"area_a={int(m['area_a'])}/area_b={int(m['area_b'])}"
                  f"{ra_str}")


def run_pair(pair_key: str, no_vis: bool = False):
    a, b = pair_key.split("_")
    homog = load_homography_meta().get(pair_key)
    if homog is None:
        print(f"  No homography metadata for {pair_key} — using defaults")
    print(f"\n  Shape-matching pair {pair_key}...")
    result = match_pair(a, b, homog)
    save_result(result)
    print_pair_summary(result)
    if not no_vis:
        vis_path = OUTPUT_DIR / f"pair_{pair_key}_overlay.png"
        visualise_pair(result, vis_path)
        print(f"\n    Visualisation: {vis_path}")


def run_all(no_vis: bool = False):
    print("\n" + "=" * 70)
    print("  STEP 8 — SHAPE-BASED PARCEL MATCHING (no OCR)")
    print("=" * 70)
    homog_all = load_homography_meta()
    summary = []
    for a, b in ADJACENT_PAIRS:
        pair_key = f"{a}_{b}"
        result = match_pair(a, b, homog_all.get(pair_key))
        save_result(result)
        print_pair_summary(result)
        if not no_vis:
            visualise_pair(result,
                           OUTPUT_DIR / f"pair_{pair_key}_overlay.png")
        summary.append({
            "pair": pair_key,
            "n_matches": result["n_matches"],
            "csv_expected": len(result["csv_expected"]),
            "median_residual_px":
                result.get("validation_median_residual_px"),
        })

    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    print(f"  {'Pair':<8} {'Matches':>8} {'CSV exp':>8}  {'Median resid':>14}")
    for s in summary:
        rp = s["median_residual_px"]
        rp_str = f"{rp:>10.0f} px" if rp is not None else "         —"
        print(f"  {s['pair']:<8} {s['n_matches']:>8} "
              f"{s['csv_expected']:>8}  {rp_str}")
    print(f"\n  Output: {OUTPUT_DIR.resolve()}")
    print("=" * 70 + "\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Shape-based parcel matching (no OCR)."
    )
    parser.add_argument("--pair", type=str, default=None,
                        help="A single pair, e.g. --pair 45_47")
    parser.add_argument("--all", action="store_true",
                        help="Run on all 10 adjacent pairs")
    parser.add_argument("--no-vis", action="store_true",
                        help="Skip the side-by-side overlay PNGs")
    args = parser.parse_args()

    if args.pair:
        run_pair(args.pair, no_vis=args.no_vis)
    elif args.all:
        run_all(no_vis=args.no_vis)
    else:
        print("Usage:")
        print("  python new_pipeline/src/step8_shape_matching.py --pair 45_47")
        print("  python new_pipeline/src/step8_shape_matching.py --all")
