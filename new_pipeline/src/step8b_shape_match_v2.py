"""
=============================================================================
MILESTONE 6 (alt) — SHAPE MATCHING V2 WITH FULL DIAGNOSTICS
=============================================================================
Masters Thesis - AI-Based Panoramic Cadastral Image Reconstruction
Author : Hussein Chalhoub

Purpose
--------
Match the same physical parcel on two adjacent map sheets purely from
its polygon shape. No OCR, no user clicks, no manual control points.

Approach
---------
A boundary parcel that straddles two sheets appears on BOTH sheets with
the same physical shape — same area, same perimeter, same polygon
outline (only the position and rotation of the sheet differ). Hu moments
of the parcel polygon are a 7-dimensional rotation/scale-invariant
fingerprint of the shape.

Algorithm (per adjacent pair A, B):
  1. Load polygons of every parcel on A and on B (from step4 output).
  2. Drop parcels with degenerate polygons (< 4 vertices) or tiny area.
  3. For every parcel a in A, compute the cost to every parcel b in B:
        cost = log-Hu-moment distance  +  ALPHA * |log(area_a/area_b)|
     This is rotation- and translation-invariant because Hu moments are.
  4. For each a, keep its TOP-K cheapest candidates in B.
  5. Run RANSAC homography over those (a -> b) candidate centroids.
     RANSAC's job is to find the largest geometrically consistent subset
     among the noisy top-K candidates.
  6. The inlier set is the auto-discovered control point list.

Differences vs the v1 attempt
------------------------------
  - No edge-band filter: boundary parcels can extend deep into the sheet,
    so we consider every parcel (only an area/aspect prefilter applies).
  - Top-K candidates per parcel instead of one-to-one assignment first:
    one-to-one collapses too aggressively when shapes collide.
  - Full diagnostic prints: number of candidates, cost histogram,
    top matches, RANSAC inlier count + transform, sample residuals.

Usage
------
  python new_pipeline/src/step8b_shape_match_v2.py --pair 45_47
  python new_pipeline/src/step8b_shape_match_v2.py --all

Output
-------
  new_pipeline/data/control_points/
    pair_<A>_<B>_shape_v2.json
    pair_<A>_<B>_shape_v2.png      side-by-side visualisation
=============================================================================
"""

import argparse
import json
from collections import Counter
from pathlib import Path

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------

PREDICTIONS_DIR  = Path("new_pipeline/data/predictions")
PREPROCESSED_DIR = Path("output/preprocessed")
OUTPUT_DIR       = Path("new_pipeline/data/control_points")

# 10 known-adjacent pairs from CSV. Adjacency itself is metadata, not
# manual click work.
ADJACENT_PAIRS = [
    ("45", "47"), ("45", "46"), ("47", "48"), ("48", "49"),
    ("49", "50"), ("50", "51"), ("52", "53"), ("52", "54"),
    ("52", "55"), ("54", "55"),
]

# Drop tiny parcels (Mask R-CNN noise, road slivers).
MIN_PARCEL_AREA_PX = 1000

# Drop parcels whose log-area on A vs B differs by more than this.
# Boundary parcels are the same physical land -> same physical area.
MAX_LOG_AREA_DIFF = np.log(1.8)        # ~80% area mismatch tolerance

# Weight of the area-ratio penalty in the combined cost.
ALPHA_AREA = 0.4

# Each parcel on A keeps its TOP-K cheapest candidates in B for RANSAC.
TOP_K = 4

# RANSAC threshold (in pixels of map B's native frame).
RANSAC_THRESHOLD_PX = 400.0


# ---------------------------------------------------------------------------
# HU-MOMENT FINGERPRINT
# ---------------------------------------------------------------------------

def hu_log_signature(polygon: list[list[int]]) -> np.ndarray | None:
    """
    Return the 7 Hu moments in log-magnitude form (preserving sign),
    which behaves much more linearly than the raw moments.
    Hu moments are translation/rotation/scale invariant — exactly the
    invariance we want for cross-sheet polygon comparison.
    """
    pts = np.array(polygon, dtype=np.int32)
    if pts.ndim != 2 or pts.shape[0] < 4:
        return None
    contour = pts.reshape(-1, 1, 2)
    try:
        m = cv2.moments(contour)
    except cv2.error:
        return None
    if abs(m.get("m00", 0)) < 1e-6:
        return None
    hu = cv2.HuMoments(m).flatten()        # (7,)
    sig = np.sign(hu) * np.log10(np.abs(hu) + 1e-30)
    if not np.all(np.isfinite(sig)):
        return None
    return sig.astype(np.float64)


# ---------------------------------------------------------------------------
# DATA LOADING
# ---------------------------------------------------------------------------

def load_parcels(map_num: str) -> list[dict]:
    p = PREDICTIONS_DIR / f"map_{map_num}_parcels.json"
    with open(p, "r", encoding="utf-8") as f:
        parcels = json.load(f)

    out = []
    for x in parcels:
        poly = x.get("polygon") or []
        if len(poly) < 4:
            continue
        area = float(x.get("mask_area_px") or x.get("area_px") or 0.0)
        if area < MIN_PARCEL_AREA_PX:
            continue
        sig = hu_log_signature(poly)
        if sig is None:
            continue
        x["_hu"] = sig
        x["_area"] = area
        out.append(x)
    return out


# ---------------------------------------------------------------------------
# CORE
# ---------------------------------------------------------------------------

def shape_match_pair(map_a: str, map_b: str, verbose: bool = True) -> dict:
    parcels_a = load_parcels(map_a)
    parcels_b = load_parcels(map_b)
    n_a, n_b = len(parcels_a), len(parcels_b)

    if verbose:
        print(f"\n  Pair {map_a}_{map_b}")
        print(f"    Parcels with usable polygon: {n_a} on {map_a}, "
              f"{n_b} on {map_b}")

    if n_a == 0 or n_b == 0:
        return {"pair": f"{map_a}_{map_b}", "matches": [], "error": "no parcels"}

    # Stack Hu signatures and areas
    Hu_a = np.stack([p["_hu"] for p in parcels_a])    # (Na, 7)
    Hu_b = np.stack([p["_hu"] for p in parcels_b])    # (Nb, 7)
    log_area_a = np.log(np.array([p["_area"] for p in parcels_a]))
    log_area_b = np.log(np.array([p["_area"] for p in parcels_b]))

    # Pairwise cost matrix
    # shape_dist[i, j] = ||Hu_a[i] - Hu_b[j]||
    diff = Hu_a[:, None, :] - Hu_b[None, :, :]
    shape_dist = np.sqrt(np.sum(diff ** 2, axis=-1))   # (Na, Nb)

    # Area mismatch penalty
    area_diff = np.abs(log_area_a[:, None] - log_area_b[None, :])
    area_mask = area_diff <= MAX_LOG_AREA_DIFF
    cost = shape_dist + ALPHA_AREA * area_diff
    cost[~area_mask] = np.inf

    # Pick top-K cheapest candidates per row
    candidates: list[tuple[int, int, float]] = []   # (idx_a, idx_b, cost)
    for i in range(n_a):
        row = cost[i]
        if not np.any(np.isfinite(row)):
            continue
        finite_idx = np.where(np.isfinite(row))[0]
        ordered = finite_idx[np.argsort(row[finite_idx])]
        for j in ordered[:TOP_K]:
            candidates.append((i, int(j), float(row[j])))

    candidates.sort(key=lambda x: x[2])
    if verbose:
        print(f"    Candidate (a, b) pairs after top-{TOP_K} filter "
              f"and area gate: {len(candidates)}")
        if candidates:
            costs = np.array([c[2] for c in candidates])
            qs = np.percentile(costs, [10, 25, 50, 75, 90])
            print(f"    Cost distribution (q10/25/50/75/90): "
                  f"{qs[0]:.3f} / {qs[1]:.3f} / {qs[2]:.3f} / "
                  f"{qs[3]:.3f} / {qs[4]:.3f}")

    if len(candidates) < 4:
        if verbose:
            print("    Not enough candidates for RANSAC (need >=4).")
        return {
            "pair": f"{map_a}_{map_b}",
            "n_candidates": len(candidates),
            "matches": [],
        }

    # Build correspondence point arrays for RANSAC
    pts_a = np.array(
        [[parcels_a[i]["cx"], parcels_a[i]["cy"]] for i, _, _ in candidates],
        dtype=np.float32,
    )
    pts_b = np.array(
        [[parcels_b[j]["cx"], parcels_b[j]["cy"]] for _, j, _ in candidates],
        dtype=np.float32,
    )

    H, mask = cv2.findHomography(
        pts_a, pts_b, cv2.RANSAC,
        ransacReprojThreshold=RANSAC_THRESHOLD_PX,
        maxIters=4000, confidence=0.995,
    )

    if H is None or mask is None:
        if verbose:
            print("    RANSAC failed to find any consensus homography.")
        return {
            "pair": f"{map_a}_{map_b}",
            "n_candidates": len(candidates),
            "matches": [],
        }

    inliers = mask.flatten().astype(bool)
    n_in = int(inliers.sum())
    if verbose:
        print(f"    RANSAC inliers: {n_in} / {len(candidates)}")

    matches = []
    for k, ((i, j, c), is_in) in enumerate(zip(candidates, inliers)):
        if not is_in:
            continue
        matches.append({
            "parcel_a":   int(parcels_a[i]["parcel_id"]),
            "parcel_b":   int(parcels_b[j]["parcel_id"]),
            "cx_a":       float(parcels_a[i]["cx"]),
            "cy_a":       float(parcels_a[i]["cy"]),
            "cx_b":       float(parcels_b[j]["cx"]),
            "cy_b":       float(parcels_b[j]["cy"]),
            "cost":       round(c, 4),
            "area_a":     round(parcels_a[i]["_area"], 0),
            "area_b":     round(parcels_b[j]["_area"], 0),
        })
    matches.sort(key=lambda m: m["cost"])

    # Stats: what fraction of (a) are matched? Are any (a) matched many times?
    a_counts = Counter(m["parcel_a"] for m in matches)
    b_counts = Counter(m["parcel_b"] for m in matches)
    if verbose and matches:
        print(f"    Distinct A parcels matched: {len(a_counts)}")
        print(f"    Distinct B parcels matched: {len(b_counts)}")
        print(f"    Top 5 inlier matches:")
        for m in matches[:5]:
            print(f"      a={m['parcel_a']:>4}  <->  b={m['parcel_b']:>4}  "
                  f"cost={m['cost']:.3f}  "
                  f"area={int(m['area_a'])}/{int(m['area_b'])}")

    return {
        "pair":          f"{map_a}_{map_b}",
        "n_candidates":  len(candidates),
        "n_inliers":     n_in,
        "H_auto":        H.tolist(),
        "matches":       matches,
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
        s = target_w / max(img.shape[:2])
        return cv2.resize(img,
                          (int(img.shape[1] * s), int(img.shape[0] * s))), s

    img_a, s_a = load_scaled(a)
    img_b, s_b = load_scaled(b)
    if img_a is None or img_b is None:
        return

    h = max(img_a.shape[0], img_b.shape[0])
    gap = 30
    canvas = np.ones((h + 60, img_a.shape[1] + gap + img_b.shape[1], 3),
                     dtype=np.uint8) * 255
    canvas[60:60 + img_a.shape[0], :img_a.shape[1]] = img_a
    canvas[60:60 + img_b.shape[0], img_a.shape[1] + gap:] = img_b
    off_b = img_a.shape[1] + gap

    colours = [
        (0, 200, 0), (200, 0, 0), (0, 0, 200), (200, 130, 0),
        (130, 0, 200), (0, 200, 200), (200, 0, 130), (60, 130, 0),
    ]
    for k, m in enumerate(result.get("matches", [])):
        col = colours[k % len(colours)]
        pa = (int(m["cx_a"] * s_a), int(m["cy_a"] * s_a) + 60)
        pb = (int(m["cx_b"] * s_b) + off_b,
              int(m["cy_b"] * s_b) + 60)
        cv2.circle(canvas, pa, 8, col, -1)
        cv2.circle(canvas, pb, 8, col, -1)
        cv2.line(canvas, pa, pb, col, 2)
        label = f"{m['parcel_a']}<->{m['parcel_b']}"
        cv2.putText(canvas, label, (pa[0] + 10, pa[1] + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1, cv2.LINE_AA)

    cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 55), (30, 30, 30), -1)
    header = (f"Pair {result['pair']}  |  "
              f"{len(result.get('matches', []))} shape matches  "
              f"({result.get('n_candidates', 0)} candidates)")
    cv2.putText(canvas, header, (10, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (255, 255, 255), 2, cv2.LINE_AA)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    ok, enc = cv2.imencode(".png", canvas)
    if ok:
        enc.tofile(str(out_path))


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def save_result(result: dict) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    p = OUTPUT_DIR / f"pair_{result['pair']}_shape_v2.json"
    with open(p, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    return p


def run_pair(pair_key: str, no_vis: bool = False):
    a, b = pair_key.split("_")
    result = shape_match_pair(a, b, verbose=True)
    save_result(result)
    if not no_vis:
        visualise_pair(result,
                       OUTPUT_DIR / f"pair_{pair_key}_shape_v2.png")
        print(f"\n    Visualisation: "
              f"{OUTPUT_DIR / f'pair_{pair_key}_shape_v2.png'}")


def run_all(no_vis: bool = False):
    print("\n" + "=" * 70)
    print("  STEP 8b — SHAPE MATCHING v2 ON ALL ADJACENT PAIRS")
    print("=" * 70)
    summary = []
    for a, b in ADJACENT_PAIRS:
        try:
            result = shape_match_pair(a, b, verbose=True)
        except FileNotFoundError as e:
            print(f"\n  Skipping {a}_{b}: {e}")
            continue
        save_result(result)
        if not no_vis:
            visualise_pair(result,
                           OUTPUT_DIR / f"pair_{a}_{b}_shape_v2.png")
        summary.append({
            "pair":         f"{a}_{b}",
            "n_candidates": result.get("n_candidates", 0),
            "n_inliers":    result.get("n_inliers", 0),
        })

    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    print(f"  {'Pair':<8} {'Candidates':>12} {'RANSAC inliers':>16}")
    for s in summary:
        print(f"  {s['pair']:<8} {s['n_candidates']:>12} "
              f"{s['n_inliers']:>16}")
    print(f"\n  Output: {OUTPUT_DIR.resolve()}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pair", type=str, default=None)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--no-vis", action="store_true")
    args = parser.parse_args()

    if args.pair:
        run_pair(args.pair, no_vis=args.no_vis)
    elif args.all:
        run_all(no_vis=args.no_vis)
    else:
        print("Usage:")
        print("  python new_pipeline/src/step8b_shape_match_v2.py --pair 45_47")
        print("  python new_pipeline/src/step8b_shape_match_v2.py --all")
