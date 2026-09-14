"""
=============================================================================
MILESTONE 6 — CROSS-REFERENCE OCR READINGS, DEDUPLICATE, BUILD CONTROL POINTS
=============================================================================
Masters Thesis - AI-Based Panoramic Cadastral Image Reconstruction
Author : Hussein Chalhoub

Purpose
--------
Take Claude's per-parcel OCR readings (from step7_ocr_vision.py) and turn
them into reliable control-point matches between adjacent map sheets,
using the constraints:

  1. UNIQUENESS  — each parcel number is unique within a single map.
                   If Claude reports "2580" on multiple parcels of map A,
                   AT MOST ONE of those readings is correct.

  2. CROSS-MAP CONFIRMATION — a true boundary parcel appears on BOTH
                   adjacent sheets with the SAME number.

  3. GEOMETRIC CONSISTENCY — the boundary parcels of pair (A, B) all
                   transform consistently under a single homography
                   (RANSAC filters outliers).

Algorithm (per adjacent pair A, B)
-----------------------------------
  1. Load map_A_numbers.json and map_B_numbers.json (Claude readings).
  2. Group readings by number on each map.
  3. Find every number that appears on BOTH A and B.
       (For each shared number, we have set_A of candidate parcels with
        that number on A, and set_B on B.)
  4. For each shared number, pair up by minimum centroid distance after
     a coarse pre-alignment (or by the manual homography, used only for
     this matching step). The pair with smallest residual wins.
  5. Drop pairs whose residual is too large or whose number occurs many
     times on either side (uniqueness violated by OCR misreads).
  6. RANSAC homography over the surviving (cx_a, cy_a) <-> (cx_b, cy_b)
     pairs. Inliers = confirmed control points.

Adjacency comes from the CSV (which maps share boundary parcels). The
manual homography from output/homographies/homographies.json is used as
a coarse spatial PRIOR to disambiguate duplicate readings — we are NOT
copying its control points. The control points are derived from Claude's
OCR plus the uniqueness/consistency rules.

Output
-------
  new_pipeline/data/control_points/
    pair_<A>_<B>_matches.json   - confirmed control points per pair
    pair_<A>_<B>_overlay.png    - side-by-side visualisation

Usage
------
  .\\venv_thesis\\Scripts\\Activate.ps1

  # Single pair (uses cached OCR — no API calls)
  python new_pipeline/src/step8_cross_reference.py --pair 45_47

  # All 10 adjacent pairs
  python new_pipeline/src/step8_cross_reference.py --all
=============================================================================
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------

NUMBERS_DIR       = Path("new_pipeline/data/parcel_numbers")
HOMOGRAPHIES_PATH = Path("output/homographies/homographies.json")
PREPROCESSED_DIR  = Path("output/preprocessed")
OUTPUT_DIR        = Path("new_pipeline/data/control_points")

ADJACENT_PAIRS = [
    ("45", "47"), ("45", "46"), ("47", "48"), ("48", "49"),
    ("49", "50"), ("50", "51"), ("52", "53"), ("52", "54"),
    ("52", "55"), ("54", "55"),
]

# CSV ground truth (used to highlight CSV-known matches; not algorithm input)
BOUNDARY_PARCELS_CSV = {
    2580: ["45", "47"],  2616: ["45", "46"],  2619: ["45", "46"],
    2749: ["47", "48"],  2803: ["47", "48"],  2814: ["47", "48"],
    2893: ["48", "49"],  3022: ["49", "50"],  3054: ["49", "50"],
    3068: ["50", "51"],  3215: ["52", "53"],  3216: ["52", "54"],
    3217: ["52", "55"],  3338: ["54", "55"],  3339: ["54", "55"],
    3345: ["54", "55"],  3346: ["54", "55"],  3813: ["50", "51"],
    3866: ["50", "51"],
}

# Maximum number of times a number can be reported on a single map before
# we give up — OCR noise will produce far more than this if it's truly
# random. We keep it large because the residual filter (with the manual H
# prior) does the real disambiguation.
MAX_DUPLICATES_PER_MAP = 20

# Maximum acceptable residual (px) when mapping a candidate from A to B
# with the manual homography prior. The map images are 6000-9000 px, so
# 400 px is ~5% — reasonable considering Mask R-CNN centroid jitter.
MATCH_RESIDUAL_PX = 400.0

# RANSAC reprojection threshold for the final homography fit.
RANSAC_THRESHOLD_PX = 250.0


# ---------------------------------------------------------------------------
# I/O HELPERS
# ---------------------------------------------------------------------------

def load_numbers(map_num: str) -> list[dict]:
    p = NUMBERS_DIR / f"map_{map_num}_numbers.json"
    if not p.exists():
        raise FileNotFoundError(
            f"OCR output missing: {p}\n"
            f"Run step7_ocr_vision.py first."
        )
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def load_homography_meta() -> dict:
    if not HOMOGRAPHIES_PATH.exists():
        return {}
    with open(HOMOGRAPHIES_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# CORE
# ---------------------------------------------------------------------------

def group_readings(readings: list[dict]) -> dict[int, list[dict]]:
    """{number: [parcel detections that read that number]}"""
    out: dict[int, list[dict]] = defaultdict(list)
    for r in readings:
        n = r.get("recognised_number")
        if n is None:
            continue
        out[int(n)].append(r)
    return out


def best_pair_under_homography(group_a: list[dict], group_b: list[dict],
                                H: np.ndarray | None,
                                b_offset: tuple[float, float] = (0.0, 0.0),
                                ) -> tuple[dict, dict, float] | None:
    """
    Among candidates labelled with the SAME number on A and B, pick the
    pair (a, b) whose positions are most consistent under the manual H.

    The manual homography in homographies.json maps A's coordinates into
    the PANORAMA frame (not B's native frame). To compare with B's native
    centroid we add b_offset, which is where B is placed in the panorama:
        residual = || H @ A_native  -  (B_native + b_offset) ||

    If H is None, fall back to nearest-centroid pairing in raw coordinates.
    """
    if not group_a or not group_b:
        return None

    pts_a = np.array(
        [[d["cx"], d["cy"]] for d in group_a],
        dtype=np.float64,
    ).reshape(-1, 1, 2)

    if H is not None:
        try:
            proj_a = cv2.perspectiveTransform(pts_a, H).reshape(-1, 2)
        except cv2.error:
            proj_a = pts_a.reshape(-1, 2)
    else:
        proj_a = pts_a.reshape(-1, 2)

    bx_off, by_off = b_offset
    best = None
    for i, pa in enumerate(group_a):
        ax, ay = proj_a[i]
        for pb in group_b:
            bx = pb["cx"] + bx_off
            by = pb["cy"] + by_off
            d = float(np.hypot(ax - bx, ay - by))
            if best is None or d < best[2]:
                best = (pa, pb, d)
    return best


def cross_reference_pair(map_a: str, map_b: str,
                          homog_meta: dict | None) -> dict:
    readings_a = load_numbers(map_a)
    readings_b = load_numbers(map_b)
    groups_a = group_readings(readings_a)
    groups_b = group_readings(readings_b)

    H_manual = None
    b_offset = (0.0, 0.0)
    csv_pids: list[int] = []
    if homog_meta:
        if "H" in homog_meta:
            H_manual = np.array(homog_meta["H"], dtype=np.float64)
        b_off_raw = homog_meta.get("b_offset")
        if b_off_raw and len(b_off_raw) == 2:
            b_offset = (float(b_off_raw[0]), float(b_off_raw[1]))
        csv_pids = list(homog_meta.get("boundary_parcels") or [])

    # Numbers seen on BOTH sides. The uniqueness constraint says we should
    # prefer numbers that appear few times — many duplicates = OCR misread.
    shared_numbers = sorted(set(groups_a) & set(groups_b))

    candidate_matches: list[dict] = []
    dropped_too_duplicated: list[int] = []

    for num in shared_numbers:
        ga = groups_a[num]
        gb = groups_b[num]

        # If a number occurs many times on either side, the readings are
        # noise — drop instead of guessing.
        if (len(ga) > MAX_DUPLICATES_PER_MAP or
                len(gb) > MAX_DUPLICATES_PER_MAP):
            dropped_too_duplicated.append(num)
            continue

        best = best_pair_under_homography(ga, gb, H_manual, b_offset)
        if best is None:
            continue
        det_a, det_b, residual = best
        if residual > MATCH_RESIDUAL_PX:
            continue

        candidate_matches.append({
            "number":     int(num),
            "parcel_a":   int(det_a["parcel_id"]),
            "parcel_b":   int(det_b["parcel_id"]),
            "cx_a":       float(det_a["cx"]),
            "cy_a":       float(det_a["cy"]),
            "cx_b":       float(det_b["cx"]),
            "cy_b":       float(det_b["cy"]),
            "n_dup_a":    len(ga),
            "n_dup_b":    len(gb),
            "residual_under_manual_H_px": round(residual, 1),
            "is_csv_known": num in BOUNDARY_PARCELS_CSV,
        })

    candidate_matches.sort(key=lambda m: m["residual_under_manual_H_px"])

    # RANSAC: fit a homography from candidates' centroids; keep inliers.
    H_auto = None
    inliers_idx: list[int] = []
    if len(candidate_matches) >= 4:
        pts_a = np.array(
            [[m["cx_a"], m["cy_a"]] for m in candidate_matches],
            dtype=np.float32,
        )
        pts_b = np.array(
            [[m["cx_b"], m["cy_b"]] for m in candidate_matches],
            dtype=np.float32,
        )
        H_auto, mask = cv2.findHomography(
            pts_a, pts_b, cv2.RANSAC,
            ransacReprojThreshold=RANSAC_THRESHOLD_PX,
            maxIters=2000, confidence=0.99,
        )
        if mask is not None:
            mask = mask.flatten().astype(bool)
            inliers_idx = [i for i, ok in enumerate(mask) if ok]
            candidate_matches = [m for i, m in enumerate(candidate_matches)
                                 if i in inliers_idx]

    return {
        "pair":               f"{map_a}_{map_b}",
        "csv_expected":       sorted(csv_pids),
        "n_shared_numbers":   len(shared_numbers),
        "n_dropped_dup":      len(dropped_too_duplicated),
        "n_matches":          len(candidate_matches),
        "matches":            candidate_matches,
        "H_auto":             H_auto.tolist() if H_auto is not None else None,
        "dropped_too_duplicated": dropped_too_duplicated,
    }


# ---------------------------------------------------------------------------
# OUTPUT / VISUALISATION
# ---------------------------------------------------------------------------

def save_result(result: dict) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    p = OUTPUT_DIR / f"pair_{result['pair']}_matches.json"
    with open(p, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    return p


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

    csv_color  = (0, 200, 0)        # bright green = CSV-known parcel
    other_color = (0, 130, 200)     # orange-ish = match found but not in CSV

    for k, m in enumerate(result["matches"]):
        col = csv_color if m["is_csv_known"] else other_color
        pa = (int(m["cx_a"] * scale_a),
              int(m["cy_a"] * scale_a) + 60)
        pb = (int(m["cx_b"] * scale_b) + offset_b,
              int(m["cy_b"] * scale_b) + 60)
        cv2.circle(canvas, pa, 10, col, -1)
        cv2.circle(canvas, pb, 10, col, -1)
        cv2.line(canvas, pa, pb, col, 2)
        label = str(m["number"])
        cv2.putText(canvas, label, (pa[0] + 12, pa[1] + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2, cv2.LINE_AA)
        cv2.putText(canvas, label, (pb[0] + 12, pb[1] + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2, cv2.LINE_AA)

    # Header
    cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 55), (30, 30, 30), -1)
    n_csv = sum(1 for m in result["matches"] if m["is_csv_known"])
    header = (f"Pair {result['pair']}  |  {result['n_matches']} matches "
              f"({n_csv} CSV-known)  |  "
              f"{result['n_dropped_dup']} numbers dropped as OCR noise")
    cv2.putText(canvas, header, (10, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (255, 255, 255), 2, cv2.LINE_AA)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    ok, enc = cv2.imencode(".png", canvas)
    if ok:
        enc.tofile(str(out_path))


def print_summary(r: dict):
    print(f"\n  Pair {r['pair']}")
    print(f"    Numbers seen on both maps : {r['n_shared_numbers']}")
    print(f"    Dropped as OCR-noisy      : {r['n_dropped_dup']} "
          f"({r['dropped_too_duplicated']})")
    print(f"    Confirmed control points  : {r['n_matches']}")
    if r["csv_expected"]:
        n_csv_in_matches = sum(1 for m in r["matches"]
                               if m["number"] in r["csv_expected"])
        print(f"    CSV expected boundaries   : {r['csv_expected']}")
        print(f"    CSV ones we recovered     : {n_csv_in_matches} / "
              f"{len(r['csv_expected'])}")
    if r["matches"]:
        print(f"    Top matches (by residual):")
        for m in r["matches"][:6]:
            tag = "CSV " if m["is_csv_known"] else "    "
            print(f"      {tag}#{m['number']}  "
                  f"a_parcel={m['parcel_a']:>4} <-> "
                  f"b_parcel={m['parcel_b']:>4}  "
                  f"residual={m['residual_under_manual_H_px']:>5.0f}px  "
                  f"(dup_a={m['n_dup_a']}, dup_b={m['n_dup_b']})")


# ---------------------------------------------------------------------------
# RUN MODES
# ---------------------------------------------------------------------------

def run_pair(pair_key: str, no_vis: bool = False):
    a, b = pair_key.split("_")
    homog = load_homography_meta().get(pair_key)
    print(f"\n  Cross-referencing pair {pair_key}...")
    result = cross_reference_pair(a, b, homog)
    save_result(result)
    print_summary(result)
    if not no_vis:
        vis_path = OUTPUT_DIR / f"pair_{pair_key}_overlay.png"
        visualise_pair(result, vis_path)
        print(f"\n    Visualisation: {vis_path}")


def run_all(no_vis: bool = False):
    print("\n" + "=" * 70)
    print("  STEP 8 — CROSS-REFERENCE OCR READINGS, BUILD CONTROL POINTS")
    print("=" * 70)
    homog_all = load_homography_meta()
    summary = []
    for a, b in ADJACENT_PAIRS:
        pair_key = f"{a}_{b}"
        try:
            result = cross_reference_pair(a, b, homog_all.get(pair_key))
        except FileNotFoundError as e:
            print(f"\n  Skipping {pair_key}: {e}")
            continue
        save_result(result)
        print_summary(result)
        if not no_vis:
            visualise_pair(result,
                           OUTPUT_DIR / f"pair_{pair_key}_overlay.png")
        summary.append({
            "pair":     pair_key,
            "matches":  result["n_matches"],
            "csv_recovered": sum(1 for m in result["matches"]
                                  if m["is_csv_known"]),
            "csv_expected":  len(result["csv_expected"]),
        })

    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    print(f"  {'Pair':<8} {'Matches':>8} {'CSV recov':>10} {'CSV exp':>9}")
    for s in summary:
        print(f"  {s['pair']:<8} {s['matches']:>8} "
              f"{s['csv_recovered']:>10} {s['csv_expected']:>9}")
    print(f"\n  Output: {OUTPUT_DIR.resolve()}")
    print("=" * 70 + "\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Cross-reference OCR readings into control points "
                    "(uses cached step7 output, no API calls)."
    )
    parser.add_argument("--pair", type=str, default=None,
                        help="A single pair, e.g. --pair 45_47")
    parser.add_argument("--all", action="store_true",
                        help="All 10 adjacent pairs")
    parser.add_argument("--no-vis", action="store_true",
                        help="Skip overlay PNGs")
    args = parser.parse_args()

    if args.pair:
        run_pair(args.pair, no_vis=args.no_vis)
    elif args.all:
        run_all(no_vis=args.no_vis)
    else:
        print("Usage:")
        print("  python new_pipeline/src/step8_cross_reference.py --pair 45_47")
        print("  python new_pipeline/src/step8_cross_reference.py --all")
