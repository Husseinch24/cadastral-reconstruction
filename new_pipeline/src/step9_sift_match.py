"""
=============================================================================
MILESTONE 7 — SIFT INSIDE MATCHED PARCELS, REFINED HOMOGRAPHY, STITCH
=============================================================================
Masters Thesis - AI-Based Panoramic Cadastral Image Reconstruction
Author : Hussein Chalhoub

Purpose
--------
Given a small set of "this parcel = this parcel" hints from the click tool
(step 8), find pixel-level corresponding points across the two adjacent
sheets by running SIFT inside each matched parcel polygon, and use those
correspondences to fit a precise full-DOF homography for stitching.

Why this is better than using click positions directly
-------------------------------------------------------
  - The cartographer wrote each parcel's number in slightly different
    relative positions on the two adjacent sheets, so the user's click
    locations are NOT exact corresponding points (often 30-100 px off).
  - With only 3 clicks, even small position errors break the homography
    fit and the seam between maps becomes visibly misaligned.
  - SIFT features inside the parcel polygon (parcel boundary lines, the
    digits themselves, paper texture) provide DOZENS of correspondences
    per parcel pair, all pixel-precise. Aggregated and RANSAC-filtered,
    these give a clean homography.

Pipeline (per adjacent pair A, B)
----------------------------------
  1. Read clicks_<A>_<B>.json from step 8.
  2. For each clicked parcel pair (a_click, b_click):
       a. Snap a_click to the nearest Mask R-CNN parcel on map A
          (uses polygons saved in step 4)  ->  polygon_A
       b. Same for map B  ->  polygon_B
       c. Crop the bbox of each polygon (with a small margin) from the
          source image. Build a binary mask of the polygon (slightly
          dilated so the parcel boundary lines are inside the mask).
       d. Run cv2.SIFT_create(), masked to the polygon, on each crop.
          SIFT is rotation-invariant, so different sheet orientations
          don't break the descriptors.
       e. Match descriptors with FLANN k=2 + Lowe's ratio test (0.7).
       f. Translate the surviving match coordinates back to the global
          image frame (by adding the crop bbox origin).
  3. Aggregate every pair's matches into a single global pool of
     correspondences (cx_a, cy_a) <-> (cx_b, cy_b) in original-image px.
  4. cv2.findHomography(B -> A) with RANSAC (5 px reproj threshold).
  5. Save the H matrix and build a panorama using it.

Outputs
--------
  new_pipeline/data/click_matches/homographies/
    H_<A>_<B>_sift.npy             - the SIFT-derived homography

  new_pipeline/data/click_matches/previews/
    panorama_<A>_<B>_sift_<source>.png            - final panorama
    sift_match_<A>_<B>_parcel_<num>_<source>.png  - per-parcel debug viz
                                                    showing matched
                                                    keypoints

Usage
------
  python new_pipeline/src/step9_sift_match.py --pair 47_48
  python new_pipeline/src/step9_sift_match.py --pair 47_48 --source binary

  Click data (clicks_<pair>.json from step 8) is REQUIRED. No new clicks
  are collected by this script — it only consumes existing ones.
=============================================================================
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------

PREPROCESSED_DIR = Path("output/preprocessed")
PREDICTIONS_DIR  = Path("new_pipeline/data/predictions")
CLICK_DIR        = Path("new_pipeline/data/click_matches")
HOMOGRAPHY_DIR   = Path("new_pipeline/data/click_matches/homographies")
PREVIEW_DIR      = Path("new_pipeline/data/click_matches/previews")

SOURCE_KIND = "clean"   # set by --source

# Extra pixels around each polygon's bbox when cropping. Small margin lets
# SIFT capture the parcel boundary lines that sit at the polygon edge.
CROP_MARGIN_PX = 30

# Dilate the polygon mask by this many pixels so the boundary line falls
# fully INSIDE the SIFT mask (boundaries are the most distinctive feature).
MASK_DILATE_PX = 8

# Lowe's ratio test threshold. Tighter (0.65) = fewer / cleaner matches;
# looser (0.8) = more matches but more noise.
LOWE_RATIO = 0.75

# RANSAC reprojection threshold for the FINAL global homography fit (px).
RANSAC_REPROJ_PX = 8.0

# Cap canvas dim for the output panorama (memory).
MAX_CANVAS_DIM = 12000


def _set_source(kind: str):
    global SOURCE_KIND
    if kind not in ("clean", "binary"):
        raise ValueError(f"--source must be 'clean' or 'binary', got {kind!r}")
    SOURCE_KIND = kind


def _image_path(map_num: str) -> Path:
    return PREPROCESSED_DIR / f"map_{map_num}_{SOURCE_KIND}.png"


# ---------------------------------------------------------------------------
# IO
# ---------------------------------------------------------------------------

def read_image(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Cannot read: {path}")
    data = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Cannot decode: {path}")
    return img


def save_image(path: Path, img: np.ndarray):
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, enc = cv2.imencode(path.suffix, img)
    if ok:
        enc.tofile(str(path))


def load_clicks(pair_key: str) -> dict:
    p = CLICK_DIR / f"clicks_{pair_key}.json"
    if not p.exists():
        raise FileNotFoundError(
            f"Click data not found: {p}\n"
            f"Run parcel_click_locator.py --pair {pair_key} first."
        )
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def load_parcels(map_num: str) -> list[dict]:
    p = PREDICTIONS_DIR / f"map_{map_num}_parcels.json"
    if not p.exists():
        raise FileNotFoundError(
            f"Parcel JSON missing: {p}\n"
            f"Run step4_segmentation.py --infer --maps {map_num} first."
        )
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def find_nearest_parcel(parcels: list[dict],
                         x: float, y: float) -> dict | None:
    if not parcels:
        return None

    # 1. Point-in-polygon: prefer a parcel whose polygon actually contains the click.
    pt = (float(x), float(y))
    for p in parcels:
        poly = p.get("polygon") or []
        if len(poly) < 3:
            continue
        pts = np.array(poly, dtype=np.float32)
        if cv2.pointPolygonTest(pts, pt, measureDist=False) >= 0:
            return p

    # 2. Fallback: nearest centroid among parcels that have a valid polygon.
    best, best_d = None, float("inf")
    for p in parcels:
        if len(p.get("polygon") or []) < 3:
            continue
        d = (p["cx"] - x) ** 2 + (p["cy"] - y) ** 2
        if d < best_d:
            best_d = d
            best = p
    return best


# ---------------------------------------------------------------------------
# CROP + MASK
# ---------------------------------------------------------------------------

def polygon_bbox(polygon: list[list[int]],
                  margin: int = 0,
                  img_shape: tuple[int, int] | None = None
                  ) -> tuple[int, int, int, int]:
    pts = np.array(polygon, dtype=np.int32)
    x_min = int(pts[:, 0].min()) - margin
    y_min = int(pts[:, 1].min()) - margin
    x_max = int(pts[:, 0].max()) + margin
    y_max = int(pts[:, 1].max()) + margin
    if img_shape is not None:
        H, W = img_shape[:2]
        x_min = max(0, x_min); y_min = max(0, y_min)
        x_max = min(W, x_max); y_max = min(H, y_max)
    return x_min, y_min, x_max, y_max


def crop_polygon(img: np.ndarray,
                  polygon: list[list[int]]
                  ) -> tuple[np.ndarray, np.ndarray, tuple[int, int]]:
    """
    Return (crop, mask, (x_min, y_min)).

    crop:  the bbox of the polygon (with CROP_MARGIN_PX margin) cut out of img.
    mask:  same H/W as crop; 255 inside the polygon (slightly dilated so the
           boundary line is included), 0 outside.
    """
    x_min, y_min, x_max, y_max = polygon_bbox(
        polygon, margin=CROP_MARGIN_PX, img_shape=img.shape
    )
    crop = img[y_min:y_max, x_min:x_max].copy()
    mask = np.zeros(crop.shape[:2], dtype=np.uint8)
    pts_local = (np.array(polygon, dtype=np.int32)
                  - np.array([x_min, y_min], dtype=np.int32))
    cv2.fillPoly(mask, [pts_local], 255)
    if MASK_DILATE_PX > 0:
        k = cv2.getStructuringElement(
            cv2.MORPH_RECT, (MASK_DILATE_PX * 2 + 1, MASK_DILATE_PX * 2 + 1)
        )
        mask = cv2.dilate(mask, k)
    return crop, mask, (x_min, y_min)


# ---------------------------------------------------------------------------
# SIFT
# ---------------------------------------------------------------------------

_sift = None


def _get_sift():
    global _sift
    if _sift is None:
        _sift = cv2.SIFT_create()
    return _sift


def _flann_matcher():
    # FLANN with KDTree index (suitable for SIFT's float descriptors).
    return cv2.FlannBasedMatcher(
        dict(algorithm=1, trees=5),     # KDTreeIndexParams
        dict(checks=50),                 # SearchParams
    )


def detect_and_match(crop_a: np.ndarray, mask_a: np.ndarray,
                      crop_b: np.ndarray, mask_b: np.ndarray
                      ) -> tuple[list, list, list]:
    """Return (good_matches, kp_a, kp_b)."""
    sift = _get_sift()
    kp_a, des_a = sift.detectAndCompute(crop_a, mask_a)
    kp_b, des_b = sift.detectAndCompute(crop_b, mask_b)
    if des_a is None or des_b is None or len(kp_a) < 2 or len(kp_b) < 2:
        return [], list(kp_a or []), list(kp_b or [])

    flann = _flann_matcher()
    raw = flann.knnMatch(des_a, des_b, k=2)

    # Lowe's ratio test: a good match must be MUCH better than the second-best.
    good = []
    for pair in raw:
        if len(pair) < 2:
            continue
        m, n = pair
        if m.distance < LOWE_RATIO * n.distance:
            good.append(m)
    return good, list(kp_a), list(kp_b)


def visualise_parcel_matches(crop_a: np.ndarray, kp_a, mask_a: np.ndarray,
                              crop_b: np.ndarray, kp_b, mask_b: np.ndarray,
                              good_matches: list,
                              out_path: Path,
                              header_text: str):
    """Side-by-side image with keypoint matches drawn as green lines."""
    if crop_a.ndim == 2:
        cb_a = cv2.cvtColor(crop_a, cv2.COLOR_GRAY2BGR)
        cb_b = cv2.cvtColor(crop_b, cv2.COLOR_GRAY2BGR)
    else:
        cb_a = crop_a.copy(); cb_b = crop_b.copy()

    # Faintly tint where the polygon mask covers (so we can see what's masked)
    cb_a[mask_a == 0] = (cb_a[mask_a == 0] * 0.55).astype(np.uint8)
    cb_b[mask_b == 0] = (cb_b[mask_b == 0] * 0.55).astype(np.uint8)

    # Limit drawn matches for legibility
    drawn = good_matches[:60]
    img = cv2.drawMatches(
        cb_a, kp_a, cb_b, kp_b, drawn, None,
        matchColor=(0, 220, 0),
        singlePointColor=(0, 0, 220),
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )

    # Header
    cv2.rectangle(img, (0, 0), (img.shape[1], 38), (30, 30, 30), -1)
    cv2.putText(img, header_text, (10, 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (255, 255, 255), 1, cv2.LINE_AA)

    save_image(out_path, img)


# ---------------------------------------------------------------------------
# PANORAMA (same compositing as parcel_click_locator)
# ---------------------------------------------------------------------------

def build_pair_panorama(map_a: str, map_b: str,
                          H: np.ndarray) -> np.ndarray:
    img_a = read_image(_image_path(map_a))
    img_b = read_image(_image_path(map_b))
    h_a, w_a = img_a.shape
    h_b, w_b = img_b.shape

    corners_b = np.float32(
        [[0, 0], [w_b, 0], [w_b, h_b], [0, h_b]]
    ).reshape(-1, 1, 2)
    warped_b_corners = cv2.perspectiveTransform(corners_b, H).reshape(-1, 2)

    all_x = np.concatenate(
        [warped_b_corners[:, 0], np.array([0, w_a, w_a, 0])]
    )
    all_y = np.concatenate(
        [warped_b_corners[:, 1], np.array([0, 0, h_a, h_a])]
    )
    x_min = int(np.floor(all_x.min())); y_min = int(np.floor(all_y.min()))
    x_max = int(np.ceil(all_x.max()));  y_max = int(np.ceil(all_y.max()))
    canvas_w = x_max - x_min
    canvas_h = y_max - y_min

    scale = 1.0
    if max(canvas_w, canvas_h) > MAX_CANVAS_DIM:
        scale = MAX_CANVAS_DIM / max(canvas_w, canvas_h)
        canvas_w = int(canvas_w * scale)
        canvas_h = int(canvas_h * scale)

    T = np.array(
        [[scale, 0,     -x_min * scale],
         [0,     scale, -y_min * scale],
         [0,     0,     1.0]],
        dtype=np.float64,
    )
    H_a = T
    H_b = T @ H

    canvas = np.full((canvas_h, canvas_w), 255, dtype=np.uint8)
    warped_a = cv2.warpPerspective(
        img_a, H_a, (canvas_w, canvas_h),
        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0,
    )
    warped_b = cv2.warpPerspective(
        img_b, H_b, (canvas_w, canvas_h),
        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0,
    )
    cover_a = (warped_a > 0).astype(np.uint8)
    cover_b = (warped_b > 0).astype(np.uint8)
    only_a  = (cover_a == 1) & (cover_b == 0)
    only_b  = (cover_a == 0) & (cover_b == 1)
    overlap = (cover_a == 1) & (cover_b == 1)
    canvas[only_a] = warped_a[only_a]
    canvas[only_b] = warped_b[only_b]
    if overlap.any():
        avg = ((warped_a[overlap].astype(np.uint16)
                + warped_b[overlap].astype(np.uint16)) // 2).astype(np.uint8)
        canvas[overlap] = avg

    return cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)


# ---------------------------------------------------------------------------
# MAIN PROCESS
# ---------------------------------------------------------------------------

def process_pair(pair_key: str) -> bool:
    print("\n" + "=" * 70)
    print(f"  STEP 9 — SIFT MATCH inside parcels for pair {pair_key}")
    print(f"  Source: {SOURCE_KIND}")
    print("=" * 70)

    clicks = load_clicks(pair_key)
    map_a = clicks["map_a"]
    map_b = clicks["map_b"]
    common = clicks.get("common") or list(clicks.get("clicks_a", {}).keys())
    common = [int(n) for n in common]
    if not common:
        print("  No clicked parcels in this file. Aborting.")
        return False
    print(f"  Clicked parcel pairs ({len(common)}): {common}")

    img_a = read_image(_image_path(map_a))
    img_b = read_image(_image_path(map_b))
    parcels_a = load_parcels(map_a)
    parcels_b = load_parcels(map_b)

    pts_a_global: list[np.ndarray] = []
    pts_b_global: list[np.ndarray] = []

    for parcel_num in common:
        click_a = clicks["clicks_a"][str(parcel_num)]
        click_b = clicks["clicks_b"][str(parcel_num)]
        pa = find_nearest_parcel(parcels_a, click_a[0], click_a[1])
        pb = find_nearest_parcel(parcels_b, click_b[0], click_b[1])
        if pa is None or pb is None:
            print(f"  Parcel #{parcel_num}: no Mask R-CNN match near click. "
                  f"Skipping.")
            continue
        poly_a = pa.get("polygon") or []
        poly_b = pb.get("polygon") or []
        if len(poly_a) < 3 or len(poly_b) < 3:
            print(f"  Parcel #{parcel_num}: polygon missing on Mask R-CNN "
                  f"detection. Skipping.")
            continue

        crop_a, mask_a, off_a = crop_polygon(img_a, poly_a)
        crop_b, mask_b, off_b = crop_polygon(img_b, poly_b)
        good, kp_a, kp_b = detect_and_match(crop_a, mask_a, crop_b, mask_b)

        print(f"  Parcel #{parcel_num}: "
              f"crop {crop_a.shape[1]}x{crop_a.shape[0]} vs "
              f"{crop_b.shape[1]}x{crop_b.shape[0]}, "
              f"kp_a={len(kp_a)}, kp_b={len(kp_b)}, "
              f"good_matches={len(good)}")

        # Save per-parcel debug image regardless of match count
        vis_path = (
            PREVIEW_DIR
            / f"sift_match_{pair_key}_parcel_{parcel_num}_{SOURCE_KIND}.png"
        )
        visualise_parcel_matches(
            crop_a, kp_a, mask_a, crop_b, kp_b, mask_b, good,
            vis_path,
            f"Pair {pair_key}  parcel #{parcel_num}  "
            f"good={len(good)} kp_a={len(kp_a)} kp_b={len(kp_b)}  "
            f"({SOURCE_KIND})",
        )

        if len(good) < 4:
            continue

        local_a = np.float32([kp_a[m.queryIdx].pt for m in good])
        local_b = np.float32([kp_b[m.trainIdx].pt for m in good])
        pts_a_global.append(local_a + np.array(off_a, dtype=np.float32))
        pts_b_global.append(local_b + np.array(off_b, dtype=np.float32))

    if not pts_a_global:
        print("\n  No SIFT matches found in any parcel. Cannot fit homography.")
        print("  Suggestions:")
        print("    - Try --source binary (boundary lines are crisper there)")
        print("    - Click larger / more textured parcels")
        return False

    pts_a = np.vstack(pts_a_global)
    pts_b = np.vstack(pts_b_global)
    print(f"\n  Total correspondences: {len(pts_a)}")

    # Fit homography mapping B -> A (same direction as parcel_click_locator's H)
    H, mask = cv2.findHomography(
        pts_b, pts_a,
        method=cv2.RANSAC,
        ransacReprojThreshold=RANSAC_REPROJ_PX,
        maxIters=5000, confidence=0.995,
    )
    if H is None or mask is None:
        print("  RANSAC failed to fit a homography.")
        return False

    inliers = int(mask.sum())
    print(f"  RANSAC inliers: {inliers} / {len(pts_a)} "
          f"({100.0 * inliers / max(len(pts_a), 1):.0f}%)")

    HOMOGRAPHY_DIR.mkdir(parents=True, exist_ok=True)
    PREVIEW_DIR.mkdir(parents=True, exist_ok=True)

    h_path = HOMOGRAPHY_DIR / f"H_{pair_key}_sift.npy"
    np.save(str(h_path), H)
    print(f"  Saved: {h_path}")

    print("\n  Building panorama with the SIFT-derived homography...")
    panorama = build_pair_panorama(map_a, map_b, H)
    pano_path = (
        PREVIEW_DIR / f"panorama_{pair_key}_sift_{SOURCE_KIND}.png"
    )
    save_image(pano_path, panorama)
    print(f"  Saved: {pano_path}  ({panorama.shape[1]}x{panorama.shape[0]})")

    print("\n" + "=" * 70)
    print(f"  PAIR {pair_key} done. Inspect:")
    print(f"    Panorama   : {pano_path}")
    print(f"    Per-parcel : "
          f"{PREVIEW_DIR}/sift_match_{pair_key}_parcel_*.png")
    print("=" * 70 + "\n")
    return True


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Refine pair homography with SIFT inside matched parcels."
    )
    parser.add_argument("--pair", type=str, required=True,
                        help="Adjacent pair, e.g. --pair 47_48")
    parser.add_argument("--source", type=str, default="clean",
                        choices=["clean", "binary"],
                        help="Which preprocessed image to use.")
    args = parser.parse_args()

    _set_source(args.source)
    process_pair(args.pair)
