"""
=============================================================================
PARCEL CLICK LOCATOR — Manual centroid annotation for boundary parcels
=============================================================================
Masters Thesis — AI-Based Panoramic Cadastral Image Reconstruction
Author  : Hussein Chalhoub

Purpose
--------
Test version: locate specific boundary parcels by manual clicking.
Tests on pair 45_47 ONLY (parcel 2580) before scaling to all pairs.

You will see two windows:
  Window 1: map 45 - find parcel 2580, click on it
  Window 2: map 47 - find parcel 2580, click on it

The tool records the centroids and immediately produces:
  - A homography H_45_47.npy
  - A stitched preview panorama_45_47.png
  - A side-by-side match visualization

If the panorama looks correct, we extend to all pairs.
If not, we adjust and try again.

Workflow
---------
1. Window opens showing map 45 zoomed out
2. Scroll wheel to zoom, right-click drag to pan
3. Find parcel 2580 (the number written inside the parcel)
4. LEFT CLICK on the centre of that parcel
5. The window closes, then map 47 opens
6. Find parcel 2580 on map 47, click on it
7. The tool computes the homography and shows you the result

Controls
---------
  Left click       : Mark the parcel centroid
  Right click drag : Pan the view
  Scroll wheel     : Zoom in/out
  F                : Fit view
  U                : Undo last click (reopens current map)
  Q or Esc         : Quit without saving

Usage
------
  .\venv_thesis\Scripts\Activate.ps1
  python new_pipeline/src/parcel_click_locator.py --pair 45_47

After test succeeds, run for all pairs:
  python new_pipeline/src/parcel_click_locator.py --all

=============================================================================
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------

PREPROCESSED_DIR = Path("output/preprocessed")
OUTPUT_DIR       = Path("new_pipeline/data/click_matches")
HOMOGRAPHY_DIR   = Path("new_pipeline/data/click_matches/homographies")
PREVIEW_DIR      = Path("new_pipeline/data/click_matches/previews")

# "clean" or "binary" — set by --source. The clicks themselves are saved
# without a source suffix so they're shared between sources (you click on
# the same physical parcels regardless). The panorama / match-viz outputs
# are namespaced per source so you can compare them.
SOURCE_KIND = "clean"


def _set_source(kind: str):
    global SOURCE_KIND
    if kind not in ("clean", "binary"):
        raise ValueError(f"--source must be 'clean' or 'binary', got {kind!r}")
    SOURCE_KIND = kind


def _image_path(map_num: str) -> Path:
    return PREPROCESSED_DIR / f"map_{map_num}_{SOURCE_KIND}.png"


def _panorama_path(pair_key: str) -> Path:
    return PREVIEW_DIR / f"panorama_{pair_key}_{SOURCE_KIND}.png"


def _matches_path(pair_key: str) -> Path:
    return PREVIEW_DIR / f"matches_{pair_key}_{SOURCE_KIND}.png"

# Ground truth from CSV: pair -> list of boundary parcel numbers
PAIR_PARCELS = {
    "45_47": [2580],
    "45_46": [2616, 2619],
    "47_48": [2749, 2803, 2814],
    "48_49": [2893],
    "49_50": [3022, 3054],
    "50_51": [3068, 3813, 3866],
    "52_53": [3215],
    "52_54": [3216],
    "52_55": [3217],
    "54_55": [3338, 3339, 3345, 3346],
}

WINDOW_W = 1500
WINDOW_H = 950

FONT          = cv2.FONT_HERSHEY_SIMPLEX
COL_HUD_BG    = (30, 30, 30)
COL_HUD_FG    = (255, 255, 255)
COL_CLICK     = (0, 0, 255)
COL_INSTRUCT  = (0, 200, 255)


# ---------------------------------------------------------------------------
# UTILITIES
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


# ---------------------------------------------------------------------------
# VIEW (zoom + pan)
# ---------------------------------------------------------------------------

class View:
    def __init__(self, canvas_w: int, canvas_h: int,
                 win_w: int, win_h: int):
        self.canvas_w = canvas_w
        self.canvas_h = canvas_h
        self.win_w    = win_w
        self.win_h    = win_h
        self.fit()

    def fit(self):
        self.scale = min(self.win_w / self.canvas_w,
                          self.win_h / self.canvas_h)
        self.pan_x = (self.win_w - self.canvas_w * self.scale) / 2
        self.pan_y = (self.win_h - self.canvas_h * self.scale) / 2

    def window_to_canvas(self, wx, wy):
        return ((wx - self.pan_x) / self.scale,
                (wy - self.pan_y) / self.scale)

    def canvas_to_window(self, cx, cy):
        return (cx * self.scale + self.pan_x,
                cy * self.scale + self.pan_y)

    def zoom_at(self, factor, wx, wy):
        cx, cy = self.window_to_canvas(wx, wy)
        self.scale = max(0.05, min(20.0, self.scale * factor))
        self.pan_x = wx - cx * self.scale
        self.pan_y = wy - cy * self.scale

    def render(self, canvas: np.ndarray) -> np.ndarray:
        """Render canvas to window-sized output applying current zoom + pan."""
        cx0, cy0 = self.window_to_canvas(0, 0)
        cx1, cy1 = self.window_to_canvas(self.win_w, self.win_h)
        cx0c = int(max(0, cx0)); cy0c = int(max(0, cy0))
        cx1c = int(min(self.canvas_w, cx1))
        cy1c = int(min(self.canvas_h, cy1))
        if cx1c <= cx0c or cy1c <= cy0c:
            return np.zeros((self.win_h, self.win_w, 3), dtype=np.uint8)
        crop   = canvas[cy0c:cy1c, cx0c:cx1c]
        sw     = max(1, int((cx1c - cx0c) * self.scale))
        sh     = max(1, int((cy1c - cy0c) * self.scale))
        interp = cv2.INTER_AREA if self.scale < 1 else cv2.INTER_LINEAR
        scaled = cv2.resize(crop, (sw, sh), interpolation=interp)
        out = np.zeros((self.win_h, self.win_w, 3), dtype=np.uint8)
        dx = int(self.pan_x + cx0c * self.scale)
        dy = int(self.pan_y + cy0c * self.scale)
        sx0 = max(0, -dx); sy0 = max(0, -dy)
        dx = max(0, dx);   dy = max(0, dy)
        h_copy = min(sh - sy0, self.win_h - dy)
        w_copy = min(sw - sx0, self.win_w - dx)
        if h_copy > 0 and w_copy > 0:
            out[dy:dy + h_copy, dx:dx + w_copy] = \
                scaled[sy0:sy0 + h_copy, sx0:sx0 + w_copy]
        return out


# ---------------------------------------------------------------------------
# CLICK COLLECTION
# ---------------------------------------------------------------------------

def collect_clicks_on_map(map_num: str,
                           parcels_to_find: list[int]
                           ) -> dict[int, tuple[float, float]]:
    """
    Open a window for the given map. For each parcel number in
    parcels_to_find, ask the user to click on it.

    Returns dict mapping: parcel_number -> (cx, cy) in original image coords.
    User can press 'Q' or close the window to abort.
    """
    img_path = _image_path(map_num)
    img_full = read_image(img_path)
    h_full, w_full = img_full.shape

    view = View(w_full, h_full, WINDOW_W, WINDOW_H)

    # State
    current_idx   = 0
    clicks        = {}     # parcel_num -> (cx, cy)
    pan_dragging  = False
    pan_start     = None
    needs_redraw  = True

    win_name = f"Click on parcels - Map {map_num}"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win_name, WINDOW_W, WINDOW_H)

    aborted = [False]   # mutable flag for mouse callback to set

    def on_mouse(event, x, y, flags, param):
        nonlocal pan_dragging, pan_start, needs_redraw, current_idx

        if event == cv2.EVENT_LBUTTONDOWN:
            if current_idx < len(parcels_to_find):
                cx, cy = view.window_to_canvas(x, y)
                cx = max(0.0, min(cx, float(w_full - 1)))
                cy = max(0.0, min(cy, float(h_full - 1)))
                parcel_num = parcels_to_find[current_idx]
                clicks[parcel_num] = (cx, cy)
                print(f"  Clicked parcel {parcel_num} at "
                      f"({cx:.0f}, {cy:.0f}) on map {map_num}")
                current_idx += 1
                needs_redraw = True

        elif event == cv2.EVENT_RBUTTONDOWN:
            pan_dragging = True
            pan_start = (x, y, view.pan_x, view.pan_y)
        elif event == cv2.EVENT_MOUSEMOVE and pan_dragging:
            view.pan_x = pan_start[2] + (x - pan_start[0])
            view.pan_y = pan_start[3] + (y - pan_start[1])
            needs_redraw = True
        elif event == cv2.EVENT_RBUTTONUP:
            pan_dragging = False

        elif event == cv2.EVENT_MOUSEWHEEL:
            factor = 1.25 if flags > 0 else 1 / 1.25
            view.zoom_at(factor, x, y)
            needs_redraw = True

    cv2.setMouseCallback(win_name, on_mouse)

    print(f"\n  Map {map_num}: locate parcels {parcels_to_find}")
    print(f"  Use scroll to zoom, right-drag to pan")
    print(f"  Click the centre of each parcel as instructed in the window.\n")

    while True:
        if needs_redraw:
            canvas = cv2.cvtColor(img_full, cv2.COLOR_GRAY2BGR)

            # Draw existing clicks
            for parcel_num, (cx, cy) in clicks.items():
                cv2.circle(canvas, (int(cx), int(cy)), 20, COL_CLICK, -1)
                cv2.putText(canvas, str(parcel_num),
                            (int(cx) + 25, int(cy) + 8),
                            FONT, 1.5, COL_CLICK, 4, cv2.LINE_AA)

            # Render through view
            display = view.render(canvas)

            # HUD
            cv2.rectangle(display, (0, 0), (display.shape[1], 60),
                           COL_HUD_BG, -1)
            if current_idx < len(parcels_to_find):
                target = parcels_to_find[current_idx]
                instr  = (f"Map {map_num}  |  "
                          f"FIND AND CLICK PARCEL: {target}  "
                          f"({current_idx + 1}/{len(parcels_to_find)})")
                color = COL_INSTRUCT
            else:
                instr = (f"Map {map_num}  |  "
                         f"All {len(parcels_to_find)} parcels marked.  "
                         f"Press SPACE to continue, U to undo, Q to abort")
                color = (0, 255, 0)
            cv2.putText(display, instr, (10, 25), FONT, 0.7, color, 2,
                         cv2.LINE_AA)

            ctrl = ("[Left click = mark]  [Right-drag = pan]  "
                    "[Scroll = zoom]  [F = fit]  [U = undo]  "
                    "[SPACE = next]  [Q = abort]")
            cv2.putText(display, ctrl, (10, 50), FONT, 0.5, COL_HUD_FG, 1,
                         cv2.LINE_AA)
            cv2.imshow(win_name, display)
            needs_redraw = False

        key = cv2.waitKey(20) & 0xFFFF
        if key == 0xFFFF:
            continue
        kc = key & 0xFF
        needs_redraw = True

        if kc in (ord('q'), ord('Q'), 27):
            aborted[0] = True
            break
        elif kc in (ord('f'), ord('F')):
            view.fit()
        elif kc in (ord('u'), ord('U')):
            if current_idx > 0:
                current_idx -= 1
                last_parcel = parcels_to_find[current_idx]
                if last_parcel in clicks:
                    del clicks[last_parcel]
                print(f"  Undid click on parcel {last_parcel}")
        elif kc == ord(' '):
            if current_idx >= len(parcels_to_find):
                break
            else:
                print(f"  Click parcel {parcels_to_find[current_idx]} first")

    cv2.destroyAllWindows()

    if aborted[0]:
        return {}

    return clicks


# ---------------------------------------------------------------------------
# HOMOGRAPHY COMPUTATION
# ---------------------------------------------------------------------------

def compute_homography(clicks_a: dict, clicks_b: dict
                        ) -> tuple[np.ndarray, list[int]] | tuple[None, list]:
    """
    Compute the homography from clicks_b -> clicks_a (B's pixels into A's frame).

    With 1 point: pure translation (no rotation/scale)
    With 2 points: similarity transform (translation + rotation + uniform scale)
    With 3 points: affine transform
    With 4+ points: full homography

    Returns (H, common_parcels) or (None, []) if not enough points.
    """
    common = sorted(set(clicks_a.keys()) & set(clicks_b.keys()))
    if not common:
        return None, []

    pts_a = np.array([clicks_a[n] for n in common], dtype=np.float32)
    pts_b = np.array([clicks_b[n] for n in common], dtype=np.float32)

    n = len(common)

    if n == 1:
        # Pure translation: t = a - b
        t = pts_a[0] - pts_b[0]
        H = np.array([
            [1.0, 0.0, t[0]],
            [0.0, 1.0, t[1]],
            [0.0, 0.0, 1.0],
        ], dtype=np.float64)
        print(f"  Computed translation-only H: dx={t[0]:.0f}, dy={t[1]:.0f}")
        return H, common

    if n == 2:
        # Similarity transform from 2 points
        M, _ = cv2.estimateAffinePartial2D(
            pts_b.reshape(-1, 1, 2),
            pts_a.reshape(-1, 1, 2),
            method=cv2.LMEDS,
        )
        if M is None:
            # Fallback to translation
            t = pts_a.mean(axis=0) - pts_b.mean(axis=0)
            H = np.array([[1, 0, t[0]], [0, 1, t[1]], [0, 0, 1]],
                          dtype=np.float64)
            return H, common
        H = np.vstack([M, [0, 0, 1]]).astype(np.float64)
        print(f"  Computed similarity H from 2 points")
        return H, common

    if n == 3:
        # Similarity transform (4 DOF: rotation + uniform scale + translation).
        # NOT cv2.getAffineTransform: that solves the 6-DOF affine exactly
        # for 3 points (including shear), which over-fits and extrapolates
        # wildly far from the click region — see issue with map 47/48 where
        # the panorama came out sheared into a parallelogram.
        # estimateAffinePartial2D is overdetermined for 3 points and noise-
        # robust; it produces a physically correct rigid+scale transform.
        M, _ = cv2.estimateAffinePartial2D(
            pts_b.reshape(-1, 1, 2),
            pts_a.reshape(-1, 1, 2),
            method=cv2.LMEDS,
        )
        if M is None:
            t = pts_a.mean(axis=0) - pts_b.mean(axis=0)
            H = np.array([[1, 0, t[0]], [0, 1, t[1]], [0, 0, 1]],
                          dtype=np.float64)
            return H, common
        H = np.vstack([M, [0, 0, 1]]).astype(np.float64)
        print(f"  Computed similarity H from 3 points "
              f"(rotation + uniform scale + translation, no shear)")
        return H, common

    # 4 points: still prefer similarity unless the user really clicked 6+
    # well-spread points. Similarity (4 DOF) is more stable than full H
    # (8 DOF) and physically appropriate for atlas sheets.
    if n < 6:
        M, _ = cv2.estimateAffinePartial2D(
            pts_b.reshape(-1, 1, 2),
            pts_a.reshape(-1, 1, 2),
            method=cv2.LMEDS,
        )
        if M is not None:
            H = np.vstack([M, [0, 0, 1]]).astype(np.float64)
            print(f"  Computed similarity H from {n} points")
            return H, common

    # 6+ points: full homography
    H, _ = cv2.findHomography(pts_b, pts_a, method=cv2.LMEDS)
    if H is None:
        return None, []
    print(f"  Computed full H from {n} points")
    return H.astype(np.float64), common


# ---------------------------------------------------------------------------
# PANORAMA PREVIEW
# ---------------------------------------------------------------------------

def build_pair_panorama(map_a: str, map_b: str,
                         H: np.ndarray) -> np.ndarray:
    """
    Build a stitched panorama of two maps using the given homography H
    (which maps B's pixels into A's frame).
    """
    img_a = read_image(_image_path(map_a))
    img_b = read_image(_image_path(map_b))
    h_a, w_a = img_a.shape
    h_b, w_b = img_b.shape

    # Compute bounding box of warped B + A in A's frame
    corners_b = np.float32([
        [0, 0], [w_b, 0], [w_b, h_b], [0, h_b]
    ]).reshape(-1, 1, 2)
    warped_b_corners = cv2.perspectiveTransform(corners_b, H).reshape(-1, 2)

    all_pts_x = np.concatenate([
        warped_b_corners[:, 0],
        np.array([0, w_a, w_a, 0]),
    ])
    all_pts_y = np.concatenate([
        warped_b_corners[:, 1],
        np.array([0, 0, h_a, h_a]),
    ])
    x_min = int(np.floor(all_pts_x.min()))
    y_min = int(np.floor(all_pts_y.min()))
    x_max = int(np.ceil(all_pts_x.max()))
    y_max = int(np.ceil(all_pts_y.max()))
    canvas_w = x_max - x_min
    canvas_h = y_max - y_min

    # Cap canvas size to keep memory reasonable
    MAX_DIM = 12000
    scale = 1.0
    if max(canvas_w, canvas_h) > MAX_DIM:
        scale = MAX_DIM / max(canvas_w, canvas_h)
        canvas_w = int(canvas_w * scale)
        canvas_h = int(canvas_h * scale)

    # Translation + scale to put everything in [0, canvas_w] x [0, canvas_h]
    T = np.array([
        [scale, 0,     -x_min * scale],
        [0,     scale, -y_min * scale],
        [0,     0,     1],
    ], dtype=np.float64)

    H_a = T
    H_b = T @ H

    canvas = np.full((canvas_h, canvas_w), 255, dtype=np.uint8)

    warped_a = cv2.warpPerspective(img_a, H_a, (canvas_w, canvas_h),
                                     flags=cv2.INTER_LINEAR,
                                     borderMode=cv2.BORDER_CONSTANT,
                                     borderValue=0)
    warped_b = cv2.warpPerspective(img_b, H_b, (canvas_w, canvas_h),
                                     flags=cv2.INTER_LINEAR,
                                     borderMode=cv2.BORDER_CONSTANT,
                                     borderValue=0)
    cover_a = (warped_a > 0).astype(np.uint8)
    cover_b = (warped_b > 0).astype(np.uint8)

    only_a   = (cover_a == 1) & (cover_b == 0)
    only_b   = (cover_a == 0) & (cover_b == 1)
    overlap  = (cover_a == 1) & (cover_b == 1)

    canvas[only_a] = warped_a[only_a]
    canvas[only_b] = warped_b[only_b]
    if overlap.any():
        # Average overlap
        avg = ((warped_a[overlap].astype(np.uint16)
                + warped_b[overlap].astype(np.uint16)) // 2).astype(np.uint8)
        canvas[overlap] = avg

    return cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)


def visualise_clicks(map_a: str, map_b: str,
                       clicks_a: dict, clicks_b: dict,
                       common: list[int]) -> np.ndarray:
    """Side-by-side visualization with coloured lines connecting matched clicks."""
    img_a_full = read_image(_image_path(map_a))
    img_b_full = read_image(_image_path(map_b))

    scale = 1500 / max(
        img_a_full.shape[0], img_a_full.shape[1],
        img_b_full.shape[0], img_b_full.shape[1],
    )
    img_a = cv2.resize(img_a_full,
                       (int(img_a_full.shape[1] * scale),
                        int(img_a_full.shape[0] * scale)))
    img_b = cv2.resize(img_b_full,
                       (int(img_b_full.shape[1] * scale),
                        int(img_b_full.shape[0] * scale)))
    img_a = cv2.cvtColor(img_a, cv2.COLOR_GRAY2BGR)
    img_b = cv2.cvtColor(img_b, cv2.COLOR_GRAY2BGR)

    h = max(img_a.shape[0], img_b.shape[0])
    gap = 30
    canvas = np.full((h + 60, img_a.shape[1] + gap + img_b.shape[1], 3),
                       255, dtype=np.uint8)
    canvas[60:60 + img_a.shape[0], :img_a.shape[1]] = img_a
    canvas[60:60 + img_b.shape[0], img_a.shape[1] + gap:] = img_b

    colours = [
        (0, 200, 0), (200, 0, 0), (0, 0, 220),
        (220, 165, 0), (140, 0, 140), (0, 220, 220),
    ]
    offset_b = img_a.shape[1] + gap

    for k, parcel_num in enumerate(common):
        col = colours[k % len(colours)]
        ax, ay = clicks_a[parcel_num]
        bx, by = clicks_b[parcel_num]
        pa = (int(ax * scale), int(ay * scale) + 60)
        pb = (int(bx * scale) + offset_b, int(by * scale) + 60)
        cv2.line(canvas, pa, pb, col, 2)
        cv2.circle(canvas, pa, 12, col, -1)
        cv2.circle(canvas, pb, 12, col, -1)
        cv2.putText(canvas, str(parcel_num),
                     (pa[0] + 15, pa[1] + 5),
                     FONT, 0.7, col, 2, cv2.LINE_AA)
        cv2.putText(canvas, str(parcel_num),
                     (pb[0] + 15, pb[1] + 5),
                     FONT, 0.7, col, 2, cv2.LINE_AA)

    cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 50), COL_HUD_BG, -1)
    cv2.putText(canvas,
                f"Map {map_a}  <-->  Map {map_b}  |  "
                f"{len(common)} matched parcels",
                (10, 35), FONT, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
    return canvas


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def load_saved_clicks(pair_key: str) -> tuple[dict, dict] | None:
    """Load clicks from a previous run if `clicks_<pair>.json` exists."""
    p = OUTPUT_DIR / f"clicks_{pair_key}.json"
    if not p.exists():
        return None
    with open(p, "r", encoding="utf-8") as f:
        d = json.load(f)
    clicks_a = {int(k): tuple(v) for k, v in d.get("clicks_a", {}).items()}
    clicks_b = {int(k): tuple(v) for k, v in d.get("clicks_b", {}).items()}
    if not clicks_a or not clicks_b:
        return None
    return clicks_a, clicks_b


def process_pair(pair_key: str, rebuild_only: bool = False):
    if pair_key not in PAIR_PARCELS:
        print(f"ERROR: Unknown pair {pair_key}")
        print(f"Available pairs: {list(PAIR_PARCELS.keys())}")
        return False

    parcels = PAIR_PARCELS[pair_key]
    map_a, map_b = pair_key.split("_")

    print(f"\n{'='*70}")
    print(f"  PAIR {pair_key} — boundary parcels: {parcels}")
    print(f"{'='*70}")

    if rebuild_only:
        # Reuse saved clicks from a previous run; skip the click UI entirely.
        cached = load_saved_clicks(pair_key)
        if cached is None:
            print(f"\n  --rebuild: no saved clicks for {pair_key}.")
            print(f"  Run without --rebuild first to collect clicks.")
            return False
        clicks_a, clicks_b = cached
        print(f"\n  Loaded saved clicks: "
              f"{sorted(clicks_a.keys())} on {map_a}, "
              f"{sorted(clicks_b.keys())} on {map_b}")
    else:
        # Step 1: collect clicks on map A
        print(f"\n  --- MAP {map_a} ---")
        clicks_a = collect_clicks_on_map(map_a, parcels)
        if not clicks_a:
            print("  Aborted on map A.")
            return False

        # Step 2: collect clicks on map B
        print(f"\n  --- MAP {map_b} ---")
        clicks_b = collect_clicks_on_map(map_b, parcels)
        if not clicks_b:
            print("  Aborted on map B.")
            return False

    # Step 3: compute homography
    print(f"\n  --- HOMOGRAPHY ---")
    H, common = compute_homography(clicks_a, clicks_b)
    if H is None or not common:
        print("  ERROR: Could not compute homography.")
        return False

    print(f"  Common parcels: {common}")

    # Step 4: save everything
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    HOMOGRAPHY_DIR.mkdir(parents=True, exist_ok=True)
    PREVIEW_DIR.mkdir(parents=True, exist_ok=True)

    # Save clicks
    clicks_data = {
        "pair":       pair_key,
        "map_a":      map_a,
        "map_b":      map_b,
        "parcels":    parcels,
        "clicks_a":   {str(k): list(v) for k, v in clicks_a.items()},
        "clicks_b":   {str(k): list(v) for k, v in clicks_b.items()},
        "common":     common,
    }
    with open(OUTPUT_DIR / f"clicks_{pair_key}.json", "w") as f:
        json.dump(clicks_data, f, indent=2)

    # Save homography
    np.save(str(HOMOGRAPHY_DIR / f"H_{pair_key}.npy"), H)
    print(f"  Saved: H_{pair_key}.npy")

    # Step 5: build panorama preview
    print(f"\n  --- BUILDING PANORAMA PREVIEW ---")
    panorama = build_pair_panorama(map_a, map_b, H)
    panorama_path = _panorama_path(pair_key)
    save_image(panorama_path, panorama)
    print(f"  Saved: {panorama_path.name} ({panorama.shape[1]} x {panorama.shape[0]})")

    # Step 6: build click visualisation
    vis = visualise_clicks(map_a, map_b, clicks_a, clicks_b, common)
    vis_path = _matches_path(pair_key)
    save_image(vis_path, vis)
    print(f"  Saved: {vis_path.name}")

    print(f"\n{'='*70}")
    print(f"  PAIR {pair_key} COMPLETE")
    print(f"{'='*70}")
    print(f"  Open the panorama to verify alignment:")
    print(f"    {panorama_path.resolve()}")
    print(f"  And the side-by-side matches:")
    print(f"    {vis_path.resolve()}")
    print(f"{'='*70}\n")
    return True


def process_all():
    print("\n" + "=" * 70)
    print("  CLICK LOCATOR — ALL PAIRS")
    print("=" * 70)
    for pair_key in PAIR_PARCELS:
        process_pair(pair_key)
        # Pause between pairs in case user wants to stop
        print("\nPress Enter to continue to next pair, or Ctrl+C to stop...")
        try:
            input()
        except KeyboardInterrupt:
            print("\nStopped by user.")
            return


# ---------------------------------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Click-based parcel locator for cadastral map registration"
    )
    parser.add_argument("--pair", type=str, default=None,
                        help="Specific pair to process, e.g. --pair 45_47")
    parser.add_argument("--all", action="store_true",
                        help="Process all pairs")
    parser.add_argument("--rebuild", action="store_true",
                        help="Skip clicking; load saved clicks_<pair>.json "
                             "and rebuild the homography + panorama.")
    parser.add_argument("--source", type=str, default="clean",
                        choices=["clean", "binary"],
                        help="Which preprocessed image to use. Click data "
                             "is shared between sources, but the panorama "
                             "and match-viz outputs are namespaced per "
                             "source so you can compare results.")
    args = parser.parse_args()

    _set_source(args.source)

    if args.pair:
        process_pair(args.pair, rebuild_only=args.rebuild)
    elif args.all:
        process_all()
    else:
        print("Usage:")
        print("  python parcel_click_locator.py --pair 45_47   (test)")
        print("  python parcel_click_locator.py --all          (all 10 pairs)")
        print("\nAvailable pairs and their boundary parcels:")
        for k, v in PAIR_PARCELS.items():
            print(f"  {k}: {v}")
