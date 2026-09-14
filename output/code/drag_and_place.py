"""
=============================================================================
DRAG-AND-PLACE TOOL v3 — Visual Map Alignment
=============================================================================
Masters Thesis — AI-Based Panoramic Cadastral Image Reconstruction
Author  : Hussein Chalhoub

What's new in v3:
  - Auto-crop of white background around each map (only the actual content
    area is shown, no white margins)
  - Fine rotation: [ and ] keys rotate by 0.5 degrees, Shift+[ ] by 5 deg.
    (R key still does coarse 90 degree rotations)
  - Saved preview has NO coloured borders or labels - just the clean
    placement of map A and map B on white background.
  - Boundary parcel numbers shown in the HUD - no more switching to Excel
  - Per-pair info banner with adjacency context

How to use
-----------
  python drag_and_place.py --pair 45_46

Controls:
  Left click + drag (on B)  : move map B
  Right click + drag        : pan the view
  Scroll wheel              : zoom in / out
  R                         : rotate B by 90 deg CCW
  [ / ]                     : rotate B by 0.5 deg (Shift+[ or Shift+] = 5 deg)
  Arrow keys                : nudge B (Shift = 10 px, Ctrl = 50 px)
  Space                     : toggle B opacity (50% / 100%)
  G                         : toggle grid lines
  F                         : fit view to canvas
  +/-                       : keyboard zoom
  S                         : save placement
  Q or Esc                  : quit (auto-saves)

Output
-------
  output/homographies/H_<A>_<B>.npy           : 3x3 H = T @ R_full
  output/homographies/homographies.json       : updated metadata
  output/matches/map_<A>_<B>_dragplace.png    : clean preview (NO borders)
=============================================================================
"""

import argparse
import json
import sys
import cv2
import numpy as np
from pathlib import Path


# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------

PREPROCESSED_DIR = Path("output/preprocessed")
HOMOGRAPHY_DIR   = Path("output/homographies")
MATCHES_DIR      = Path("output/matches")

WINDOW_W = 1500
WINDOW_H = 950
DISPLAY_SCALE = 0.30

FONT             = cv2.FONT_HERSHEY_SIMPLEX
COL_BORDER_A     = (255, 80, 0)
COL_BORDER_B     = (0, 140, 255)
COL_HUD_BG       = (30, 30, 30)
COL_HUD_FG       = (255, 255, 255)
COL_GRID         = (200, 200, 200)

# Auto-crop: pixels with intensity below this are considered "content"
# (cadastral maps have dark ink on light paper).  Any column or row that
# contains no dark pixels is part of the white margin and gets trimmed.
CROP_INK_THRESHOLD = 200    # pixel value <= 200 is "ink/content"
CROP_MIN_INK_FRAC  = 0.001  # at least 0.1% of pixels must be dark

# Boundary parcels per pair (from the PDF index)
BOUNDARY_PARCELS = {
    "45_47": [2580],
    "45_46": [2616, 2619],
    "47_48": [2749, 2803, 2814],
    "48_49": [2893],
    "49_50": [3022, 3054],
    "50_51": [3068, 3813, 3866],
    "52_53": [3215],
    "52_54": [3216],
    "52_55": [3217],
    "53_54": [3302],
    "54_55": [3338, 3339, 3345, 3346],
}


# ---------------------------------------------------------------------------
# UTILITIES
# ---------------------------------------------------------------------------

def read_image(path: Path) -> np.ndarray:
    data = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Cannot read: {path}")
    return img


def save_image(path: Path, img: np.ndarray):
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, enc = cv2.imencode(path.suffix, img)
    if ok:
        enc.tofile(str(path))


def auto_crop_white(img: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """
    Auto-crop white margins around the map content.

    Algorithm:
      1. Find rows/columns that contain at least CROP_MIN_INK_FRAC of dark pixels
      2. Crop to the bounding box of those rows/columns
      3. Add a small padding so we don't clip ink right at the edge

    Returns:
      cropped_img         : the trimmed image
      bbox                : (x0, y0, x1, y1) in original image coordinates
                              the crop offset (x0, y0) is needed to map saved
                              homography back to original-image coords
    """
    h, w = img.shape
    is_ink = img <= CROP_INK_THRESHOLD

    row_density = is_ink.mean(axis=1)
    col_density = is_ink.mean(axis=0)

    rows_with_ink = np.where(row_density >= CROP_MIN_INK_FRAC)[0]
    cols_with_ink = np.where(col_density >= CROP_MIN_INK_FRAC)[0]

    if len(rows_with_ink) == 0 or len(cols_with_ink) == 0:
        return img, (0, 0, w, h)

    pad = 20
    y0 = max(0, int(rows_with_ink[0]) - pad)
    y1 = min(h, int(rows_with_ink[-1]) + pad + 1)
    x0 = max(0, int(cols_with_ink[0]) - pad)
    x1 = min(w, int(cols_with_ink[-1]) + pad + 1)

    return img[y0:y1, x0:x1], (x0, y0, x1, y1)


def load_pair_metadata(map_a: str, map_b: str) -> dict:
    p = HOMOGRAPHY_DIR / "homographies.json"
    if not p.exists():
        return {}
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f).get(f"{map_a}_{map_b}", {})


def save_homography(map_a: str, map_b: str,
                     H_full: np.ndarray,
                     rotation_b: float,
                     b_offset_full: tuple[int, int]):
    HOMOGRAPHY_DIR.mkdir(parents=True, exist_ok=True)
    np.save(str(HOMOGRAPHY_DIR / f"H_{map_a}_{map_b}.npy"), H_full)

    p = HOMOGRAPHY_DIR / "homographies.json"
    all_data = {}
    if p.exists():
        with open(p, "r", encoding="utf-8") as f:
            all_data = json.load(f)
    key = f"{map_a}_{map_b}"
    if key not in all_data:
        all_data[key] = {}
    all_data[key]["map_a"]          = map_a
    all_data[key]["map_b"]          = map_b
    all_data[key]["rotation_b_deg"] = float(rotation_b)
    all_data[key]["b_offset"]       = list(b_offset_full)
    all_data[key]["H"]              = H_full.tolist()
    all_data[key]["refinement"]     = {"method": "drag_and_place_v3"}
    with open(p, "w", encoding="utf-8") as f:
        json.dump(all_data, f, indent=2, ensure_ascii=False)
    print(f"  Saved: H_{map_a}_{map_b}.npy + homographies.json")


def rotate_image_arbitrary(img: np.ndarray, angle_deg: float
                             ) -> tuple[np.ndarray, np.ndarray]:
    """
    Rotate an image by an arbitrary angle CCW around its centre.
    Expands the canvas so no content is clipped.

    Returns:
      rotated       : the rotated image
      M_3x3         : 3x3 homogeneous matrix mapping ORIGINAL pixel coords
                      (in the input image) to coords in `rotated`.
                      Use this to compute the full-resolution H.
    """
    h, w = img.shape
    angle_rad = np.deg2rad(angle_deg)
    cos_a = abs(np.cos(angle_rad))
    sin_a = abs(np.sin(angle_rad))
    new_w = int(w * cos_a + h * sin_a)
    new_h = int(w * sin_a + h * cos_a)

    cx, cy = w / 2.0, h / 2.0
    M_2x3 = cv2.getRotationMatrix2D((cx, cy), angle_deg, 1.0)
    # Adjust translation so the rotated image is centred in the new canvas
    M_2x3[0, 2] += (new_w - w) / 2.0
    M_2x3[1, 2] += (new_h - h) / 2.0

    rotated = cv2.warpAffine(
        img, M_2x3, (new_w, new_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=255
    )
    M_3x3 = np.vstack([M_2x3, [0, 0, 1]]).astype(np.float64)
    return rotated, M_3x3


# ---------------------------------------------------------------------------
# VIEW (zoom + pan)
# ---------------------------------------------------------------------------

class View:
    def __init__(self, canvas_w: int, canvas_h: int,
                 win_w: int, win_h: int):
        self.canvas_w = canvas_w
        self.canvas_h = canvas_h
        self.win_w = win_w
        self.win_h = win_h
        self.fit()

    def fit(self):
        self.scale = min(self.win_w / self.canvas_w,
                         self.win_h / self.canvas_h)
        self.pan_x = (self.win_w - self.canvas_w * self.scale) / 2
        self.pan_y = (self.win_h - self.canvas_h * self.scale) / 2

    def update_canvas_size(self, new_w: int, new_h: int):
        # Preserve current centre/zoom when canvas grows due to free rotation
        old_cx = (self.win_w / 2 - self.pan_x) / self.scale
        old_cy = (self.win_h / 2 - self.pan_y) / self.scale
        self.canvas_w = new_w
        self.canvas_h = new_h
        # Re-centre on same canvas point
        self.pan_x = self.win_w / 2 - old_cx * self.scale
        self.pan_y = self.win_h / 2 - old_cy * self.scale

    def window_to_canvas(self, wx, wy):
        return ((wx - self.pan_x) / self.scale,
                (wy - self.pan_y) / self.scale)

    def zoom_at(self, factor, wx, wy):
        cx, cy = self.window_to_canvas(wx, wy)
        self.scale = max(0.05, min(8.0, self.scale * factor))
        self.pan_x = wx - cx * self.scale
        self.pan_y = wy - cy * self.scale

    def render(self, canvas: np.ndarray) -> np.ndarray:
        cx0, cy0 = self.window_to_canvas(0, 0)
        cx1, cy1 = self.window_to_canvas(self.win_w, self.win_h)
        cx0c = int(max(0, cx0)); cy0c = int(max(0, cy0))
        cx1c = int(min(self.canvas_w, cx1)); cy1c = int(min(self.canvas_h, cy1))
        if cx1c <= cx0c or cy1c <= cy0c:
            return np.zeros((self.win_h, self.win_w, 3), dtype=np.uint8)
        crop = canvas[cy0c:cy1c, cx0c:cx1c]
        sw = max(1, int((cx1c - cx0c) * self.scale))
        sh = max(1, int((cy1c - cy0c) * self.scale))
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
# RENDER CANVAS
# ---------------------------------------------------------------------------

def render_canvas(img_a_disp: np.ndarray,
                    img_b_disp: np.ndarray,
                    a_pos: tuple[int, int],
                    b_pos: tuple[int, int],
                    show_grid: bool,
                    b_alpha: float,
                    canvas_w: int, canvas_h: int,
                    overlay_order: str,
                    show_borders: bool = True) -> np.ndarray:
    canvas = np.ones((canvas_h, canvas_w, 3), dtype=np.uint8) * 255

    if show_grid:
        for x in range(0, canvas_w, 100):
            cv2.line(canvas, (x, 0), (x, canvas_h), COL_GRID, 1)
        for y in range(0, canvas_h, 100):
            cv2.line(canvas, (0, y), (canvas_w, y), COL_GRID, 1)

    bx, by = b_pos
    bh, bw = img_b_disp.shape
    ax, ay = a_pos
    ah, aw = img_a_disp.shape

    def draw_map_a():
        a_x0 = max(0, ax); a_y0 = max(0, ay)
        a_x1 = min(canvas_w, ax + aw); a_y1 = min(canvas_h, ay + ah)
        if a_x1 > a_x0 and a_y1 > a_y0:
            a_region = img_a_disp[a_y0 - ay:a_y1 - ay, a_x0 - ax:a_x1 - ax]
            canvas[a_y0:a_y1, a_x0:a_x1] = cv2.cvtColor(a_region, cv2.COLOR_GRAY2BGR)

    def draw_map_b():
        b_x0 = max(0, bx); b_y0 = max(0, by)
        b_x1 = min(canvas_w, bx + bw); b_y1 = min(canvas_h, by + bh)
        if b_x1 > b_x0 and b_y1 > b_y0:
            b_region = img_b_disp[b_y0 - by:b_y1 - by, b_x0 - bx:b_x1 - bx]
            b_bgr = cv2.cvtColor(b_region, cv2.COLOR_GRAY2BGR)
            if b_alpha >= 1.0:
                canvas[b_y0:b_y1, b_x0:b_x1] = b_bgr
            else:
                existing = canvas[b_y0:b_y1, b_x0:b_x1].astype(np.float32)
                blended = existing * (1 - b_alpha) + b_bgr.astype(np.float32) * b_alpha
                canvas[b_y0:b_y1, b_x0:b_x1] = blended.astype(np.uint8)

    if overlay_order == "B":
        draw_map_a()
        draw_map_b()
    else:
        draw_map_b()
        draw_map_a()

    if show_borders:
        cv2.rectangle(canvas, (ax, ay), (ax + aw - 1, ay + ah - 1),
                       COL_BORDER_A, 4, cv2.LINE_AA)
        cv2.rectangle(canvas, (bx, by), (bx + bw - 1, by + bh - 1),
                       COL_BORDER_B, 4, cv2.LINE_AA)
    return canvas


def draw_hud(display: np.ndarray, view: View,
              map_a: str, map_b: str,
              rotation_b: float, b_offset_full: tuple[int, int],
              b_alpha: float, overlay_order: str, over_top: bool,
              boundary_parcels: list[int]):
    win_w, win_h = view.win_w, view.win_h

    # Top status bar
    parcels_str = ", ".join(str(p) for p in boundary_parcels) if boundary_parcels else "none"
    status1 = (f"Pair {map_a}_{map_b}  |  Boundary parcels: {parcels_str}")
    status2 = (f"rot={rotation_b:.1f}deg  |  "
               f"B offset (full res): ({b_offset_full[0]}, {b_offset_full[1]})  |  "
               f"alpha={b_alpha:.1f}  |  zoom={view.scale:.2f}x  |  "
               f"Layer: {overlay_order} over other  |  "
               f"{'ON TOP MAP (drag will work)' if over_top else ''}")
    cv2.rectangle(display, (0, 0), (win_w, 56), COL_HUD_BG, -1)
    cv2.putText(display, status1, (10, 22), FONT, 0.55, (180, 220, 255), 1, cv2.LINE_AA)
    cv2.putText(display, status2, (10, 46), FONT, 0.5, COL_HUD_FG, 1, cv2.LINE_AA)

    # Bottom controls bar
    hud = ("[Drag B]  [Right-drag = pan]  [Scroll = zoom]  "
           "[R = rot 90]  [ [ / ] = rot 0.5deg ]  [Arrows = nudge]  "
            "[Space = alpha]  [G = grid]  [T = order]  [F = fit]  [S = save]  [Q = quit]")
    cv2.rectangle(display, (0, win_h - 26), (win_w, win_h), COL_HUD_BG, -1)
    cv2.putText(display, hud, (10, win_h - 8), FONT, 0.42, COL_HUD_FG, 1, cv2.LINE_AA)


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def run(map_a: str, map_b: str):
    print(f"\n  Loading pair {map_a}_{map_b}...")
    full_a = read_image(PREPROCESSED_DIR / f"map_{map_a}_clean.png")
    full_b = read_image(PREPROCESSED_DIR / f"map_{map_b}_clean.png")
    H_A_full_orig, W_A_full_orig = full_a.shape
    H_B_full_orig, W_B_full_orig = full_b.shape
    print(f"  Map A original: {full_a.shape}  |  Map B original: {full_b.shape}")

    # Auto-crop white margins from both maps
    full_a_cropped, a_bbox = auto_crop_white(full_a)
    full_b_cropped, b_bbox = auto_crop_white(full_b)
    print(f"  Map A cropped: {full_a_cropped.shape}  bbox={a_bbox}")
    print(f"  Map B cropped: {full_b_cropped.shape}  bbox={b_bbox}")

    # The crop offsets (a_bbox[0], a_bbox[1]) represent where the cropped
    # image's (0,0) was in the original.  We need this to convert the
    # final homography back to original-image coordinates.
    a_crop_offset = (a_bbox[0], a_bbox[1])
    b_crop_offset = (b_bbox[0], b_bbox[1])

    H_A_crop, W_A_crop = full_a_cropped.shape
    H_B_crop, W_B_crop = full_b_cropped.shape

    # Boundary parcels for this pair (for HUD display)
    pair_key = f"{map_a}_{map_b}"
    parcels = BOUNDARY_PARCELS.get(pair_key, [])

    # Load any previous placement
    meta = load_pair_metadata(map_a, map_b)
    rotation_b = float(meta.get("rotation_b_deg", 0.0))
    b_offset_full = tuple(meta.get("b_offset", [0, H_A_full_orig + 200]))
    if "b_offset" in meta:
        print(f"  Loaded previous: rot={rotation_b}deg, offset={b_offset_full}")

    # Convert full-res offset (relative to original A's frame) to a
    # display-space offset relative to A_cropped.  When loading a previous
    # placement we account for both crop offsets so the position is consistent.
    def offset_full_to_b_pos(off_full):
        # off_full is the translation of B's ORIGINAL pixel (0,0) in A_ORIGINAL frame
        # We want b_pos: where B_cropped is placed on the canvas (which uses A_cropped at margin)
        # Convert "B original (0,0) in A original frame" to display canvas coords:
        # Step 1: subtract A's crop offset because canvas anchor of A is margin,
        #         not (0,0) of original
        x_in_a_cropped_full = off_full[0] - a_crop_offset[0]
        y_in_a_cropped_full = off_full[1] - a_crop_offset[1]
        # Step 2: scale to display
        x_in_canvas = a_pos[0] + int(x_in_a_cropped_full * DISPLAY_SCALE)
        y_in_canvas = a_pos[1] + int(y_in_a_cropped_full * DISPLAY_SCALE)
        # NOTE: we do NOT subtract b_crop_offset here - the user dragged the
        # cropped image as a whole and the homography accounts for B's crop
        # internally when computing the final H below.
        return (x_in_canvas, y_in_canvas)

    def b_pos_to_offset_full(pos):
        x_in_a_cropped_full = (pos[0] - a_pos[0]) / DISPLAY_SCALE
        y_in_a_cropped_full = (pos[1] - a_pos[1]) / DISPLAY_SCALE
        x_full = int(x_in_a_cropped_full + a_crop_offset[0])
        y_full = int(y_in_a_cropped_full + a_crop_offset[1])
        return (x_full, y_full)

    # Downscale the cropped maps for display
    a_disp = cv2.resize(
        full_a_cropped,
        (int(W_A_crop * DISPLAY_SCALE), int(H_A_crop * DISPLAY_SCALE)),
        interpolation=cv2.INTER_AREA
    )

    # Cache for rotated B images at different angles
    rotation_cache = {}

    def get_b_disp_and_M(rot: float):
        """Return display-size rotated B + 3x3 matrix in FULL CROPPED resolution."""
        key = round(rot, 2)
        if key in rotation_cache:
            return rotation_cache[key]
        # Rotate cropped B
        rotated_full, M_3x3 = rotate_image_arbitrary(full_b_cropped, rot)
        rotated_disp = cv2.resize(
            rotated_full,
            (int(rotated_full.shape[1] * DISPLAY_SCALE),
             int(rotated_full.shape[0] * DISPLAY_SCALE)),
            interpolation=cv2.INTER_AREA
        )
        rotation_cache[key] = (rotated_disp, M_3x3, rotated_full.shape)
        return rotation_cache[key]

    b_disp, R_B, b_full_rot_shape = get_b_disp_and_M(rotation_b)
    print(f"  Display: A={a_disp.shape}  B_initial={b_disp.shape}")

    margin = 100
    a_pos = (margin, margin)

    canvas_w = a_disp.shape[1] + max(b_disp.shape[1], a_disp.shape[1]) + 4 * margin
    canvas_h = a_disp.shape[0] + max(b_disp.shape[0], a_disp.shape[0]) + 4 * margin

    b_pos = offset_full_to_b_pos(b_offset_full)

    show_grid = False
    b_alpha = 1.0
    overlay_order = 'B'   # 'B' = B over A (default), 'A' = A over B
    needs_redraw = True

    dragging_a = False
    dragging_b = False
    dragging_pan = False
    drag_start_win = None
    drag_start_a   = None
    drag_start_b   = None
    drag_start_pan = None
    mouse_canvas = (0.0, 0.0)

    view = View(canvas_w, canvas_h, WINDOW_W, WINDOW_H)
    win_name = f"Drag and place: {map_a} <-> {map_b}"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win_name, WINDOW_W, WINDOW_H)

    print(f"\n  Boundary parcels for this pair: {parcels}")
    print(f"  Drag map B with LEFT mouse, right-drag pan, scroll = zoom")
    print(f"  T = toggle overlay order (A over B / B over A)")
    print(f"  R = 90deg rotate, [ / ] = 0.5deg fine rotate")
    print(f"  Arrows = nudge B (Shift = 10px)")
    print(f"  S = save, Q = quit\n")

    def is_in_a(canvas_xy):
        return (a_pos[0] <= canvas_xy[0] < a_pos[0] + a_disp.shape[1] and
                a_pos[1] <= canvas_xy[1] < a_pos[1] + a_disp.shape[0])

    def is_in_b(canvas_xy):
        return (b_pos[0] <= canvas_xy[0] < b_pos[0] + b_disp.shape[1] and
                b_pos[1] <= canvas_xy[1] < b_pos[1] + b_disp.shape[0])

    def on_mouse(event, x, y, flags, param):
        nonlocal dragging_a, dragging_b, dragging_pan
        nonlocal drag_start_win, drag_start_a, drag_start_b, drag_start_pan
        nonlocal a_pos, b_pos, needs_redraw, mouse_canvas
        mouse_canvas = view.window_to_canvas(x, y)
        needs_redraw = True
        if event == cv2.EVENT_LBUTTONDOWN:
            if overlay_order == 'B' and is_in_b(mouse_canvas):
                dragging_b = True
                drag_start_win = (x, y)
                drag_start_b   = b_pos
            elif overlay_order == 'A' and is_in_a(mouse_canvas):
                dragging_a = True
                drag_start_win = (x, y)
                drag_start_a   = a_pos
        elif event == cv2.EVENT_RBUTTONDOWN:
            dragging_pan = True
            drag_start_win = (x, y)
            drag_start_pan = (view.pan_x, view.pan_y)
        elif event == cv2.EVENT_MOUSEMOVE:
            if dragging_a:
                dwx = x - drag_start_win[0]
                dwy = y - drag_start_win[1]
                a_pos = (int(drag_start_a[0] + dwx / view.scale),
                         int(drag_start_a[1] + dwy / view.scale))
            elif dragging_b:
                dwx = x - drag_start_win[0]
                dwy = y - drag_start_win[1]
                b_pos = (int(drag_start_b[0] + dwx / view.scale),
                         int(drag_start_b[1] + dwy / view.scale))
            elif dragging_pan:
                dwx = x - drag_start_win[0]
                dwy = y - drag_start_win[1]
                view.pan_x = drag_start_pan[0] + dwx
                view.pan_y = drag_start_pan[1] + dwy
        elif event == cv2.EVENT_LBUTTONUP:
            dragging_a = False
            dragging_b = False
        elif event == cv2.EVENT_RBUTTONUP:
            dragging_pan = False
        elif event == cv2.EVENT_MOUSEWHEEL:
            factor = 1.2 if flags > 0 else 1 / 1.2
            view.zoom_at(factor, x, y)

    cv2.setMouseCallback(win_name, on_mouse)

    def update_b_after_rotation():
        """Called after rotation_b changes - update b_disp, R_B, and canvas size."""
        nonlocal b_disp, R_B, b_full_rot_shape, canvas_w, canvas_h
        b_disp, R_B, b_full_rot_shape = get_b_disp_and_M(rotation_b)
        new_canvas_w = a_disp.shape[1] + max(b_disp.shape[1], a_disp.shape[1]) + 4 * margin
        new_canvas_h = a_disp.shape[0] + max(b_disp.shape[0], a_disp.shape[0]) + 4 * margin
        if new_canvas_w != canvas_w or new_canvas_h != canvas_h:
            canvas_w = new_canvas_w
            canvas_h = new_canvas_h
            view.update_canvas_size(canvas_w, canvas_h)

    def compute_full_homography():
        """
        Build the homography in FULL ORIGINAL resolution coordinates.

        Pipeline:
          B_original_pixel
            -> B_cropped pixel  (subtract b_crop_offset)
            -> B_rotated_cropped pixel  (apply R_B)
            -> A_cropped pixel  (translate by user-set offset in cropped space)
            -> A_original pixel  (add a_crop_offset)
        """
        # Compute B_offset in CROPPED A space (from current b_pos)
        b_off_in_cropped_x = (b_pos[0] - a_pos[0]) / DISPLAY_SCALE
        b_off_in_cropped_y = (b_pos[1] - a_pos[1]) / DISPLAY_SCALE

        # Step 1: subtract b_crop_offset (B_original -> B_cropped)
        T_unc_b = np.array([[1, 0, -b_crop_offset[0]],
                            [0, 1, -b_crop_offset[1]],
                            [0, 0, 1]], dtype=np.float64)
        # Step 2: R_B (B_cropped -> B_rotated_cropped)
        # Step 3: translate to where user placed it in cropped A frame
        T_place = np.array([[1, 0, b_off_in_cropped_x],
                            [0, 1, b_off_in_cropped_y],
                            [0, 0, 1]], dtype=np.float64)
        # Step 4: add a_crop_offset (A_cropped -> A_original)
        T_a_back = np.array([[1, 0, a_crop_offset[0]],
                             [0, 1, a_crop_offset[1]],
                             [0, 0, 1]], dtype=np.float64)

        H = T_a_back @ T_place @ R_B @ T_unc_b
        return H

    while True:
        if needs_redraw:
            canvas = render_canvas(
                a_disp, b_disp, a_pos, b_pos,
                show_grid, b_alpha, canvas_w, canvas_h,
                overlay_order,
                show_borders=True
            )
            display = view.render(canvas)
            b_offset_full = b_pos_to_offset_full(b_pos)
            over_top = is_in_b(mouse_canvas) if overlay_order == 'B' else is_in_a(mouse_canvas)
            draw_hud(display, view, map_a, map_b,
                      rotation_b, b_offset_full, b_alpha,
                      overlay_order, over_top, parcels)
            cv2.imshow(win_name, display)
            needs_redraw = False

        key = cv2.waitKey(20) & 0xFFFF
        if key == 0xFFFF:
            continue
        kc = key & 0xFF
        needs_redraw = True

        if kc in (ord('q'), ord('Q'), 27):
            break
        elif kc in (ord('r'), ord('R')):
            rotation_b = (rotation_b + 90) % 360
            update_b_after_rotation()
            print(f"  Rotation -> {rotation_b}deg")
        elif kc == ord('['):
            rotation_b = (rotation_b + 0.5) % 360
            update_b_after_rotation()
        elif kc == ord(']'):
            rotation_b = (rotation_b - 0.5) % 360
            update_b_after_rotation()
        elif kc == ord('{'):
            rotation_b = (rotation_b + 5) % 360
            update_b_after_rotation()
        elif kc == ord('}'):
            rotation_b = (rotation_b - 5) % 360
            update_b_after_rotation()
        elif kc == ord(' '):
            b_alpha = 0.5 if b_alpha >= 1.0 else 1.0
        elif kc in (ord('g'), ord('G')):
            show_grid = not show_grid
        elif kc in (ord('t'), ord('T')):
            overlay_order = 'A' if overlay_order == 'B' else 'B'
            print(f"  Overlay order -> {overlay_order} over other")
        elif kc in (ord('f'), ord('F')):
            view.fit()
        elif kc in (ord('+'), ord('=')):
            view.zoom_at(1.25, WINDOW_W // 2, WINDOW_H // 2)
        elif kc == ord('-'):
            view.zoom_at(0.8, WINDOW_W // 2, WINDOW_H // 2)
        elif kc in (ord('s'), ord('S')):
            H = compute_full_homography()
            save_homography(map_a, map_b, H, rotation_b, b_offset_full)
            # Save preview WITHOUT borders (clean output)
            preview = render_canvas(a_disp, b_disp, a_pos, b_pos,
                                      False, 1.0, canvas_w, canvas_h,
                                      overlay_order,
                                      show_borders=False)
            save_image(MATCHES_DIR / f"map_{map_a}_{map_b}_match.png", preview)
            print(f"  SAVED placement.")
        elif kc == 81 or key == 0x250000:  # left arrow
            step = 10 if (key & 0x10000) else 1
            b_pos = (b_pos[0] - step, b_pos[1])
        elif kc == 82 or key == 0x260000:  # up arrow
            step = 10 if (key & 0x10000) else 1
            b_pos = (b_pos[0], b_pos[1] - step)
        elif kc == 83 or key == 0x270000:  # right arrow
            step = 10 if (key & 0x10000) else 1
            b_pos = (b_pos[0] + step, b_pos[1])
        elif kc == 84 or key == 0x280000:  # down arrow
            step = 10 if (key & 0x10000) else 1
            b_pos = (b_pos[0], b_pos[1] + step)

    # Auto-save on exit
    H = compute_full_homography()
    save_homography(map_a, map_b, H, rotation_b, b_offset_full)
    preview = render_canvas(a_disp, b_disp, a_pos, b_pos,
                              False, 1.0, canvas_w, canvas_h,
                              overlay_order,
                              show_borders=False)
    save_image(MATCHES_DIR / f"map_{map_a}_{map_b}_match.png", preview)
    print(f"\n  Final placement saved. Done.")
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Drag-and-place v3")
    parser.add_argument("--pair", required=True, help="e.g. --pair 45_46")
    args = parser.parse_args()
    if "_" not in args.pair:
        print("ERROR: --pair must be A_B (e.g. 45_46)")
        sys.exit(1)
    a, b = args.pair.split("_")
    run(a, b)