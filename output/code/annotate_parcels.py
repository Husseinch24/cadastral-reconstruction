"""
=============================================================================
MANUAL ANNOTATION TOOL — Boundary Parcel Centroid Labelling
=============================================================================
Masters Thesis — AI-Based Panoramic Cadastral Image Reconstruction
Author  : Hussein Chalhoub

Purpose
--------
OCR engines (Tesseract, EasyOCR) cannot read handwritten mirrored
Arabic-Indic numerals on aged blueprint paper.  This tool lets you
manually click on each boundary parcel and type its number.  The output
JSON files are identical in format to what Step 5 would have produced,
so Step 4 Stage 2 activates automatically when you re-run step4_matching.py.

You only need to annotate the BOUNDARY PARCELS — the ones that appear on
two map sheets.  These are listed in the table below.  You do NOT need
to annotate every parcel on every map.

Boundary parcels to annotate (from your PDF index)
---------------------------------------------------
  Maps 45 & 47 : parcel 2580
  Maps 45 & 46 : parcels 2616, 2619
  Maps 47 & 48 : parcels 2749, 2803, 2814
  Maps 48 & 49 : parcel 2893
  Maps 49 & 50 : parcels 3022, 3054
  Maps 50 & 51 : parcels 3068, 3813, 3866
  Maps 52 & 53 : parcel 3215
  Maps 52 & 54 : parcel 3216
  Maps 52 & 55 : parcel 3217
  Maps 54 & 55 : parcels 3338, 3339, 3345, 3346

For each parcel ID, you need to annotate it on BOTH maps it appears on.
Example: parcel 2580 → annotate on map 45 AND on map 47.

How to use
-----------
  python annotate_parcels.py --map 45

Controls:
  Left click        : place a marker at that position
  Type number       : enter the parcel number (shown in the title bar)
  Enter             : confirm and save the annotation
  Backspace         : delete last digit typed
  Right click       : remove the nearest existing marker
  S                 : save current annotations to JSON
  Q or Esc          : quit and save

Navigation:
  Scroll wheel      : zoom in/out around cursor
  Middle click drag : pan the view
  Arrow keys        : pan slowly

Output
-------
  output/detected/map_<N>_parcels.json — same format as Step 5 output,
  read directly by Step 4 Stage 2 without any changes.

Running order
--------------
  1. python annotate_parcels.py --map 45   (annotate boundary parcels on map 45)
  2. python annotate_parcels.py --map 47   (annotate boundary parcels on map 47)
  3. ... repeat for all maps that have boundary parcels ...
  4. python step4_matching.py              (Stage 2 activates automatically)
  5. python step6_panorama.py             (final panorama)

=============================================================================
"""

import argparse
import json
import math
import sys
import cv2
import numpy as np
from pathlib import Path


# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------

PREPROCESSED_DIR = Path("output/preprocessed")
DETECTED_DIR     = Path("output/detected")

# Which parcels each map should contain (for the title bar hint)
BOUNDARY_PARCELS_BY_MAP = {
    "45": [2580, 2616, 2619],
    "46": [2616, 2619],
    "47": [2580, 2749, 2803, 2814],
    "48": [2749, 2803, 2814, 2893],
    "49": [2893, 3022, 3054],
    "50": [3022, 3054, 3068, 3813, 3866],
    "51": [3068, 3813, 3866],
    "52": [3215, 3216, 3217],
    "53": [3215],
    "54": [3216, 3338, 3339, 3345, 3346],
    "55": [3217, 3338, 3339, 3345, 3346],
}

PARCEL_RANGE_PER_MAP = {
    "43": (2477, 2500),  "44": (2451, 2476),
    "45": (2501, 2750),  "46": (2751, 3000),
    "47": (2701, 2750),  "48": (2751, 3000),
    "49": (2951, 3250),  "50": (3201, 3500),
    "51": (3051, 3250),  "52": (3201, 3500),
    "53": (3251, 3500),  "54": (3301, 3500),
    "55": (3351, 3500),
}
BORDER_TOLERANCE = 50

# Display settings
WINDOW_W = 1280
WINDOW_H = 900
MARKER_RADIUS   = 18
FONT            = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE      = 0.7
FONT_THICKNESS  = 2


# ---------------------------------------------------------------------------
# COLOURS (BGR)
# ---------------------------------------------------------------------------
COL_MARKER_PENDING  = (0, 180, 255)   # orange — placed but not confirmed
COL_MARKER_VALID    = (0, 200, 0)     # green  — confirmed VALID
COL_MARKER_BORDER   = (0, 220, 255)   # cyan   — confirmed BORDER
COL_MARKER_INVALID  = (0, 0, 220)     # red    — confirmed but out of range
COL_TEXT_BG         = (30, 30, 30)
COL_TEXT_FG         = (255, 255, 255)
COL_CROSSHAIR       = (200, 200, 200)


# ---------------------------------------------------------------------------
# VALIDATION
# ---------------------------------------------------------------------------

def validate(number: int, map_num: str) -> str:
    if map_num not in PARCEL_RANGE_PER_MAP:
        return "UNKNOWN"
    lo, hi = PARCEL_RANGE_PER_MAP[map_num]
    if lo <= number <= hi:
        return "VALID"
    if (lo - BORDER_TOLERANCE) <= number <= (hi + BORDER_TOLERANCE):
        return "BORDER"
    return "INVALID"


def validate_colour(status: str):
    return {
        "VALID":   COL_MARKER_VALID,
        "BORDER":  COL_MARKER_BORDER,
        "INVALID": COL_MARKER_INVALID,
    }.get(status, COL_TEXT_FG)


# ---------------------------------------------------------------------------
# UTILITY
# ---------------------------------------------------------------------------

def read_image(path: Path) -> np.ndarray:
    data = np.fromfile(str(path), dtype=np.uint8)
    img  = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Cannot read: {path}")
    return img


def load_existing_json(map_num: str) -> list[dict]:
    """Load annotations already saved for this map (for resuming sessions)."""
    path = DETECTED_DIR / f"map_{map_num}_parcels.json"
    if not path.exists():
        return []
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    # Keep only manually annotated entries (detected_number is not None)
    return [p for p in data if p.get("detected_number") is not None]


def save_json(map_num: str, annotations: list[dict]):
    """
    Save annotations to JSON in the same format Step 5 would produce.
    Step 4 Stage 2 reads this file — field names must match exactly.
    """
    DETECTED_DIR.mkdir(parents=True, exist_ok=True)
    output = []
    for i, a in enumerate(annotations):
        num = a["detected_number"]
        output.append({
            "parcel_id":       i,
            "detected_number": num,
            "cx":              round(float(a["cx"]), 2),
            "cy":              round(float(a["cy"]), 2),
            "confidence":      1.0,           # manual = perfect confidence
            "validation":      validate(num, map_num),
            "area_px":         0.0,           # not available from manual annotation
            "bbox":            [0, 0, 0, 0],  # not available from manual annotation
        })
    path = DETECTED_DIR / f"map_{map_num}_parcels.json"
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"  Saved {len(output)} annotations → {path}")


# ---------------------------------------------------------------------------
# VIEW STATE (pan + zoom)
# ---------------------------------------------------------------------------

class ViewState:
    """
    Manages the mapping between screen coordinates and image coordinates.
    Allows smooth pan and zoom over the large map images.
    """
    def __init__(self, img_h: int, img_w: int,
                 win_h: int, win_w: int):
        self.img_h = img_h
        self.img_w = img_w
        self.win_h = win_h
        self.win_w = win_w

        # Initial zoom: fit image in window
        self.scale = min(win_w / img_w, win_h / img_h)
        # Pan: offset of image top-left in screen coords
        self.pan_x = (win_w - img_w * self.scale) / 2
        self.pan_y = (win_h - img_h * self.scale) / 2

        self._drag_start  = None
        self._drag_pan    = None

    def img_to_screen(self, ix, iy):
        sx = ix * self.scale + self.pan_x
        sy = iy * self.scale + self.pan_y
        return int(sx), int(sy)

    def screen_to_img(self, sx, sy):
        ix = (sx - self.pan_x) / self.scale
        iy = (sy - self.pan_y) / self.scale
        return ix, iy

    def zoom(self, factor: float, cx_screen: int, cy_screen: int):
        """Zoom in/out keeping the point under the cursor fixed."""
        ix, iy = self.screen_to_img(cx_screen, cy_screen)
        self.scale = max(0.05, min(20.0, self.scale * factor))
        self.pan_x = cx_screen - ix * self.scale
        self.pan_y = cy_screen - iy * self.scale

    def start_drag(self, sx, sy):
        self._drag_start = (sx, sy)
        self._drag_pan   = (self.pan_x, self.pan_y)

    def update_drag(self, sx, sy):
        if self._drag_start is None:
            return
        dx = sx - self._drag_start[0]
        dy = sy - self._drag_start[1]
        self.pan_x = self._drag_pan[0] + dx
        self.pan_y = self._drag_pan[1] + dy

    def end_drag(self):
        self._drag_start = None

    def pan(self, dpx, dpy):
        self.pan_x += dpx
        self.pan_y += dpy

    def render(self, img_bgr: np.ndarray) -> np.ndarray:
        """Render the current view into a WINDOW_H × WINDOW_W frame."""
        # Compute the region of the image visible in the window
        x0, y0 = self.screen_to_img(0, 0)
        x1, y1 = self.screen_to_img(self.win_w, self.win_h)

        x0c = int(max(0, x0))
        y0c = int(max(0, y0))
        x1c = int(min(self.img_w, x1))
        y1c = int(min(self.img_h, y1))

        if x1c <= x0c or y1c <= y0c:
            return np.zeros((self.win_h, self.win_w, 3), dtype=np.uint8)

        crop = img_bgr[y0c:y1c, x0c:x1c]

        # Scale the crop to its on-screen size
        sw = int((x1c - x0c) * self.scale)
        sh = int((y1c - y0c) * self.scale)
        if sw < 1 or sh < 1:
            return np.zeros((self.win_h, self.win_w, 3), dtype=np.uint8)

        interp = cv2.INTER_AREA if self.scale < 1 else cv2.INTER_LINEAR
        scaled  = cv2.resize(crop, (sw, sh), interpolation=interp)

        # Place into canvas
        canvas = np.zeros((self.win_h, self.win_w, 3), dtype=np.uint8)
        dst_x0 = int(self.pan_x + x0c * self.scale)
        dst_y0 = int(self.pan_y + y0c * self.scale)

        # Clip to canvas bounds
        src_x0 = max(0, -dst_x0)
        src_y0 = max(0, -dst_y0)
        dst_x0 = max(0, dst_x0)
        dst_y0 = max(0, dst_y0)

        h_copy = min(sh - src_y0, self.win_h - dst_y0)
        w_copy = min(sw - src_x0, self.win_w - dst_x0)

        if h_copy > 0 and w_copy > 0:
            canvas[dst_y0:dst_y0 + h_copy,
                   dst_x0:dst_x0 + w_copy] = \
                scaled[src_y0:src_y0 + h_copy,
                       src_x0:src_x0 + w_copy]
        return canvas


# ---------------------------------------------------------------------------
# ANNOTATION STATE
# ---------------------------------------------------------------------------

class AnnotationState:
    def __init__(self, map_num: str, existing: list[dict]):
        self.map_num     = map_num
        self.annotations = []   # confirmed annotations [{cx, cy, detected_number}]
        self.pending_cx  = None  # clicked position not yet confirmed
        self.pending_cy  = None
        self.typing      = ""    # digits being typed

        # Load existing annotations
        for a in existing:
            self.annotations.append({
                "cx":              float(a["cx"]),
                "cy":              float(a["cy"]),
                "detected_number": int(a["detected_number"]),
            })

    def click(self, ix: float, iy: float):
        """Place a pending marker at image coords (ix, iy)."""
        self.pending_cx = ix
        self.pending_cy = iy
        self.typing     = ""

    def type_digit(self, ch: str):
        if ch.isdigit() and len(self.typing) < 6:
            self.typing += ch

    def backspace(self):
        self.typing = self.typing[:-1]

    def confirm(self) -> bool:
        """Confirm the pending marker with the typed number."""
        if self.pending_cx is None or not self.typing:
            return False
        num = int(self.typing)
        self.annotations.append({
            "cx":              self.pending_cx,
            "cy":              self.pending_cy,
            "detected_number": num,
        })
        self.pending_cx = None
        self.pending_cy = None
        self.typing     = ""
        return True

    def remove_nearest(self, ix: float, iy: float, radius_img: float = 50):
        """Remove the annotation closest to (ix, iy) within radius_img pixels."""
        best_idx  = None
        best_dist = radius_img
        for i, a in enumerate(self.annotations):
            d = math.hypot(a["cx"] - ix, a["cy"] - iy)
            if d < best_dist:
                best_dist = d
                best_idx  = i
        if best_idx is not None:
            removed = self.annotations.pop(best_idx)
            print(f"  Removed annotation: parcel {removed['detected_number']} "
                  f"at ({removed['cx']:.0f}, {removed['cy']:.0f})")

    def cancel_pending(self):
        self.pending_cx = None
        self.pending_cy = None
        self.typing     = ""


# ---------------------------------------------------------------------------
# DRAWING
# ---------------------------------------------------------------------------

def draw_overlay(canvas: np.ndarray,
                  state: AnnotationState,
                  view: ViewState,
                  mouse_sx: int,
                  mouse_sy: int,
                  map_num: str):
    """Draw all markers, labels, and UI elements onto the canvas."""

    # Light crosshair at mouse position
    cv2.line(canvas, (mouse_sx, 0), (mouse_sx, view.win_h),
             COL_CROSSHAIR, 1, cv2.LINE_AA)
    cv2.line(canvas, (0, mouse_sy), (view.win_w, mouse_sy),
             COL_CROSSHAIR, 1, cv2.LINE_AA)

    # Confirmed annotations
    for a in state.annotations:
        sx, sy = view.img_to_screen(a["cx"], a["cy"])
        num    = a["detected_number"]
        status = validate(num, map_num)
        col    = validate_colour(status)

        cv2.circle(canvas, (sx, sy), MARKER_RADIUS, col, 2, cv2.LINE_AA)
        cv2.circle(canvas, (sx, sy), 3, col, -1, cv2.LINE_AA)

        label = str(num)
        (tw, th), _ = cv2.getTextSize(label, FONT, FONT_SCALE, FONT_THICKNESS)
        lx = sx - tw // 2
        ly = sy - MARKER_RADIUS - 6
        cv2.rectangle(canvas,
                       (lx - 3, ly - th - 3),
                       (lx + tw + 3, ly + 3),
                       COL_TEXT_BG, -1)
        cv2.putText(canvas, label, (lx, ly),
                    FONT, FONT_SCALE, col, FONT_THICKNESS, cv2.LINE_AA)

    # Pending marker
    if state.pending_cx is not None:
        sx, sy = view.img_to_screen(state.pending_cx, state.pending_cy)
        cv2.circle(canvas, (sx, sy), MARKER_RADIUS, COL_MARKER_PENDING,
                   2, cv2.LINE_AA)
        cv2.circle(canvas, (sx, sy), 3, COL_MARKER_PENDING, -1, cv2.LINE_AA)

        typing_label = f"Type number: {state.typing}_"
        (tw, th), _ = cv2.getTextSize(typing_label, FONT, FONT_SCALE, FONT_THICKNESS)
        lx = sx - tw // 2
        ly = sy - MARKER_RADIUS - 6
        cv2.rectangle(canvas,
                       (lx - 3, ly - th - 3),
                       (lx + tw + 3, ly + 3),
                       COL_TEXT_BG, -1)
        cv2.putText(canvas, typing_label, (lx, ly),
                    FONT, FONT_SCALE, COL_MARKER_PENDING,
                    FONT_THICKNESS, cv2.LINE_AA)

    # Bottom HUD
    hud_y = view.win_h - 10
    ix, iy = view.screen_to_img(mouse_sx, mouse_sy)
    hud = (f"Map {map_num}  |  "
           f"Img pos: ({int(ix)}, {int(iy)})  |  "
           f"Zoom: {view.scale:.2f}x  |  "
           f"Annotations: {len(state.annotations)}  |  "
           f"[Click=place] [Type=number] [Enter=confirm] "
           f"[RClick=remove] [S=save] [Q=quit]")
    (tw, th), _ = cv2.getTextSize(hud, FONT, 0.45, 1)
    cv2.rectangle(canvas,
                   (0, hud_y - th - 6),
                   (view.win_w, view.win_h),
                   (20, 20, 20), -1)
    cv2.putText(canvas, hud, (8, hud_y - 4),
                FONT, 0.45, COL_TEXT_FG, 1, cv2.LINE_AA)

    # Boundary parcel hint panel (right side)
    needed = BOUNDARY_PARCELS_BY_MAP.get(map_num, [])
    annotated_nums = {a["detected_number"] for a in state.annotations}
    panel_x = view.win_w - 220
    panel_y = 10

    cv2.rectangle(canvas,
                   (panel_x - 5, panel_y),
                   (view.win_w - 5, panel_y + 25 + len(needed) * 22),
                   (20, 20, 20), -1)
    cv2.putText(canvas, "Boundary parcels needed:",
                (panel_x, panel_y + 18),
                FONT, 0.5, COL_TEXT_FG, 1, cv2.LINE_AA)

    for i, pid in enumerate(needed):
        done   = pid in annotated_nums
        colour = (0, 200, 0) if done else (0, 180, 255)
        mark   = "[x]" if done else "[ ]"
        cv2.putText(canvas, f"{mark} {pid}",
                    (panel_x, panel_y + 40 + i * 22),
                    FONT, 0.5, colour, 1, cv2.LINE_AA)


# ---------------------------------------------------------------------------
# MAIN LOOP
# ---------------------------------------------------------------------------

def run(map_num: str):
    # Load image
    img_path = PREPROCESSED_DIR / f"map_{map_num}_clean.png"
    if not img_path.exists():
        print(f"ERROR: {img_path} not found. Run step1_preprocessing.py first.")
        sys.exit(1)

    print(f"\n  Loading map {map_num}...")
    img = read_image(img_path)
    img_h, img_w = img.shape[:2]
    print(f"  Image size: {img_w} × {img_h}")

    # Load existing annotations
    existing = load_existing_json(map_num)
    if existing:
        print(f"  Loaded {len(existing)} existing annotations")

    view  = ViewState(img_h, img_w, WINDOW_H, WINDOW_W)
    state = AnnotationState(map_num, existing)

    needed = BOUNDARY_PARCELS_BY_MAP.get(map_num, [])
    print(f"\n  Boundary parcels to annotate on map {map_num}:")
    for p in needed:
        print(f"    {p}")
    print(f"\n  Controls:")
    print(f"    Left click        = place marker")
    print(f"    Type digits       = enter parcel number")
    print(f"    Enter             = confirm annotation")
    print(f"    Right click       = remove nearest marker")
    print(f"    Scroll            = zoom in/out")
    print(f"    Middle drag       = pan")
    print(f"    S                 = save")
    print(f"    Q or Esc          = quit and save")
    print()

    win_name = f"Annotate Map {map_num} — boundary parcel labelling"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win_name, WINDOW_W, WINDOW_H)

    mouse_sx = WINDOW_W // 2
    mouse_sy = WINDOW_H // 2
    dragging = False
    needs_redraw = True

    def on_mouse(event, x, y, flags, param):
        nonlocal mouse_sx, mouse_sy, dragging, needs_redraw

        mouse_sx, mouse_sy = x, y
        needs_redraw = True

        if event == cv2.EVENT_LBUTTONDOWN:
            ix, iy = view.screen_to_img(x, y)
            if 0 <= ix < img_w and 0 <= iy < img_h:
                state.click(ix, iy)

        elif event == cv2.EVENT_RBUTTONDOWN:
            ix, iy = view.screen_to_img(x, y)
            # Convert click radius to image pixels
            radius_img = 50 / view.scale
            state.remove_nearest(ix, iy, radius_img)

        elif event == cv2.EVENT_MBUTTONDOWN:
            view.start_drag(x, y)
            dragging = True

        elif event == cv2.EVENT_MOUSEMOVE:
            if dragging:
                view.update_drag(x, y)

        elif event == cv2.EVENT_MBUTTONUP:
            view.end_drag()
            dragging = False

        elif event == cv2.EVENT_MOUSEWHEEL:
            factor = 1.15 if flags > 0 else 1 / 1.15
            view.zoom(factor, x, y)

    cv2.setMouseCallback(win_name, on_mouse)

    while True:
        if needs_redraw:
            canvas = view.render(img)
            draw_overlay(canvas, state, view, mouse_sx, mouse_sy, map_num)
            cv2.imshow(win_name, canvas)
            needs_redraw = False

        key = cv2.waitKey(20) & 0xFF

        if key == 255:  # no key
            continue

        needs_redraw = True

        # Quit
        if key in (ord('q'), ord('Q'), 27):  # Q or Esc
            save_json(map_num, state.annotations)
            break

        # Save
        elif key in (ord('s'), ord('S')):
            save_json(map_num, state.annotations)
            print(f"  Saved {len(state.annotations)} annotations")

        # Confirm annotation
        elif key == 13:  # Enter
            if state.confirm():
                a = state.annotations[-1]
                status = validate(a["detected_number"], map_num)
                print(f"  + Parcel {a['detected_number']} "
                      f"at ({a['cx']:.0f}, {a['cy']:.0f}) [{status}]")
            else:
                print("  (nothing to confirm — click a location first, "
                      "then type a number)")

        # Backspace
        elif key == 8:
            state.backspace()

        # Escape pending without confirming
        elif key == 27:
            state.cancel_pending()

        # Digit keys
        elif chr(key) in "0123456789":
            state.type_digit(chr(key))

        # Pan with arrow keys
        elif key == 81 or key == 2:   # left
            view.pan(30, 0)
        elif key == 83 or key == 3:   # right
            view.pan(-30, 0)
        elif key == 82 or key == 0:   # up
            view.pan(0, 30)
        elif key == 84 or key == 1:   # down
            view.pan(0, -30)

        # Zoom with + / -
        elif key in (ord('+'), ord('=')):
            view.zoom(1.2, WINDOW_W // 2, WINDOW_H // 2)
        elif key == ord('-'):
            view.zoom(1 / 1.2, WINDOW_W // 2, WINDOW_H // 2)

    cv2.destroyAllWindows()
    print(f"\n  Session complete. {len(state.annotations)} parcels annotated on map {map_num}.")
    annotated_nums = {a["detected_number"] for a in state.annotations}
    needed         = BOUNDARY_PARCELS_BY_MAP.get(map_num, [])
    missing        = [p for p in needed if p not in annotated_nums]
    if missing:
        print(f"  Still needed: {missing}")
    else:
        print(f"  All boundary parcels annotated for map {map_num}!")


# ---------------------------------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Manual parcel annotation tool for cadastral maps"
    )
    parser.add_argument(
        "--map", required=True,
        help="Map number to annotate (e.g. --map 45)"
    )
    args = parser.parse_args()
    run(args.map)
