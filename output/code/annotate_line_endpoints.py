"""
=============================================================================
LINE-ENDPOINT ANNOTATION TOOL — Fine-tune Step 4 Homographies
=============================================================================
Masters Thesis — AI-Based Panoramic Cadastral Image Reconstruction
Author  : Hussein Chalhoub

Purpose
--------
Step 4 places maps next to each other but the parcel boundary lines
sometimes don't connect perfectly across the join.  This tool lets you
click pairs of corresponding LINE ENDPOINTS — points where parcel boundary
lines exit map A's edge and where the SAME line continues on map B's edge.

These pairs become ground-truth correspondences used to refine the homography
beyond what parcel centroids alone could achieve.  Line endpoints are more
precise than centroids because:
  - A line endpoint is a specific pixel on a specific stroke
  - You can match it exactly between the two maps
  - 3-5 endpoint pairs give enough constraint for an affine or homography
    refinement that respects rotation, scale, AND position

How to use
-----------
  python annotate_line_endpoints.py --pair 45_46

This loads:
  - Map 45 (top/left, called "A")  AND  Map 46 (bottom/right, called "B")
  - The current registration produced by Step 4

It opens a window showing both maps placed according to the current Step 4
result.  You then:

  1. Click on a line endpoint on Map A near the shared boundary
     (where a parcel line exits map A's edge)
  2. Click on the SAME line's continuation on Map B
     (where the same parcel line enters map B's edge)
  3. Press Enter to confirm the pair
  4. Repeat 3-5 times for different lines along the boundary
  5. Press R to recompute and refresh the alignment
  6. Press S to save the refined homography to homographies.json
  7. Press Q to quit

Visual feedback
----------------
  Yellow dot     : 1st click (on Map A) - waiting for partner
  Green numbered : confirmed pair (cyan line connects the two endpoints)
  Red lines      : currently unconfirmed first click

Controls
---------
  Left click       : place an endpoint
  Enter            : confirm the pair (1st click + 2nd click)
  Backspace        : undo last action
  R                : recompute homography from confirmed pairs
  S                : save refined homography
  Q or Esc         : quit and save
  Scroll           : zoom
  Middle drag      : pan

Output
-------
  Updates output/homographies/H_<A>_<B>.npy with the refined homography
  Updates output/homographies/homographies.json
  Saves output/matches/map_<A>_<B>_registration_refined.png

=============================================================================
"""

import argparse
import json
import sys
import math
import cv2
import numpy as np
from pathlib import Path


# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------

PREPROCESSED_DIR = Path("output/preprocessed")
HOMOGRAPHY_DIR   = Path("output/homographies")
MATCHES_DIR      = Path("output/matches")

WINDOW_W = 1400
WINDOW_H = 900

MARKER_RADIUS    = 14
LINE_THICKNESS   = 2
FONT             = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE       = 0.6
FONT_THICKNESS   = 2

COL_PENDING      = (0, 220, 255)   # yellow — waiting for second click
COL_CONFIRMED_A  = (0, 200, 0)     # green  — confirmed point on A
COL_CONFIRMED_B  = (0, 200, 0)     # green  — confirmed point on B
COL_PAIR_LINK    = (255, 200, 0)   # cyan   — line connecting the pair
COL_TEXT_BG      = (30, 30, 30)
COL_TEXT_FG      = (255, 255, 255)
COL_BORDER_A     = (255, 80, 0)    # blue
COL_BORDER_B     = (0, 140, 255)   # orange
COL_CROSSHAIR    = (170, 170, 170)


# ---------------------------------------------------------------------------
# UTILITIES
# ---------------------------------------------------------------------------

def read_image(path: Path, grayscale: bool = True) -> np.ndarray:
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


def load_pair_metadata(map_a: str, map_b: str) -> dict:
    """Load existing homography metadata produced by Step 4."""
    json_path = HOMOGRAPHY_DIR / "homographies.json"
    if not json_path.exists():
        raise FileNotFoundError(
            f"{json_path} not found. Run step4_matching.py first."
        )
    with open(json_path, "r", encoding="utf-8") as f:
        all_data = json.load(f)
    key = f"{map_a}_{map_b}"
    if key not in all_data:
        raise KeyError(
            f"Pair {key} not in homographies.json. Run step4_matching.py first."
        )
    return all_data[key]


def save_refined_homography(map_a: str, map_b: str,
                              H: np.ndarray,
                              n_pairs: int):
    """Update homographies.json with refined homography."""
    np.save(str(HOMOGRAPHY_DIR / f"H_{map_a}_{map_b}.npy"), H)
    json_path = HOMOGRAPHY_DIR / "homographies.json"
    with open(json_path, "r", encoding="utf-8") as f:
        all_data = json.load(f)
    key = f"{map_a}_{map_b}"
    all_data[key]["H"] = H.tolist()
    all_data[key]["refinement"] = {
        "method":  "manual_line_endpoints",
        "n_pairs": int(n_pairs),
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_data, f, indent=2, ensure_ascii=False)
    print(f"  Saved refined H_{map_a}_{map_b}.npy + homographies.json")


# ---------------------------------------------------------------------------
# VIEW STATE (pan + zoom)
# ---------------------------------------------------------------------------

class ViewState:
    def __init__(self, canvas_h: int, canvas_w: int,
                 win_h: int, win_w: int):
        self.canvas_h = canvas_h
        self.canvas_w = canvas_w
        self.win_h    = win_h
        self.win_w    = win_w
        self.scale    = min(win_w / canvas_w, win_h / canvas_h)
        self.pan_x    = (win_w - canvas_w * self.scale) / 2
        self.pan_y    = (win_h - canvas_h * self.scale) / 2
        self._drag_start = None
        self._drag_pan   = None

    def canvas_to_screen(self, cx, cy):
        return int(cx * self.scale + self.pan_x), int(cy * self.scale + self.pan_y)

    def screen_to_canvas(self, sx, sy):
        return ((sx - self.pan_x) / self.scale,
                (sy - self.pan_y) / self.scale)

    def zoom(self, factor, cx_screen, cy_screen):
        cx, cy = self.screen_to_canvas(cx_screen, cy_screen)
        self.scale = max(0.05, min(20.0, self.scale * factor))
        self.pan_x = cx_screen - cx * self.scale
        self.pan_y = cy_screen - cy * self.scale

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

    def render(self, canvas: np.ndarray) -> np.ndarray:
        """Crop the canvas to what's visible and fit the window."""
        x0, y0 = self.screen_to_canvas(0, 0)
        x1, y1 = self.screen_to_canvas(self.win_w, self.win_h)

        x0c = int(max(0, x0)); y0c = int(max(0, y0))
        x1c = int(min(self.canvas_w, x1)); y1c = int(min(self.canvas_h, y1))

        if x1c <= x0c or y1c <= y0c:
            return np.zeros((self.win_h, self.win_w, 3), dtype=np.uint8)

        crop = canvas[y0c:y1c, x0c:x1c]
        sw = int((x1c - x0c) * self.scale)
        sh = int((y1c - y0c) * self.scale)
        if sw < 1 or sh < 1:
            return np.zeros((self.win_h, self.win_w, 3), dtype=np.uint8)

        interp = cv2.INTER_AREA if self.scale < 1 else cv2.INTER_LINEAR
        scaled = cv2.resize(crop, (sw, sh), interpolation=interp)

        out = np.zeros((self.win_h, self.win_w, 3), dtype=np.uint8)
        dst_x = int(self.pan_x + x0c * self.scale)
        dst_y = int(self.pan_y + y0c * self.scale)

        src_x0 = max(0, -dst_x); src_y0 = max(0, -dst_y)
        dst_x  = max(0, dst_x);  dst_y  = max(0, dst_y)
        h_copy = min(sh - src_y0, self.win_h - dst_y)
        w_copy = min(sw - src_x0, self.win_w - dst_x)

        if h_copy > 0 and w_copy > 0:
            out[dst_y:dst_y + h_copy, dst_x:dst_x + w_copy] = \
                scaled[src_y0:src_y0 + h_copy, src_x0:src_x0 + w_copy]
        return out


# ---------------------------------------------------------------------------
# REGISTRATION CANVAS BUILDER
# ---------------------------------------------------------------------------

def build_registration_canvas(gray_a: np.ndarray,
                                 gray_b: np.ndarray,
                                 H: np.ndarray,
                                 rotation_b_deg: int = 0,
                                 edge_a: str = "right",
                                 edge_b: str = "left"
                                 ) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """
    Build a SIDE-BY-SIDE canvas where map A and map B sit in their own
    separate panels with a visible gap between them.  No overlap, no blending,
    no homography distortion - just two clean panels you can click on.

    Map B is rotated by rotation_b_deg first (so it appears in the correct
    orientation), then placed next to map A according to the edge pair:
      edge_a=right, edge_b=left   -> A on left,  B on right (horizontal)
      edge_a=left,  edge_b=right  -> A on right, B on left  (horizontal)
      edge_a=bottom, edge_b=top   -> A on top,   B on bottom (vertical)
      edge_a=top,   edge_b=bottom -> A on bottom, B on top  (vertical)

    Returns:
      canvas_bgr  : composed image
      M_A         : 3x3 transform mapping ORIGINAL A pixels -> canvas pixels
      M_B         : 3x3 transform mapping ORIGINAL B pixels -> canvas pixels
                    (includes the rotation_b_deg!)
      meta        : dict with shape_a, shape_b, canvas size, etc.

    H is no longer used here - it was misleading.  We use rotation+edge
    metadata directly.
    """
    H_A, W_A = gray_a.shape
    H_B_orig, W_B_orig = gray_b.shape

    # Rotate B image and compute the 3x3 rotation matrix mapping original B
    # pixel coords -> rotated B pixel coords.
    if rotation_b_deg == 0:
        gray_b_rot = gray_b.copy()
        H_B_rot, W_B_rot = H_B_orig, W_B_orig
        R_B = np.eye(3, dtype=np.float64)
    elif rotation_b_deg == 90:
        gray_b_rot = cv2.rotate(gray_b, cv2.ROTATE_90_COUNTERCLOCKWISE)
        H_B_rot, W_B_rot = W_B_orig, H_B_orig
        R_B = np.array([[0, 1, 0], [-1, 0, W_B_orig - 1], [0, 0, 1]], dtype=np.float64)
    elif rotation_b_deg == 180:
        gray_b_rot = cv2.rotate(gray_b, cv2.ROTATE_180)
        H_B_rot, W_B_rot = H_B_orig, W_B_orig
        R_B = np.array([[-1, 0, W_B_orig - 1], [0, -1, H_B_orig - 1], [0, 0, 1]], dtype=np.float64)
    elif rotation_b_deg == 270:
        gray_b_rot = cv2.rotate(gray_b, cv2.ROTATE_90_CLOCKWISE)
        H_B_rot, W_B_rot = W_B_orig, H_B_orig
        R_B = np.array([[0, -1, H_B_orig - 1], [1, 0, 0], [0, 0, 1]], dtype=np.float64)
    else:
        gray_b_rot = gray_b.copy()
        H_B_rot, W_B_rot = H_B_orig, W_B_orig
        R_B = np.eye(3, dtype=np.float64)

    # Decide layout based on edges
    GAP = 100   # pixels of white space between panels

    if edge_a == "right" and edge_b == "left":
        # A on left, B on right
        canvas_w = W_A + GAP + W_B_rot
        canvas_h = max(H_A, H_B_rot)
        a_offset = (0, 0)
        b_offset = (W_A + GAP, 0)
    elif edge_a == "left" and edge_b == "right":
        # A on right, B on left
        canvas_w = W_B_rot + GAP + W_A
        canvas_h = max(H_A, H_B_rot)
        a_offset = (W_B_rot + GAP, 0)
        b_offset = (0, 0)
    elif edge_a == "bottom" and edge_b == "top":
        # A on top, B on bottom
        canvas_w = max(W_A, W_B_rot)
        canvas_h = H_A + GAP + H_B_rot
        a_offset = (0, 0)
        b_offset = (0, H_A + GAP)
    elif edge_a == "top" and edge_b == "bottom":
        # A on bottom, B on top
        canvas_w = max(W_A, W_B_rot)
        canvas_h = H_B_rot + GAP + H_A
        a_offset = (0, H_B_rot + GAP)
        b_offset = (0, 0)
    else:
        # Default: side by side horizontally
        canvas_w = W_A + GAP + W_B_rot
        canvas_h = max(H_A, H_B_rot)
        a_offset = (0, 0)
        b_offset = (W_A + GAP, 0)

    # Cap canvas size for memory
    MAX_DIM = 9000
    scale_factor = 1.0
    if max(canvas_w, canvas_h) > MAX_DIM:
        scale_factor = MAX_DIM / max(canvas_w, canvas_h)
        canvas_w = int(canvas_w * scale_factor)
        canvas_h = int(canvas_h * scale_factor)
        a_offset = (int(a_offset[0] * scale_factor),
                    int(a_offset[1] * scale_factor))
        b_offset = (int(b_offset[0] * scale_factor),
                    int(b_offset[1] * scale_factor))

    S = np.array([[scale_factor, 0, 0],
                  [0, scale_factor, 0],
                  [0, 0, 1]], dtype=np.float64)
    T_A = np.array([[1, 0, a_offset[0]],
                    [0, 1, a_offset[1]],
                    [0, 0, 1]], dtype=np.float64)
    T_B = np.array([[1, 0, b_offset[0]],
                    [0, 1, b_offset[1]],
                    [0, 0, 1]], dtype=np.float64)
    # M_A maps ORIGINAL A pixels -> canvas pixels (translation + scale)
    M_A = T_A @ S
    # M_B maps ORIGINAL B pixels -> canvas pixels.  Original B is rotated
    # by rotation_b_deg first (R_B), then translated and scaled.
    M_B = T_B @ S @ R_B

    # Build canvas (white background)
    canvas_gray = np.ones((canvas_h, canvas_w), dtype=np.uint8) * 255

    # Place A
    if scale_factor != 1.0:
        a_resized = cv2.resize(
            gray_a,
            (int(W_A * scale_factor), int(H_A * scale_factor)),
            interpolation=cv2.INTER_AREA
        )
    else:
        a_resized = gray_a
    ah, aw = a_resized.shape
    ay, ax = a_offset[1], a_offset[0]
    canvas_gray[ay:ay + ah, ax:ax + aw] = a_resized

    # Place B (rotated)
    if scale_factor != 1.0:
        b_resized = cv2.resize(
            gray_b_rot,
            (int(W_B_rot * scale_factor), int(H_B_rot * scale_factor)),
            interpolation=cv2.INTER_AREA
        )
    else:
        b_resized = gray_b_rot
    bh, bw = b_resized.shape
    by, bx = b_offset[1], b_offset[0]
    canvas_gray[by:by + bh, bx:bx + bw] = b_resized

    canvas = cv2.cvtColor(canvas_gray, cv2.COLOR_GRAY2BGR)

    # Draw clear coloured borders around each map
    border_thickness = max(3, int(8 * scale_factor))
    cv2.rectangle(canvas, (ax, ay), (ax + aw - 1, ay + ah - 1),
                   COL_BORDER_A, border_thickness, cv2.LINE_AA)
    cv2.rectangle(canvas, (bx, by), (bx + bw - 1, by + bh - 1),
                   COL_BORDER_B, border_thickness, cv2.LINE_AA)

    # Big "A" and "B" labels for clarity
    label_scale = max(3.0, 6.0 * scale_factor)
    cv2.putText(canvas, "MAP A",
                (ax + 30, ay + int(80 * scale_factor) + 30),
                cv2.FONT_HERSHEY_SIMPLEX, label_scale,
                (255, 255, 255), int(15 * scale_factor) + 6, cv2.LINE_AA)
    cv2.putText(canvas, "MAP A",
                (ax + 30, ay + int(80 * scale_factor) + 30),
                cv2.FONT_HERSHEY_SIMPLEX, label_scale,
                COL_BORDER_A, int(8 * scale_factor) + 2, cv2.LINE_AA)
    cv2.putText(canvas, "MAP B",
                (bx + 30, by + int(80 * scale_factor) + 30),
                cv2.FONT_HERSHEY_SIMPLEX, label_scale,
                (255, 255, 255), int(15 * scale_factor) + 6, cv2.LINE_AA)
    cv2.putText(canvas, "MAP B",
                (bx + 30, by + int(80 * scale_factor) + 30),
                cv2.FONT_HERSHEY_SIMPLEX, label_scale,
                COL_BORDER_B, int(8 * scale_factor) + 2, cv2.LINE_AA)

    meta = {
        "shape_a":         (H_A, W_A),
        "shape_b":         (H_B_orig, W_B_orig),  # ORIGINAL shape (pre-rotation)
        "shape_b_rot":     (H_B_rot, W_B_rot),
        "rotation_b_deg":  rotation_b_deg,
        "canvas_w":        canvas_w,
        "canvas_h":        canvas_h,
        "scale_factor":    scale_factor,
        "a_offset":        a_offset,
        "b_offset":        b_offset,
        "edge_a":          edge_a,
        "edge_b":          edge_b,
    }
    return canvas, M_A, M_B, meta


def map_canvas_to_image(canvas_xy: tuple[float, float],
                          M: np.ndarray) -> tuple[float, float]:
    """Convert a canvas pixel back to original-image pixel using inverse M."""
    M_inv = np.linalg.inv(M)
    p = np.array([canvas_xy[0], canvas_xy[1], 1.0], dtype=np.float64)
    out = M_inv @ p
    out /= out[2]
    return float(out[0]), float(out[1])


def map_image_to_canvas(img_xy: tuple[float, float],
                          M: np.ndarray) -> tuple[float, float]:
    p = np.array([img_xy[0], img_xy[1], 1.0], dtype=np.float64)
    out = M @ p
    out /= out[2]
    return float(out[0]), float(out[1])


# ---------------------------------------------------------------------------
# ANNOTATION STATE
# ---------------------------------------------------------------------------

class AnnotationState:
    """
    Holds line-endpoint pairs in ORIGINAL image coordinates so they survive
    homography refinement (we re-project them onto the new canvas after each R).
    """
    def __init__(self):
        self.pairs   = []  # list of dicts: {a_xy: (x,y), b_xy: (x,y)}
        self.pending = None  # 1st click as image-coords on A or B
        self.pending_side = None  # 'A' or 'B'

    def add_first_click(self, side: str, img_xy: tuple[float, float]):
        self.pending = img_xy
        self.pending_side = side

    def confirm(self, side: str, img_xy: tuple[float, float]) -> bool:
        if self.pending is None or self.pending_side is None:
            return False
        if side == self.pending_side:
            return False  # second click must be on the OTHER side
        if self.pending_side == 'A':
            self.pairs.append({"a_xy": self.pending, "b_xy": img_xy})
        else:
            self.pairs.append({"a_xy": img_xy, "b_xy": self.pending})
        self.pending = None
        self.pending_side = None
        return True

    def undo(self):
        if self.pending is not None:
            self.pending = None
            self.pending_side = None
        elif self.pairs:
            removed = self.pairs.pop()
            print(f"  Removed pair: A={removed['a_xy']}  B={removed['b_xy']}")


# ---------------------------------------------------------------------------
# HOMOGRAPHY REFINEMENT
# ---------------------------------------------------------------------------

def refine_homography(pairs: list[dict],
                       H_current: np.ndarray) -> tuple[np.ndarray, str]:
    """
    Compute a refined homography from line-endpoint pairs.

    Pairs contain ORIGINAL image-space coordinates for both A and B (the
    canvas-to-image mapping in M_A and M_B already accounts for B's rotation).

    The new homography directly maps ORIGINAL B pixel coords to A's frame
    using the user's manual correspondences.

      1 pair  : pure translation that takes B's centroid to A's centroid
                (preserves Step 4's rotation/edge information from H_current)
      2 pairs : pure translation (median of per-point shifts)
      3 pairs : affine RANSAC (translation + rotation + uniform scale)
      4+      : full homography with RANSAC

    For 1-2 pairs we cannot estimate rotation/scale reliably, so we keep
    the rotation+edge structure from H_current and only correct the position.

    Returns (H_refined, status_message).
    """
    n = len(pairs)
    if n == 0:
        return H_current, "no pairs"

    pts_a = np.float32([list(p["a_xy"]) for p in pairs])
    pts_b = np.float32([list(p["b_xy"]) for p in pairs])

    if n <= 2:
        # Translation-only: use H_current to predict B's points in A frame,
        # measure the offset, apply as correction.
        diffs = []
        for i in range(n):
            pb_h = np.array([pts_b[i, 0], pts_b[i, 1], 1.0], dtype=np.float64)
            out = H_current @ pb_h
            out /= out[2]
            diffs.append([pts_a[i, 0] - out[0], pts_a[i, 1] - out[1]])
        diffs = np.array(diffs)
        dx = float(np.median(diffs[:, 0]))
        dy = float(np.median(diffs[:, 1]))
        T = np.array([[1, 0, dx], [0, 1, dy], [0, 0, 1]], dtype=np.float64)
        H_refined = T @ H_current
        return H_refined, f"translation (n={n}): dx={dx:.0f}px dy={dy:.0f}px"

    if n == 3:
        # Affine: rotation + uniform scale + translation
        M, mask = cv2.estimateAffinePartial2D(
            pts_b.reshape(-1, 1, 2),
            pts_a.reshape(-1, 1, 2),
            method=cv2.RANSAC,
            ransacReprojThreshold=30.0
        )
        if M is None:
            return H_current, f"affine failed (n=3)"
        H_refined = np.vstack([M, [0, 0, 1]]).astype(np.float64)
        return H_refined, f"affine (n=3)"

    # 4+ pairs: full homography with RANSAC
    H_new, mask = cv2.findHomography(
        pts_b.reshape(-1, 1, 2),
        pts_a.reshape(-1, 1, 2),
        method=cv2.RANSAC,
        ransacReprojThreshold=20.0
    )
    if H_new is None:
        return H_current, f"homography failed (n={n})"
    inliers = int(mask.sum()) if mask is not None else n
    return H_new, f"homography (n={n}, {inliers} inliers)"


# ---------------------------------------------------------------------------
# DRAWING
# ---------------------------------------------------------------------------

def draw_overlay(canvas: np.ndarray,
                  state: AnnotationState,
                  M_A: np.ndarray,
                  M_B: np.ndarray,
                  view: ViewState,
                  mouse_sx: int,
                  mouse_sy: int,
                  pair_label: str,
                  status_msg: str):
    """Draw all annotations on top of the registration canvas."""
    # Crosshair
    cv2.line(canvas, (mouse_sx, 0), (mouse_sx, view.win_h),
             COL_CROSSHAIR, 1, cv2.LINE_AA)
    cv2.line(canvas, (0, mouse_sy), (view.win_w, mouse_sy),
             COL_CROSSHAIR, 1, cv2.LINE_AA)

    # Confirmed pairs
    for i, p in enumerate(state.pairs):
        a_canvas = map_image_to_canvas(p["a_xy"], M_A)
        b_canvas = map_image_to_canvas(p["b_xy"], M_B)
        a_screen = view.canvas_to_screen(*a_canvas)
        b_screen = view.canvas_to_screen(*b_canvas)

        # Connecting line
        cv2.line(canvas, a_screen, b_screen, COL_PAIR_LINK,
                 LINE_THICKNESS, cv2.LINE_AA)
        # Endpoints
        cv2.circle(canvas, a_screen, MARKER_RADIUS, COL_CONFIRMED_A,
                   2, cv2.LINE_AA)
        cv2.circle(canvas, b_screen, MARKER_RADIUS, COL_CONFIRMED_B,
                   2, cv2.LINE_AA)
        # Label
        label = f"{i + 1}"
        cv2.putText(canvas, label,
                    (a_screen[0] + MARKER_RADIUS + 4, a_screen[1] + 5),
                    FONT, FONT_SCALE, COL_CONFIRMED_A, FONT_THICKNESS, cv2.LINE_AA)
        cv2.putText(canvas, label,
                    (b_screen[0] + MARKER_RADIUS + 4, b_screen[1] + 5),
                    FONT, FONT_SCALE, COL_CONFIRMED_B, FONT_THICKNESS, cv2.LINE_AA)

    # Pending point
    if state.pending is not None:
        M = M_A if state.pending_side == 'A' else M_B
        canvas_xy = map_image_to_canvas(state.pending, M)
        screen_xy = view.canvas_to_screen(*canvas_xy)
        cv2.circle(canvas, screen_xy, MARKER_RADIUS, COL_PENDING,
                   2, cv2.LINE_AA)
        cv2.circle(canvas, screen_xy, 4, COL_PENDING, -1, cv2.LINE_AA)
        # Hint
        cv2.putText(canvas,
                    f"Now click matching point on Map {'B' if state.pending_side == 'A' else 'A'}",
                    (screen_xy[0] + 20, screen_xy[1]),
                    FONT, 0.6, COL_PENDING, FONT_THICKNESS, cv2.LINE_AA)

    # Bottom HUD
    hud_y = view.win_h - 12
    pairs_count = len(state.pairs)
    hud = (f"Pair {pair_label}  |  Confirmed pairs: {pairs_count}  |  "
           f"Pending: {'YES on ' + state.pending_side if state.pending else 'NO'}  |  "
           f"[Click=mark | Enter=confirm pair via 2 clicks | R=recompute | "
           f"S=save | U=undo | Q=quit]")
    (tw, th), _ = cv2.getTextSize(hud, FONT, 0.45, 1)
    cv2.rectangle(canvas, (0, hud_y - th - 6),
                   (view.win_w, view.win_h), (20, 20, 20), -1)
    cv2.putText(canvas, hud, (8, hud_y - 4),
                FONT, 0.45, COL_TEXT_FG, 1, cv2.LINE_AA)

    # Top status banner
    status_text = (f"Map A (BLUE border)  +  Map B (ORANGE border)  |  {status_msg}")
    cv2.rectangle(canvas, (0, 0), (view.win_w, 32), (20, 20, 20), -1)
    cv2.putText(canvas, status_text, (10, 22),
                FONT, 0.55, COL_TEXT_FG, 1, cv2.LINE_AA)


# ---------------------------------------------------------------------------
# DETERMINE WHICH MAP A POINT IS ON
# ---------------------------------------------------------------------------

def hit_test(canvas_xy: tuple[float, float],
              M_A: np.ndarray, M_B: np.ndarray,
              shape_a: tuple[int, int],
              shape_b: tuple[int, int]) -> str | None:
    """
    Determine if a canvas point lies inside map A, map B, both, or neither.
    Returns 'A', 'B', or None (or 'overlap' if both - we use whichever side
    has an active pending click).
    """
    # Check A
    M_A_inv = np.linalg.inv(M_A)
    p = np.array([canvas_xy[0], canvas_xy[1], 1.0])
    pa = M_A_inv @ p
    pa /= pa[2]
    in_a = (0 <= pa[0] < shape_a[1]) and (0 <= pa[1] < shape_a[0])

    # Check B
    M_B_inv = np.linalg.inv(M_B)
    pb = M_B_inv @ p
    pb /= pb[2]
    in_b = (0 <= pb[0] < shape_b[1]) and (0 <= pb[1] < shape_b[0])

    if in_a and not in_b:
        return 'A'
    if in_b and not in_a:
        return 'B'
    if in_a and in_b:
        return 'overlap'
    return None


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def run(map_a: str, map_b: str,
        forced_edges: tuple | None = None,
        forced_rotation: int | None = None):
    print(f"\n  Loading pair {map_a}_{map_b}...")
    img_a = read_image(PREPROCESSED_DIR / f"map_{map_a}_clean.png", grayscale=True)
    img_b = read_image(PREPROCESSED_DIR / f"map_{map_b}_clean.png", grayscale=True)
    print(f"  Map {map_a}: {img_a.shape}  |  Map {map_b}: {img_b.shape}")

    meta = load_pair_metadata(map_a, map_b)
    H = np.array(meta["H"], dtype=np.float64)
    rotation_b = int(meta.get("rotation_b_deg", 0))
    edge_a     = meta.get("edge_a", "right")
    edge_b     = meta.get("edge_b", "left")

    # Apply user overrides if provided
    if forced_edges is not None:
        edge_a, edge_b = forced_edges
        print(f"  USER OVERRIDE: layout = {edge_a} -> {edge_b}")
        # When layout changes, the existing H is no longer correct.
        # Reset H to identity for B coords -> A coords.  The user clicks
        # will define the homography from scratch.
        H = np.eye(3, dtype=np.float64)
    if forced_rotation is not None:
        rotation_b = forced_rotation
        print(f"  USER OVERRIDE: rotation = {rotation_b}deg")
        H = np.eye(3, dtype=np.float64)

    print(f"  Using rotation={rotation_b}deg, edges={edge_a}->{edge_b}")

    # Build initial side-by-side canvas (no overlap, no homography distortion)
    canvas, M_A, M_B, canvas_meta = build_registration_canvas(
        img_a, img_b, H, rotation_b, edge_a, edge_b
    )
    print(f"  Canvas size: {canvas_meta['canvas_w']} x {canvas_meta['canvas_h']}")

    state = AnnotationState()
    view  = ViewState(canvas_meta["canvas_h"], canvas_meta["canvas_w"],
                      WINDOW_H, WINDOW_W)

    win_name = f"Line endpoint annotation: {map_a} <-> {map_b}"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win_name, WINDOW_W, WINDOW_H)

    print(f"\n  Click line endings on Map A (boundary edge), then click")
    print(f"  the SAME line continuation on Map B, then press Enter to confirm.")
    print(f"  After 3+ pairs, press R to recompute, then S to save.")
    print(f"\n  Controls: Click=mark, Enter=confirm pair, R=recompute,")
    print(f"            S=save, U=undo, Q=quit, Scroll=zoom, Middle drag=pan\n")

    mouse_sx = WINDOW_W // 2
    mouse_sy = WINDOW_H // 2
    dragging = False
    needs_redraw = True
    status_msg = "Click a line ending on Map A, then on Map B"
    last_two_clicks = []  # list of (img_xy, side) — temporarily stored for Enter

    def on_mouse(event, x, y, flags, param):
        nonlocal mouse_sx, mouse_sy, dragging, needs_redraw, last_two_clicks, status_msg
        mouse_sx, mouse_sy = x, y
        needs_redraw = True

        if event == cv2.EVENT_LBUTTONDOWN:
            cx, cy = view.screen_to_canvas(x, y)
            side = hit_test(
                (cx, cy), M_A, M_B,
                canvas_meta["shape_a"], canvas_meta["shape_b"]
            )
            if side is None:
                status_msg = "Click inside one of the maps"
                return
            if side == 'overlap':
                # Use whichever side has not been clicked recently
                if last_two_clicks and last_two_clicks[-1][1] == 'A':
                    side = 'B'
                else:
                    side = 'A'
                status_msg = f"Overlap zone — assigned to side {side}"
            # Convert canvas xy to original image xy
            M = M_A if side == 'A' else M_B
            img_xy = map_canvas_to_image((cx, cy), M)

            # If we already had a pending on the other side, this completes a pair
            if state.pending is not None and state.pending_side != side:
                state.confirm(side, img_xy)
                last_two_clicks = []
                status_msg = f"Pair {len(state.pairs)} confirmed!"
                print(f"  + Pair {len(state.pairs)}: A={state.pairs[-1]['a_xy']}, "
                      f"B={state.pairs[-1]['b_xy']}")
            else:
                state.add_first_click(side, img_xy)
                last_two_clicks = [(img_xy, side)]
                status_msg = f"1st click on Map {side} - now click matching point on Map {'B' if side == 'A' else 'A'}"

        elif event == cv2.EVENT_MBUTTONDOWN:
            view.start_drag(x, y); dragging = True
        elif event == cv2.EVENT_MOUSEMOVE:
            if dragging: view.update_drag(x, y)
        elif event == cv2.EVENT_MBUTTONUP:
            view.end_drag(); dragging = False
        elif event == cv2.EVENT_MOUSEWHEEL:
            factor = 1.15 if flags > 0 else 1 / 1.15
            view.zoom(factor, x, y)

    cv2.setMouseCallback(win_name, on_mouse)

    while True:
        if needs_redraw:
            display = view.render(canvas)
            draw_overlay(display, state, M_A, M_B, view,
                          mouse_sx, mouse_sy,
                          f"{map_a}_{map_b}", status_msg)
            cv2.imshow(win_name, display)
            needs_redraw = False

        key = cv2.waitKey(20) & 0xFF
        if key == 255:
            continue
        needs_redraw = True

        if key in (ord('q'), ord('Q'), 27):
            # Save before quitting if there are pairs
            if state.pairs:
                H_new, msg = refine_homography(state.pairs, H)
                save_refined_homography(map_a, map_b, H_new, len(state.pairs))
                # Save final preview
                final_canvas, _, _, _ = build_registration_canvas(img_a, img_b, H_new, rotation_b, edge_a, edge_b)
                save_image(MATCHES_DIR / f"map_{map_a}_{map_b}_registration_refined.png",
                            final_canvas)
                print(f"  Refined: {msg}")
            break

        elif key in (ord('s'), ord('S')):
            if not state.pairs:
                status_msg = "Need at least 1 pair to save"
                continue
            H_new, msg = refine_homography(state.pairs, H)
            save_refined_homography(map_a, map_b, H_new, len(state.pairs))
            final_canvas, _, _, _ = build_registration_canvas(img_a, img_b, H_new, rotation_b, edge_a, edge_b)
            save_image(MATCHES_DIR / f"map_{map_a}_{map_b}_registration_refined.png",
                        final_canvas)
            status_msg = f"SAVED — {msg}"
            print(f"  Saved: {msg}")

        elif key in (ord('r'), ord('R')):
            if not state.pairs:
                status_msg = "Need at least 1 pair to recompute"
                continue
            H_new, msg = refine_homography(state.pairs, H)
            H = H_new
            canvas, M_A, M_B, canvas_meta = build_registration_canvas(
                img_a, img_b, H, rotation_b, edge_a, edge_b
            )
            view = ViewState(canvas_meta["canvas_h"], canvas_meta["canvas_w"],
                              WINDOW_H, WINDOW_W)
            status_msg = f"Recomputed: {msg}"
            print(f"  Recomputed: {msg}")

        elif key in (ord('u'), ord('U'), 8):  # U or Backspace
            state.undo()
            status_msg = "Undid last action"

        elif key == 13:  # Enter
            # Enter is used to manually confirm without needing the second click
            # workflow.  But our auto-pair-on-second-click usually handles this.
            status_msg = f"Confirmed pairs so far: {len(state.pairs)}"

    cv2.destroyAllWindows()
    print(f"\n  Session complete. {len(state.pairs)} pairs annotated.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Manual line-endpoint annotation tool for cadastral map registration"
    )
    parser.add_argument(
        "--pair", required=True,
        help="Pair to annotate, e.g. --pair 45_46"
    )
    parser.add_argument(
        "--layout", default=None,
        choices=["A_left_B_right", "A_right_B_left",
                 "A_top_B_bottom", "A_bottom_B_top"],
        help="Force layout direction. Overrides Step 4's edge choice. "
             "Examples: A_top_B_bottom means Map A on TOP, Map B BELOW it. "
             "Default: use Step 4's saved layout."
    )
    parser.add_argument(
        "--rotation", default=None, type=int,
        choices=[0, 90, 180, 270],
        help="Force rotation of Map B in degrees. Overrides Step 4's choice."
    )
    args = parser.parse_args()

    if "_" not in args.pair:
        print("ERROR: --pair must be in format A_B, e.g. 45_46")
        sys.exit(1)

    # Translate user-friendly layout to internal edge_a/edge_b convention
    layout_to_edges = {
        "A_left_B_right":  ("right",  "left"),
        "A_right_B_left":  ("left",   "right"),
        "A_top_B_bottom":  ("bottom", "top"),
        "A_bottom_B_top":  ("top",    "bottom"),
    }
    forced_edges = layout_to_edges.get(args.layout) if args.layout else None

    map_a, map_b = args.pair.split("_")
    run(map_a, map_b, forced_edges=forced_edges, forced_rotation=args.rotation)