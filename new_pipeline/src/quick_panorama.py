"""
=============================================================================
QUICK PANORAMA PREVIEW — blended, no visible seams
=============================================================================
Builds a panoramic view of 5 connected cadastral maps (45-49) using
orientation results from edge_orientation_finder.py.

How it works
-------------
1. Load best orientation (rotation + offset) for each pair from saved JSON.
2. Rotate every map to its correct orientation relative to map 45.
3. Trim white scan borders from each contact edge (_trim_contact_edge).
4. Compute canvas positions by chaining edge connections, with OVR-pixel
   overlap at every seam.
5. Build a per-map alpha mask:
     - interior pixels  → alpha = 1.0
     - contact edge OVR rows/cols → alpha ramps linearly 0 → 1 (incoming)
       or 1 → 0 (outgoing) so complementary alphas always sum to 1 in the
       overlap zone — same linear blend used in bestwithoutspacing.png.
6. Accumulate weighted pixels: acc += pixel * alpha, wgt += alpha.
7. Final canvas = acc / wgt  (normalised weighted average).
=============================================================================
"""

import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from edge_orientation_finder import (
    FONT, PREPROCESSED_DIR, RESULTS_DIR,
    auto_crop, read_image, rotate_image, _trim_contact_edge,
)

SCALE    = 0.10    # display scale — 10 % of original size
OVR      = 35      # overlap zone width in display pixels (= 350 px real)
PAD      = 180     # white border around full panorama
OUT_PATH = RESULTS_DIR / "panorama_45_47_48.png"


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def load_best(pair_key: str) -> dict:
    path = RESULTS_DIR / f"pair_{pair_key}_scores.json"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing results for pair {pair_key}. "
            f"Run:  python edge_orientation_finder.py --pair {pair_key} --visualise"
        )
    with open(path, encoding="utf-8") as f:
        return json.load(f)["best"]


def load_map(map_num: str, rot: float, contact_edge: str) -> np.ndarray:
    """Load, auto-crop, rotate, re-crop (removes warpAffine white fill), trim contact edge, then scale."""
    img = auto_crop(read_image(PREPROCESSED_DIR / f"map_{map_num}_clean.png"))
    img = rotate_image(img, rot)
    img = auto_crop(img)   # remove white corners added by non-90° rotation
    img = _trim_contact_edge(img, contact_edge)
    h, w = img.shape
    return cv2.resize(img, (int(w * SCALE), int(h * SCALE)),
                      interpolation=cv2.INTER_AREA)


def make_alpha(h: int, w: int, contact_edges: list) -> np.ndarray:
    """
    Per-map alpha mask.
    At each contact edge, alpha ramps from 0 (at edge) to 1 (OVR inward).
    Outside contact edges, alpha stays 1.
    """
    alpha = np.ones((h, w), dtype=np.float32)
    ramp  = np.linspace(0.0, 1.0, OVR, dtype=np.float32)

    for edge in contact_edges:
        if edge == 'top':
            # incoming: 0 at top edge, grows to 1 going down
            alpha[:OVR, :]   *= ramp[:, np.newaxis]
        elif edge == 'bottom':
            # outgoing: 1 going up, fades to 0 at bottom edge
            alpha[-OVR:, :]  *= ramp[::-1, np.newaxis]
        elif edge == 'left':
            # incoming: 0 at left edge, grows to 1 going right
            alpha[:, :OVR]   *= ramp[np.newaxis, :]
        elif edge == 'right':
            # outgoing: 1 going left, fades to 0 at right edge
            alpha[:, -OVR:]  *= ramp[::-1][np.newaxis, :]

    return alpha


def accumulate(acc: np.ndarray, wgt: np.ndarray,
               img: np.ndarray, alpha: np.ndarray,
               x: int, y: int):
    """Add img*alpha and alpha into the float accumulators at position (x,y)."""
    h, w   = img.shape
    ch, cw = acc.shape

    x1 = max(0, x);      y1 = max(0, y)
    x2 = min(cw, x + w); y2 = min(ch, y + h)
    ix1 = x1 - x; iy1 = y1 - y
    ix2 = ix1 + (x2 - x1); iy2 = iy1 + (y2 - y1)

    if x2 > x1 and y2 > y1:
        a = alpha[iy1:iy2, ix1:ix2]
        acc[y1:y2, x1:x2] += img[iy1:iy2, ix1:ix2].astype(np.float32) * a
        wgt[y1:y2, x1:x2] += a


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def run():
    print("\nLoading orientation results ...")
    b4748 = load_best("47_48")   # A:top   <-> B:bottom  (hub seam)
    b4548 = load_best("45_48")   # A:right <-> B:left    (45 direct to 48)

    # Canvas layout
    #      [45]
    #      [48]
    #      [47]
    print("Rotating and trimming maps ...")
    m47 = load_map("47", 0.0,               "top")
    m48 = load_map("48", b4748["total_rot"], "bottom")
    m45 = load_map("45", 270.0,             "bottom")

    h45, w45 = m45.shape
    h47, w47 = m47.shape
    h48, w48 = m48.shape

    # ------------------------------------------------------------------
    # Canvas positions — OVR-pixel overlap at every seam
    # ------------------------------------------------------------------
    x47, y47 = 0, 0

    # 48 above 47
    x48 = x47 + (w47 - w48) // 2
    y48 = y47 - h48 + OVR

    # 45 above 48
    x45 = x48 + (w48 - w45) // 2
    y45 = y48 - h45 + OVR

    # ------------------------------------------------------------------
    # Alpha masks — contact_edges = edges that TOUCH another map
    # ------------------------------------------------------------------
    a47 = make_alpha(h47, w47, ['top'])
    a48 = make_alpha(h48, w48, ['bottom', 'top'])
    a45 = make_alpha(h45, w45, ['bottom'])

    # ------------------------------------------------------------------
    # Build canvas — allocate with generous padding
    # ------------------------------------------------------------------
    positions = [
        (m47, a47, x47, y47),
        (m48, a48, x48, y48),
        (m45, a45, x45, y45),
    ]

    min_x = min(p[2] for p in positions) - PAD
    min_y = min(p[3] for p in positions) - PAD
    max_x = max(p[2] + p[0].shape[1] for p in positions) + PAD
    max_y = max(p[3] + p[0].shape[0] for p in positions) + PAD

    cw = max_x - min_x
    ch = max_y - min_y

    acc = np.zeros((ch, cw), dtype=np.float32)
    wgt = np.zeros((ch, cw), dtype=np.float32)

    print(f"Canvas: {cw} × {ch} px  (scale={SCALE})")

    for (img, alpha, x, y) in positions:
        content = np.clip((245.0 - img.astype(np.float32)) / 15.0, 0.0, 1.0)
        accumulate(acc, wgt, img, alpha * content, x - min_x, y - min_y)

    # ------------------------------------------------------------------
    # Normalise: covered pixels → weighted average; empty → white
    # ------------------------------------------------------------------
    mask   = wgt > 1e-6
    canvas = np.full((ch, cw), 255.0, dtype=np.float32)
    canvas[mask] = acc[mask] / wgt[mask]
    canvas = np.clip(canvas, 0, 255).astype(np.uint8)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ok, enc = cv2.imencode(".png", canvas)
    if ok:
        enc.tofile(str(OUT_PATH))
        print(f"Saved: {OUT_PATH}")
    else:
        print("ERROR: could not encode image")


if __name__ == "__main__":
    run()
