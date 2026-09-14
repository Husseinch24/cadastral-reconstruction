"""
=============================================================================
PANORAMA — ALL 13 CADASTRAL MAPS (43–55)
=============================================================================
Builds a panoramic view of all 13 cadastral maps using a greedy BFS spanning
tree seeded on the validated pair_47_48 result.

Placement rules
---------------
• Each canvas edge of a map may hold at most ONE neighbour (occupied_edges).
• If two candidates compete for the same edge the higher-scoring one wins;
  the loser stays in the heap and finds another slot via a different pair.
• Before committing a placement the centre of the incoming map must not fall
  inside any already-placed map's bounding box — this prevents a bad pair
  score from stacking one map on top of another.
• After the main BFS a second pass retries every still-unplaced map against
  ALL loaded pairs so that maps stranded by edge-conflicts are rescued.

Algorithm
---------
1. Pre-place map 47 at 0° and map 48 at 179° (from pair_47_48 best result).
2. Load all pair_X_Y_scores.json files and build a complete scored edge graph.
3. Greedy BFS: at each step pick the highest-scoring candidate from the heap.
4. For each placed map compute canvas rotation via the chain formula and
   canvas position via the edge_facing placement rule:

     Forward  (pair A→B, A is ref already placed):
       alpha_B = (alpha_A + total_rot) % 360
       direction A→B = edge_facing(edge_a, alpha_A)

     Backward (pair A→B, B is ref already placed):
       alpha_A = (alpha_B - total_rot + 360) % 360
       direction B→A = -edge_facing(edge_a, alpha_A)

     edge_facing(edge, alpha): apply (dx,dy)→(dy,-dx) for each 90° CCW
     quarter-turn, matching cv2.ROTATE_90_COUNTERCLOCKWISE convention.

5. Accumulate all contact edges per map, build linear-ramp alpha masks and
   blend with the weighted-average accumulator.
=============================================================================
"""

import argparse
import heapq
import json
from pathlib import Path
import sys

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from edge_orientation_finder import (
    PREPROCESSED_DIR, RESULTS_DIR,
    auto_crop, read_image, rotate_image, _trim_contact_edge,
)

SCALE    = 0.10
OVR      = 35
PAD      = 200
MAPS     = [str(n) for n in range(43, 56)]
OUT_PATH = RESULTS_DIR / "panorama_all_maps.png"

_EDGE_DIR = {'top': (0, -1), 'bottom': (0, 1), 'left': (-1, 0), 'right': (1, 0)}
_DIR_EDGE = {v: k for k, v in _EDGE_DIR.items()}


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def edge_facing(edge: str, alpha: float):
    """Canvas direction this edge faces after rotating the image by alpha° CCW."""
    dx, dy = _EDGE_DIR[edge]
    for _ in range(round(alpha / 90) % 4):
        dx, dy = dy, -dx
    return dx, dy


def place_offset(dx: int, dy: int,
                 x_ref: int, y_ref: int, w_ref: int, h_ref: int,
                 w_new: int, h_new: int):
    """Canvas top-left corner of new map given direction (dx,dy) from ref to new."""
    if dy == -1:
        return x_ref + (w_ref - w_new) // 2, y_ref - h_new + OVR
    elif dy == 1:
        return x_ref + (w_ref - w_new) // 2, y_ref + h_ref - OVR
    elif dx == -1:
        return x_ref - w_new + OVR, y_ref + (h_ref - h_new) // 2
    else:
        return x_ref + w_ref - OVR, y_ref + (h_ref - h_new) // 2


def overlaps_placed(x_new: int, y_new: int, w_new: int, h_new: int,
                    placed: dict) -> bool:
    """
    True if the new map overlaps any placed map by more than half of the new
    map's own area in BOTH axes simultaneously.

    Legitimate OVR-pixel seam overlaps are narrow in one axis (≤ OVR+5 px) so
    they always pass.  Only maps placed almost entirely on top of an existing
    map (wrong pair score) are rejected.
    """
    threshold_x = w_new * 0.45
    threshold_y = h_new * 0.45
    for info in placed.values():
        xi, yi = info["x"], info["y"]
        wi, hi = info["img"].shape[1], info["img"].shape[0]
        ox = max(0, min(x_new + w_new, xi + wi) - max(x_new, xi))
        oy = max(0, min(y_new + h_new, yi + hi) - max(y_new, yi))
        if ox > threshold_x and oy > threshold_y:
            return True
    return False


# ---------------------------------------------------------------------------
# Image I/O
# ---------------------------------------------------------------------------

def load_img(map_num: str, rot: float, trim_edge=None) -> np.ndarray:
    img = auto_crop(read_image(PREPROCESSED_DIR / f"map_{map_num}_clean.png"))
    img = rotate_image(img, rot)
    img = auto_crop(img)   # remove white fill added by warpAffine for non-90° angles
    if trim_edge is not None:
        img = _trim_contact_edge(img, trim_edge)
    h, w = img.shape
    return cv2.resize(img, (int(w * SCALE), int(h * SCALE)),
                      interpolation=cv2.INTER_AREA)


# ---------------------------------------------------------------------------
# Alpha / blending
# ---------------------------------------------------------------------------

def make_alpha(h: int, w: int, contact_edges: list) -> np.ndarray:
    alpha = np.ones((h, w), dtype=np.float32)
    ramp  = np.linspace(0.0, 1.0, OVR, dtype=np.float32)
    for edge in contact_edges:
        if   edge == 'top':    alpha[:OVR, :]   *= ramp[:, np.newaxis]
        elif edge == 'bottom': alpha[-OVR:, :]  *= ramp[::-1, np.newaxis]
        elif edge == 'left':   alpha[:, :OVR]   *= ramp[np.newaxis, :]
        elif edge == 'right':  alpha[:, -OVR:]  *= ramp[::-1][np.newaxis, :]
    return alpha


def accumulate(acc: np.ndarray, wgt: np.ndarray,
               img: np.ndarray, alpha: np.ndarray,
               x: int, y: int):
    h, w   = img.shape
    ch, cw = acc.shape
    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(cw, x + w), min(ch, y + h)
    ix1, iy1 = x1 - x, y1 - y
    ix2, iy2 = ix1 + (x2 - x1), iy1 + (y2 - y1)
    if x2 > x1 and y2 > y1:
        a = alpha[iy1:iy2, ix1:ix2]
        acc[y1:y2, x1:x2] += img[iy1:iy2, ix1:ix2].astype(np.float32) * a
        wgt[y1:y2, x1:x2] += a


# ---------------------------------------------------------------------------
# Pair loading
# ---------------------------------------------------------------------------

def load_all_pairs(maps: list = None) -> dict:
    """Return dict (map_a_str, map_b_str) → best-result dict for all pairs."""
    active = maps or MAPS
    pairs = {}
    for mn_a in active:
        for mn_b in active:
            if int(mn_a) >= int(mn_b):
                continue
            path = RESULTS_DIR / f"pair_{mn_a}_{mn_b}_scores.json"
            if path.exists():
                with open(path, encoding="utf-8") as f:
                    pairs[(mn_a, mn_b)] = json.load(f)["best"]
    return pairs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(exclude: list = None, out_path: Path = None):
    maps    = [m for m in MAPS if m not in (exclude or [])]
    out     = out_path or OUT_PATH
    print("\nLoading all pair results ...")
    pairs = load_all_pairs(maps)
    print(f"  {len(pairs)} pairs loaded  (maps: {', '.join(maps)})")

    # ------------------------------------------------------------------
    # Seed: pick the best-scoring available pair as the starting point.
    # Prefer pair 47_48 if both maps are selected and the result exists,
    # otherwise fall back to whichever pair scores highest.
    # ------------------------------------------------------------------
    if not pairs:
        print("ERROR: no orientation results found for the selected maps.")
        print("Run edge_orientation_finder.py first for at least one pair.")
        return

    if ("47", "48") in pairs:
        seed_key = ("47", "48")
    else:
        seed_key = max(pairs.keys(), key=lambda k: pairs[k]["combined_score"])

    map_a_s, map_b_s = seed_key
    best_s  = pairs[seed_key]
    rot_s   = best_s["total_rot"]
    edge_as = best_s["edge_a"]   # edge of A that touches B
    edge_bs = best_s["edge_b"]   # edge of B that touches A (in rotated frame)

    print(f"Loading seed maps {map_a_s} and {map_b_s}  "
          f"(score={best_s['combined_score']:.4f}) ...")

    img_as = load_img(map_a_s, 0.0,   edge_as)
    img_bs = load_img(map_b_s, rot_s, edge_bs)
    h_as, w_as = img_as.shape
    h_bs, w_bs = img_bs.shape

    # Direction from A to B on the canvas
    dx_s, dy_s = edge_facing(edge_as, 0.0)
    x_as, y_as = 0, 0
    x_bs, y_bs = place_offset(dx_s, dy_s, x_as, y_as, w_as, h_as, w_bs, h_bs)

    placed = {
        map_a_s: {"rot": 0.0,   "x": x_as, "y": y_as, "img": img_as},
        map_b_s: {"rot": rot_s, "x": x_bs, "y": y_bs, "img": img_bs},
    }
    contact_edges  = {map_a_s: [edge_as],  map_b_s: [edge_bs]}
    occupied_edges = {map_a_s: {edge_as},  map_b_s: {edge_bs}}
    unplaced = set(maps) - set(placed.keys())

    # ------------------------------------------------------------------
    # Placement helper — enforces all three uniqueness constraints
    # ------------------------------------------------------------------
    def try_place(ref_map: str, new_map: str, pair_key: tuple, case: str) -> bool:
        """
        Attempt to place new_map relative to ref_map using the given pair.
        Returns True and commits the placement, or returns False (no side-effect).

        Blocked when:
          • new_map is already placed
          • ref_map's connecting edge is occupied by another map
          • new_map's own connecting edge is already claimed
          • the new map's centre would land inside an existing map (collision)
        """
        if new_map in placed or ref_map not in placed:
            return False

        best      = pairs[pair_key]
        edge_a    = best["edge_a"]
        total_rot = best["total_rot"]
        alpha_ref = placed[ref_map]["rot"]

        if case == "fwd":
            alpha_new = (alpha_ref + total_rot) % 360
            dx, dy    = edge_facing(edge_a, alpha_ref)
        else:
            alpha_new = (alpha_ref - total_rot + 360) % 360
            dx0, dy0  = edge_facing(edge_a, alpha_new)
            dx, dy    = -dx0, -dy0

        contact_new      = _DIR_EDGE[(-dx, -dy)]   # new map's seam edge
        contact_ref_side = _DIR_EDGE[(dx, dy)]     # ref map's seam edge

        # One neighbour per side for BOTH maps
        if contact_ref_side in occupied_edges.get(ref_map, set()):
            return False
        if contact_new in occupied_edges.get(new_map, set()):
            return False

        ref_info     = placed[ref_map]
        h_ref, w_ref = ref_info["img"].shape
        x_ref, y_ref = ref_info["x"], ref_info["y"]

        img_new      = load_img(new_map, alpha_new, contact_new)
        h_new, w_new = img_new.shape
        x_new, y_new = place_offset(dx, dy, x_ref, y_ref, w_ref, h_ref, w_new, h_new)

        # Spatial collision guard: reject if new map heavily overlaps any placed map
        if overlaps_placed(x_new, y_new, w_new, h_new, placed):
            return False

        # Commit
        placed[new_map] = {"rot": alpha_new, "x": x_new, "y": y_new, "img": img_new}
        contact_edges[new_map] = [contact_new]
        contact_edges[ref_map].append(contact_ref_side)
        occupied_edges.setdefault(ref_map, set()).add(contact_ref_side)
        occupied_edges[new_map] = {contact_new}
        unplaced.discard(new_map)

        score = best["combined_score"]
        print(f"  map {new_map:>2s}  rot={alpha_new:6.1f}°  "
              f"placed {_DIR_EDGE[(dx, dy)]:>6s} of {ref_map}  "
              f"score={score:.4f}  pair={pair_key[0]}_{pair_key[1]}")
        return True

    # ------------------------------------------------------------------
    # Build initial heap — all pairs that touch a seed map
    # heap entry: (-score, ref_map, new_map, pair_key, case)
    # ------------------------------------------------------------------
    heap = []
    for (a, b), best in pairs.items():
        score = best["combined_score"]
        if a in placed and b not in placed:
            heapq.heappush(heap, (-score, a, b, (a, b), "fwd"))
        elif b in placed and a not in placed:
            heapq.heappush(heap, (-score, b, a, (a, b), "bwd"))

    print("\nBFS placement ...")
    while heap and unplaced:
        _, ref_map, new_map, pair_key, case = heapq.heappop(heap)
        if new_map in placed or ref_map not in placed:
            continue

        if try_place(ref_map, new_map, pair_key, case):
            # Expand frontier from the newly placed map
            for (a, b), best in pairs.items():
                s = best["combined_score"]
                if a == new_map and b not in placed:
                    heapq.heappush(heap, (-s, new_map, b, (a, b), "fwd"))
                elif b == new_map and a not in placed:
                    heapq.heappush(heap, (-s, new_map, a, (a, b), "bwd"))

    # ------------------------------------------------------------------
    # Second pass — rescue maps stranded by edge conflicts
    # Try ALL loaded pairs, best-score first, until no more progress
    # ------------------------------------------------------------------
    if unplaced:
        print(f"\n  Second-pass for stranded maps: {sorted(unplaced)} ...")
        sorted_pairs = sorted(pairs.items(), key=lambda kv: -kv[1]["combined_score"])
        changed = True
        while changed and unplaced:
            changed = False
            for new_map in sorted(unplaced):
                for (a, b), _ in sorted_pairs:
                    placed_ok = False
                    if a == new_map and b in placed:
                        placed_ok = try_place(b, new_map, (a, b), "bwd")
                    elif b == new_map and a in placed:
                        placed_ok = try_place(a, new_map, (a, b), "fwd")
                    if placed_ok:
                        changed = True
                        break

    # ------------------------------------------------------------------
    # Overflow row — any map that still has no valid auto-position is placed
    # in a clean strip below the main panorama so all 13 maps stay visible.
    # ------------------------------------------------------------------
    if unplaced:
        print(f"\n  Overflow row for {sorted(unplaced)} (no valid auto-position found) ...")
        max_y_placed = max(info["y"] + info["img"].shape[0] for info in placed.values())
        ox = min(info["x"] for info in placed.values())
        oy = max_y_placed + PAD
        for map_num in sorted(unplaced):
            img_ov      = load_img(map_num, 0.0, None)
            _, w_ov  = img_ov.shape
            placed[map_num] = {"rot": 0.0, "x": ox, "y": oy, "img": img_ov}
            contact_edges[map_num]  = []
            occupied_edges[map_num] = set()
            ox += w_ov + PAD // 2
            print(f"    map {map_num}  → overflow at ({placed[map_num]['x']}, {oy})")
        unplaced.clear()

    # ------------------------------------------------------------------
    # Build canvas and blend
    # ------------------------------------------------------------------
    all_pos = [
        (placed[m]["img"], placed[m]["x"], placed[m]["y"])
        for m in placed
    ]
    min_x = min(p[1] for p in all_pos) - PAD
    min_y = min(p[2] for p in all_pos) - PAD
    max_x = max(p[1] + p[0].shape[1] for p in all_pos) + PAD
    max_y = max(p[2] + p[0].shape[0] for p in all_pos) + PAD

    cw, ch = max_x - min_x, max_y - min_y
    acc = np.zeros((ch, cw), dtype=np.float32)
    wgt = np.zeros((ch, cw), dtype=np.float32)
    print(f"\nCanvas: {cw} × {ch} px  (scale={SCALE})")

    for m, info in placed.items():
        img   = info["img"]
        h, w  = img.shape
        x     = info["x"] - min_x
        y     = info["y"] - min_y
        edges = list(dict.fromkeys(contact_edges.get(m, [])))
        alpha = make_alpha(h, w, edges)
        # Fade remaining near-white residual pixels at seam edges
        content = np.clip((245.0 - img.astype(np.float32)) / 15.0, 0.0, 1.0)
        alpha  *= content
        accumulate(acc, wgt, img, alpha, x, y)

    mask   = wgt > 1e-6
    canvas = np.full((ch, cw), 255.0, dtype=np.float32)
    canvas[mask] = acc[mask] / wgt[mask]
    canvas = np.clip(canvas, 0, 255).astype(np.uint8)

    for m, info in placed.items():
        x = info["x"] - min_x + info["img"].shape[1] // 2
        y = info["y"] - min_y + info["img"].shape[0] // 2
        cv2.putText(canvas, m, (x - 15, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, 0, 2, cv2.LINE_AA)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ok, enc = cv2.imencode(".png", canvas)
    if ok:
        enc.tofile(str(out))
        print(f"Saved → {out}")
    else:
        print("ERROR: could not encode image")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--exclude", nargs="+", metavar="MAP", default=[],
                    help="Map numbers to skip, e.g. --exclude 44 45")
    ap.add_argument("--out", metavar="FILE", default=None,
                    help="Output PNG path (default: panorama_all_maps.png)")
    args = ap.parse_args()
    out_path = Path(args.out) if args.out else None
    run(exclude=args.exclude, out_path=out_path)
