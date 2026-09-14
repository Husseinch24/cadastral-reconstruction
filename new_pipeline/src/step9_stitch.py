"""
=============================================================================
MILESTONE 7 — STITCH MAPS USING THE MANUAL HOMOGRAPHIES (no OCR needed)
=============================================================================
Masters Thesis - AI-Based Panoramic Cadastral Image Reconstruction
Author : Hussein Chalhoub

Purpose
--------
Take the manually-fit pair-wise homographies in
  output/homographies/homographies.json
and produce one panorama image per connected component of adjacent sheets.

This is the END of the pipeline given known control points. We are
deliberately decoupling stitching from automatic control-point discovery
(OCR / shape matching) so we can validate that the stitching machinery
itself is correct. Once the OCR / shape-matching pipeline outputs a
homographies.json of its own, this script consumes it unchanged.

Algorithm
----------
  1. Build an undirected adjacency graph: nodes = maps, edges = pair
     homographies. For each edge (A, B) we know H_AB which maps map B's
     native coordinates into map A's native coordinates.
  2. Find connected components.
  3. For each component:
       a. Pick the anchor — the map with the most connections in the
          component (BFS frontier shrinks fastest from there).
       b. BFS from anchor; for every other map M, compose the chain of
          pair-wise H matrices to get T_M : M_native -> Anchor_native.
       c. Compute the bounding box of all four warped corners of every
          map. Translate the whole canvas so the bbox starts at (0, 0) —
          this removes the whitespace around the result automatically.
       d. Optionally downscale if the canvas exceeds MAX_CANVAS_PX.
       e. Warp each map onto the canvas with cv2.warpPerspective.
          Compositing rule: first-writer-wins, with the anchor warped
          first so its data is preserved where overlaps occur.

Output
-------
  new_pipeline/data/panoramas/
    component_<i>_anchor_<N>.png            - the stitched panorama
    component_<i>_anchor_<N>_manifest.json  - sizes, transforms, scale

Usage
------
  .\\venv_thesis\\Scripts\\Activate.ps1
  python new_pipeline/src/step9_stitch.py            # all components
  python new_pipeline/src/step9_stitch.py --anchor 45
=============================================================================
"""

import argparse
import json
from collections import defaultdict, deque
from pathlib import Path

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------

PREPROCESSED_DIR  = Path("output/preprocessed")
HOMOGRAPHIES_PATH = Path("output/homographies/homographies.json")
OUTPUT_DIR        = Path("new_pipeline/data/panoramas")

# Cap the canvas longest side. Larger = more detail but more RAM. The output
# panorama component_0 has ~7 maps; full resolution would be ~15000 x 25000.
MAX_CANVAS_PX = 12000


# ---------------------------------------------------------------------------
# UTILITIES
# ---------------------------------------------------------------------------

def read_image_color(path: Path) -> np.ndarray | None:
    if not path.exists():
        return None
    data = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def save_image(path: Path, img: np.ndarray):
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, enc = cv2.imencode(path.suffix, img)
    if ok:
        enc.tofile(str(path))


# ---------------------------------------------------------------------------
# ADJACENCY + TRANSFORM COMPOSITION
# ---------------------------------------------------------------------------

def load_homographies() -> dict:
    if not HOMOGRAPHIES_PATH.exists():
        raise FileNotFoundError(
            f"Manual homographies not found at {HOMOGRAPHIES_PATH}.\n"
            f"This script depends on the user's manual click work."
        )
    with open(HOMOGRAPHIES_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def build_graph(homog: dict) -> dict[str, dict[str, np.ndarray]]:
    """
    Build a directed adjacency graph where graph[A][B] is the homography
    that maps map B's native pixel coordinates into map A's native pixel
    coordinates. We also store the inverse so the graph is fully traversable.
    """
    graph: dict[str, dict[str, np.ndarray]] = defaultdict(dict)
    for pair_key, data in homog.items():
        if not isinstance(data, dict) or "H" not in data:
            continue
        a = str(data.get("map_a") or pair_key.split("_")[0])
        b = str(data.get("map_b") or pair_key.split("_")[1])
        H_ba = np.array(data["H"], dtype=np.float64)   # B -> A
        if H_ba.shape != (3, 3):
            continue
        graph[a][b] = H_ba
        try:
            graph[b][a] = np.linalg.inv(H_ba)
        except np.linalg.LinAlgError:
            pass
    return graph


def connected_components(graph: dict[str, dict[str, np.ndarray]]
                          ) -> list[list[str]]:
    visited: set[str] = set()
    comps: list[list[str]] = []
    for start in graph:
        if start in visited:
            continue
        comp: list[str] = []
        q = deque([start])
        while q:
            x = q.popleft()
            if x in visited:
                continue
            visited.add(x)
            comp.append(x)
            for nb in graph.get(x, {}):
                if nb not in visited:
                    q.append(nb)
        comps.append(sorted(comp))
    return comps


def compose_transforms(graph: dict[str, dict[str, np.ndarray]],
                        anchor: str,
                        component: set[str]
                        ) -> dict[str, np.ndarray]:
    """
    BFS from the anchor through the component. For every map M, return T_M
    such that  T_M @ p_M = p_in_anchor_frame  for any homogeneous point p.
    """
    transforms: dict[str, np.ndarray] = {anchor: np.eye(3, dtype=np.float64)}
    q = deque([anchor])
    while q:
        x = q.popleft()
        for nb, H_xn in graph.get(x, {}).items():
            if nb in transforms or nb not in component:
                continue
            # H_xn maps nb -> x (native coords). transforms[x] maps x ->
            # anchor. So composition gives nb -> anchor.
            transforms[nb] = transforms[x] @ H_xn
            q.append(nb)
    return transforms


# ---------------------------------------------------------------------------
# PANORAMA CONSTRUCTION
# ---------------------------------------------------------------------------

def warped_corners(map_num: str, T: np.ndarray
                    ) -> tuple[np.ndarray, tuple[int, int]] | None:
    """Return the four corners of map_num's image warped by T, plus its (h, w)."""
    img = read_image_color(PREPROCESSED_DIR / f"map_{map_num}_clean.png")
    if img is None:
        return None
    h, w = img.shape[:2]
    corners = np.array(
        [[0, 0, 1], [w, 0, 1], [w, h, 1], [0, h, 1]],
        dtype=np.float64,
    ).T   # 3x4
    warped = T @ corners
    warped = (warped[:2] / warped[2:3]).T   # 4x2
    return warped, (h, w)


def stitch_component(anchor: str,
                      component: list[str],
                      graph: dict[str, dict[str, np.ndarray]],
                      output_path: Path) -> dict:
    print(f"\n  Component anchor={anchor}, "
          f"{len(component)} maps: {component}")

    transforms = compose_transforms(graph, anchor, set(component))
    missing = set(component) - set(transforms)
    if missing:
        print(f"    Could not reach maps {missing} from anchor — skipping them")
        component = [m for m in component if m in transforms]

    # Compute the bounding box of every map's warped corners.
    all_corners: list[np.ndarray] = []
    sizes: dict[str, tuple[int, int]] = {}
    for m in component:
        info = warped_corners(m, transforms[m])
        if info is None:
            print(f"    Map {m}: image not found, skipping")
            continue
        corners, size = info
        all_corners.append(corners)
        sizes[m] = size

    if not all_corners:
        print("    No usable maps in this component.")
        return {}

    pts = np.vstack(all_corners)
    x_min, y_min = pts.min(axis=0)
    x_max, y_max = pts.max(axis=0)

    raw_w = float(x_max - x_min)
    raw_h = float(y_max - y_min)

    # Cap canvas for memory; everything scales together so the geometry stays
    # right.
    scale = min(1.0, MAX_CANVAS_PX / max(raw_w, raw_h, 1.0))

    # canvas_T : anchor coordinates -> canvas pixel coordinates
    canvas_T = np.array(
        [[scale, 0,     -scale * x_min],
         [0,     scale, -scale * y_min],
         [0,     0,     1.0           ]],
        dtype=np.float64,
    )
    canvas_w = int(np.ceil(raw_w * scale))
    canvas_h = int(np.ceil(raw_h * scale))
    print(f"    Canvas: {canvas_w} x {canvas_h}  (scale {scale:.3f})")

    canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
    written = np.zeros((canvas_h, canvas_w),    dtype=np.uint8)

    # Warp the anchor first so its pixels survive overlaps. Then the rest in
    # BFS order — each one fills only where the canvas is still empty.
    bfs_order = [anchor] + [m for m in component if m != anchor]

    manifest_transforms: dict[str, list[list[float]]] = {}

    for m in bfs_order:
        if m not in sizes:
            continue
        T_full = canvas_T @ transforms[m]
        manifest_transforms[m] = T_full.tolist()

        img = read_image_color(PREPROCESSED_DIR / f"map_{m}_clean.png")
        if img is None:
            continue

        warped = cv2.warpPerspective(
            img, T_full, (canvas_w, canvas_h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT, borderValue=0,
        )
        mask = cv2.warpPerspective(
            np.full(img.shape[:2], 255, dtype=np.uint8),
            T_full, (canvas_w, canvas_h),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT, borderValue=0,
        )
        free = (written == 0) & (mask > 0)
        canvas[free] = warped[free]
        written = np.maximum(written, mask)

    # Trim any all-black border that may remain (shouldn't, but safe).
    ys, xs = np.where(written > 0)
    if len(xs) and len(ys):
        x0, x1 = xs.min(), xs.max() + 1
        y0, y1 = ys.min(), ys.max() + 1
        canvas = canvas[y0:y1, x0:x1]
        written = written[y0:y1, x0:x1]
        canvas_w, canvas_h = canvas.shape[1], canvas.shape[0]

    save_image(output_path, canvas)
    coverage = float((written > 0).mean())
    print(f"    Saved: {output_path.name}  "
          f"({canvas_w} x {canvas_h}, coverage {coverage * 100:.1f}%)")

    return {
        "anchor":         anchor,
        "maps":           component,
        "canvas_size":    [canvas_w, canvas_h],
        "scale":          scale,
        "anchor_bbox_in_native":
            [float(x_min), float(y_min), float(x_max), float(y_max)],
        "transforms":     manifest_transforms,
        "panorama_file":  output_path.name,
        "coverage":       coverage,
    }


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def run(anchor_override: str | None = None):
    print("\n" + "=" * 70)
    print("  STEP 9 — STITCH MAPS FROM MANUAL HOMOGRAPHIES")
    print("=" * 70)

    homog = load_homographies()
    graph = build_graph(homog)
    if not graph:
        print("  Empty homography graph — nothing to stitch.")
        return

    components = connected_components(graph)
    print(f"\n  Found {len(components)} connected component(s):")
    for i, c in enumerate(components):
        print(f"    Component {i}: {c}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    manifests = []

    for i, comp in enumerate(components):
        if len(comp) < 2:
            print(f"\n  Skipping single-map component: {comp}")
            continue

        if anchor_override and anchor_override in comp:
            anchor = anchor_override
        else:
            # The map with most connections IN THIS COMPONENT
            anchor = max(comp,
                         key=lambda m: sum(1 for n in graph.get(m, {})
                                            if n in comp))

        out_name = f"component_{i}_anchor_{anchor}.png"
        out_path = OUTPUT_DIR / out_name
        manifest = stitch_component(anchor, comp, graph, out_path)
        if manifest:
            manifest["component_index"] = i
            manifests.append(manifest)
            mf_path = OUTPUT_DIR / out_path.with_suffix(".json").name
            with open(mf_path, "w", encoding="utf-8") as f:
                json.dump(manifest, f, indent=2)

    print("\n" + "=" * 70)
    print(f"  Stitched {len(manifests)} panorama(s)")
    for m in manifests:
        print(f"    - {m['panorama_file']}: anchor {m['anchor']}, "
              f"{len(m['maps'])} maps, "
              f"canvas {m['canvas_size'][0]}x{m['canvas_size'][1]}")
    print(f"\n  Output: {OUTPUT_DIR.resolve()}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Stitch maps using manual homographies."
    )
    parser.add_argument("--anchor", type=str, default=None,
                        help="Force a specific anchor map "
                             "(must be present in a component). "
                             "Default: most-connected map per component.")
    args = parser.parse_args()
    run(args.anchor)
