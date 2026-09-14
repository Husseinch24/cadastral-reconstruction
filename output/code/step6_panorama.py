"""
=============================================================================
STEP 6: PANORAMIC RECONSTRUCTION
=============================================================================
Masters Thesis - AI-Based Panoramic Cadastral Image Reconstruction
Author  : Hussein Chalhoub

Purpose
--------
Take the per-pair homographies produced by Step 4 (and refined by the
drag-and-place tool) and assemble all maps into final panoramic images
at full original resolution.

Pipeline
---------
  Stage A : Load all homographies from output/homographies/
  Stage B : Build adjacency graph from available pairs
  Stage C : Find connected components (groups of maps that link together)
  Stage D : For each component, pick an anchor map (most central one)
  Stage E : Chain homographies via BFS so every map has a transform to anchor
  Stage F : Compute the panorama bounding box in anchor's coordinate frame
  Stage G : Warp each map into the canvas at full resolution
  Stage H : Blend overlapping regions (feathered alpha blend for smooth seams)
  Stage I : Save each component as a separate panorama PNG
  Stage J : Save a manifest JSON describing what's in each panorama

Output files
-------------
  output/panorama/component_<N>_panorama.png   - the assembled image
  output/panorama/component_<N>_anchor.png     - same anchor map alone (for ref)
  output/panorama/manifest.json                - which maps in which panorama

Why two panoramas (or more)?
-----------------------------
Step 4 produced homographies for some pairs but not all.  When pairs are
missing (e.g. the user could not annotate parcel 3216 on map 54), the
adjacency graph splits into disconnected components.  Each component
becomes its own panorama.  When the missing pairs get aligned later,
re-running Step 6 will produce a single unified panorama.

=============================================================================
"""

import argparse
import json
import sys
import time
import cv2
import numpy as np
from pathlib import Path
from collections import deque, defaultdict


# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------

PREPROCESSED_DIR = Path("output/preprocessed")
HOMOGRAPHY_DIR   = Path("output/homographies")
PANORAMA_DIR     = Path("output/panorama")

# All maps present in the dataset
ALL_MAPS = [str(n) for n in range(43, 56)]

# Memory cap for the panorama canvas - if exceeded, downscale uniformly
# 200 megapixels = roughly 14000 x 14000 - more than enough for 13 maps
# but prevents accidental 1 GB allocations
MAX_PANORAMA_PIXELS = 200_000_000

# Blend feather width in pixels - controls smoothness of overlap seams
BLEND_FEATHER_PX = 80


# ---------------------------------------------------------------------------
# UTILITIES
# ---------------------------------------------------------------------------

def print_step(msg: str):
    print(f"  [Step 6] {msg}")


def read_image(path: Path, grayscale: bool = True) -> np.ndarray:
    """Windows-safe image reader."""
    data = np.fromfile(str(path), dtype=np.uint8)
    flag = cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR
    img = cv2.imdecode(data, flag)
    if img is None:
        raise FileNotFoundError(f"Cannot read: {path}")
    return img


def save_image(path: Path, img: np.ndarray):
    """Windows-safe image writer."""
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, enc = cv2.imencode(path.suffix, img)
    if ok:
        enc.tofile(str(path))


def load_homographies() -> dict:
    """
    Load all pair homographies from the JSON file.
    Returns dict mapping "A_B" -> 3x3 numpy array.
    """
    json_path = HOMOGRAPHY_DIR / "homographies.json"
    if not json_path.exists():
        print_step(f"ERROR: {json_path} not found.  Run Step 4 / drag_and_place first.")
        sys.exit(1)

    with open(json_path, "r", encoding="utf-8") as f:
        all_data = json.load(f)

    pairs = {}
    for key, info in all_data.items():
        if "H" not in info:
            continue
        H = np.array(info["H"], dtype=np.float64)
        # Sanity check: bad homographies sometimes have huge values - skip these
        if np.any(np.isnan(H)) or np.any(np.isinf(H)):
            print_step(f"WARNING: skipping {key} - contains NaN/Inf")
            continue
        pairs[key] = H
    return pairs


# ---------------------------------------------------------------------------
# STAGE B + C: BUILD ADJACENCY GRAPH AND FIND CONNECTED COMPONENTS
# ---------------------------------------------------------------------------

def build_graph(pairs: dict) -> dict:
    """
    Build adjacency mapping: {map_id: [(neighbour_id, H_neighbour_to_self), ...]}

    For pair "A_B" with homography H_AB (which maps B's pixels INTO A's frame):
      - From A's perspective: neighbour is B, transform B->A is H_AB
      - From B's perspective: neighbour is A, transform A->B is inv(H_AB)
    """
    graph = defaultdict(list)
    for key, H_AB in pairs.items():
        a, b = key.split("_")
        graph[a].append((b, H_AB))
        try:
            H_BA = np.linalg.inv(H_AB)
            graph[b].append((a, H_BA))
        except np.linalg.LinAlgError:
            print_step(f"WARNING: H_{key} is singular, skipping reverse direction")
    return dict(graph)


def find_connected_components(graph: dict) -> list[list[str]]:
    """
    Standard BFS to find connected components in the adjacency graph.
    Returns list of components, each component is a list of map IDs.
    """
    visited = set()
    components = []
    for start in graph:
        if start in visited:
            continue
        comp = []
        queue = deque([start])
        visited.add(start)
        while queue:
            node = queue.popleft()
            comp.append(node)
            for neighbour, _ in graph[node]:
                if neighbour not in visited:
                    visited.add(neighbour)
                    queue.append(neighbour)
        components.append(sorted(comp))
    # Sort components by size descending (biggest first)
    components.sort(key=lambda c: -len(c))
    return components


# ---------------------------------------------------------------------------
# STAGE D + E: PICK ANCHOR AND CHAIN HOMOGRAPHIES
# ---------------------------------------------------------------------------

def pick_anchor(component: list[str], graph: dict) -> str:
    """
    Choose the anchor map for a component.

    Heuristic: pick the map with the most direct neighbours within the
    component.  This minimises the chain length for all other maps and
    therefore minimises accumulated transform error.
    """
    best_node = component[0]
    best_degree = -1
    for node in component:
        degree = len([n for n, _ in graph[node] if n in component])
        if degree > best_degree:
            best_degree = degree
            best_node = node
    return best_node


def compute_transforms_to_anchor(anchor: str,
                                    component: list[str],
                                    graph: dict) -> dict[str, np.ndarray]:
    """
    BFS from the anchor to every other map in the component, accumulating
    homographies along the way.

    For each map M, returns the 3x3 matrix that transforms M's pixel
    coordinates into the anchor's coordinate frame.  The anchor itself
    gets the identity matrix.

    Composition rule: if M -> N has H_MN (mapping N into M's frame), then
    transform_to_anchor(N) = transform_to_anchor(M) @ H_MN
    """
    transforms = {anchor: np.eye(3, dtype=np.float64)}
    queue = deque([anchor])
    while queue:
        current = queue.popleft()
        T_current = transforms[current]
        for neighbour, H_current_to_neighbour in graph[current]:
            if neighbour in transforms or neighbour not in component:
                continue
            # H_current_to_neighbour maps neighbour's pixels into current's frame
            # so we chain: anchor <- current <- neighbour
            T_neighbour = T_current @ H_current_to_neighbour
            transforms[neighbour] = T_neighbour
            queue.append(neighbour)
    return transforms


# ---------------------------------------------------------------------------
# STAGE F: PANORAMA BOUNDING BOX
# ---------------------------------------------------------------------------

def compute_panorama_bbox(maps: list[str],
                            transforms: dict[str, np.ndarray]
                            ) -> tuple[int, int, int, int]:
    """
    Compute the union bounding box of all warped maps in the anchor frame.
    Returns (x_min, y_min, x_max, y_max) - integer canvas-aligned.
    """
    all_corners = []
    for m in maps:
        img = read_image(PREPROCESSED_DIR / f"map_{m}_clean.png")
        h, w = img.shape
        corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
        warped = cv2.perspectiveTransform(corners, transforms[m]).reshape(-1, 2)
        all_corners.extend(warped.tolist())

    arr = np.array(all_corners)
    x_min = int(np.floor(arr[:, 0].min()))
    y_min = int(np.floor(arr[:, 1].min()))
    x_max = int(np.ceil(arr[:, 0].max()))
    y_max = int(np.ceil(arr[:, 1].max()))
    return x_min, y_min, x_max, y_max


def maybe_downscale(canvas_w: int, canvas_h: int) -> float:
    """
    If the panorama would exceed MAX_PANORAMA_PIXELS, return a uniform
    downscale factor < 1.0 to apply.  Otherwise return 1.0.
    """
    pixels = canvas_w * canvas_h
    if pixels <= MAX_PANORAMA_PIXELS:
        return 1.0
    factor = (MAX_PANORAMA_PIXELS / pixels) ** 0.5
    print_step(f"WARNING: panorama would be {pixels / 1e6:.0f} MP - "
               f"downscaling by {factor:.3f}")
    return factor


# ---------------------------------------------------------------------------
# STAGE G + H: WARP AND BLEND
# ---------------------------------------------------------------------------

def make_feather_mask(h: int, w: int, feather_px: int) -> np.ndarray:
    """
    Build a feather alpha mask: 1.0 in the centre, falling off linearly
    near the edges to 0.0 at the very edge.  Used for smooth blending
    of overlapping panorama regions.
    """
    if feather_px <= 0:
        return np.ones((h, w), dtype=np.float32)

    f = feather_px
    # 1D feather curves
    fy = np.ones(h, dtype=np.float32)
    fx = np.ones(w, dtype=np.float32)
    if h > 2 * f:
        fy[:f] = np.linspace(0, 1, f, dtype=np.float32)
        fy[-f:] = np.linspace(1, 0, f, dtype=np.float32)
    else:
        # Image too small - use a simple bell curve
        fy = np.hanning(h).astype(np.float32) ** 0.5
    if w > 2 * f:
        fx[:f] = np.linspace(0, 1, f, dtype=np.float32)
        fx[-f:] = np.linspace(1, 0, f, dtype=np.float32)
    else:
        fx = np.hanning(w).astype(np.float32) ** 0.5

    return np.outer(fy, fx)


def assemble_component(component: list[str],
                          transforms: dict[str, np.ndarray],
                          anchor: str
                          ) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Render all maps in this component onto a single canvas.

    Returns:
      panorama   : final blended grayscale image
      anchor_img : just the anchor map alone in the same canvas (for reference)
      info       : dict with bbox, downscale, list of maps placed
    """
    print_step(f"Computing panorama bounding box...")
    x_min, y_min, x_max, y_max = compute_panorama_bbox(component, transforms)
    canvas_w = x_max - x_min
    canvas_h = y_max - y_min
    print_step(f"  Bounding box: ({x_min}, {y_min}) -> ({x_max}, {y_max})  "
               f"= {canvas_w} x {canvas_h} px")

    downscale = maybe_downscale(canvas_w, canvas_h)
    if downscale != 1.0:
        canvas_w = int(canvas_w * downscale)
        canvas_h = int(canvas_h * downscale)
        print_step(f"  Final canvas: {canvas_w} x {canvas_h}")

    # Translation so all warped pixels land in [0, canvas_w] x [0, canvas_h]
    T_offset = np.array([
        [downscale, 0,         -x_min * downscale],
        [0,         downscale, -y_min * downscale],
        [0,         0,         1                  ]
    ], dtype=np.float64)

    # Accumulators for weighted average blending:
    #   accum_image  : sum of (pixel_value * weight) over all maps
    #   accum_weight : sum of weights
    # Final pixel = accum_image / accum_weight (where weight > 0)
    accum_image  = np.zeros((canvas_h, canvas_w), dtype=np.float32)
    accum_weight = np.zeros((canvas_h, canvas_w), dtype=np.float32)
    anchor_image = np.full((canvas_h, canvas_w), 255, dtype=np.uint8)

    for m in component:
        print_step(f"  Warping map {m}...")
        img = read_image(PREPROCESSED_DIR / f"map_{m}_clean.png")
        h, w = img.shape

        # Final transform = panorama offset + downscale + map's transform-to-anchor
        T_full = T_offset @ transforms[m]

        warped = cv2.warpPerspective(
            img, T_full, (canvas_w, canvas_h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT, borderValue=0
        )
        # Coverage mask: 255 where the source map placed pixels
        cover = cv2.warpPerspective(
            np.ones((h, w), dtype=np.float32), T_full,
            (canvas_w, canvas_h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT, borderValue=0
        )
        # Feather mask - smooth alpha falloff at edges of source map
        feather = make_feather_mask(h, w, BLEND_FEATHER_PX)
        feather_warped = cv2.warpPerspective(
            feather, T_full, (canvas_w, canvas_h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT, borderValue=0
        )
        # Combine: only weigh where coverage > 0
        weight = feather_warped * cover

        accum_image  += warped.astype(np.float32) * weight
        accum_weight += weight

        if m == anchor:
            valid = cover > 0
            anchor_image[valid] = warped[valid]

    # Final blend: divide accumulated image by total weight where weight > 0
    print_step(f"  Blending {len(component)} maps...")
    panorama = np.full((canvas_h, canvas_w), 255, dtype=np.uint8)
    valid = accum_weight > 1e-3
    panorama[valid] = np.clip(
        accum_image[valid] / accum_weight[valid], 0, 255
    ).astype(np.uint8)

    info = {
        "anchor":      anchor,
        "maps":        component,
        "bbox_anchor": [x_min, y_min, x_max, y_max],
        "canvas_size": [canvas_w, canvas_h],
        "downscale":   float(downscale),
    }
    return panorama, anchor_image, info


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def run(exclude: list = None):
    print("\n" + "=" * 70)
    print("  STEP 6 - PANORAMIC RECONSTRUCTION")
    print("  Cadastral Panoramic Reconstruction - Masters Thesis")
    print("=" * 70)

    PANORAMA_DIR.mkdir(parents=True, exist_ok=True)

    # Stage A: load homographies
    pairs = load_homographies()
    print_step(f"Loaded {len(pairs)} pair homographies:")
    for k in sorted(pairs.keys()):
        print(f"           {k}")
    if not pairs:
        print_step("ERROR: no homographies to assemble. Aborting.")
        return False

    # Stage B: build graph
    graph = build_graph(pairs)
    map_ids_in_graph = sorted(graph.keys())
    print_step(f"Maps appearing in graph: {map_ids_in_graph}")

    # Maps that exist in the dataset but have no homography are skipped.
    # Note them so the user knows.
    missing = [m for m in ALL_MAPS
               if (PREPROCESSED_DIR / f"map_{m}_clean.png").exists()
               and m not in graph]
    if missing:
        print_step(f"WARNING: maps not in any pair (will be skipped): {missing}")

    # Stage C: connected components
    components = find_connected_components(graph)
    if exclude:
        components = [[m for m in comp if m not in exclude] for comp in components]
        components = [comp for comp in components if comp]
        print_step(f"Excluding maps: {exclude}")
    print_step(f"Connected components: {len(components)}")
    for i, comp in enumerate(components):
        print(f"           Component {i}: {comp}")

    manifest = {"components": []}

    for i, component in enumerate(components):
        print(f"\n  {'-' * 60}")
        print(f"  Assembling component {i}: {component}")
        print(f"  {'-' * 60}")

        # Stage D: pick anchor
        anchor = pick_anchor(component, graph)
        print_step(f"Anchor: map {anchor} ({len([n for n, _ in graph[anchor] if n in component])} direct neighbours)")

        # Stage E: chain homographies
        transforms = compute_transforms_to_anchor(anchor, component, graph)
        if len(transforms) != len(component):
            print_step(f"WARNING: only {len(transforms)}/{len(component)} maps reachable")

        t0 = time.time()

        # Stage F+G+H: bbox, warp, blend
        panorama, _, info = assemble_component(
            list(transforms.keys()), transforms, anchor
        )

        elapsed = time.time() - t0

        # Stage I: save
        out_path = PANORAMA_DIR / f"component_{i}_panorama.png"
        save_image(out_path, panorama)
        print_step(f"  Saved: {out_path.name}  ({panorama.shape[1]} x {panorama.shape[0]})")
        print_step(f"  Done in {elapsed:.1f}s")

        info["panorama_file"] = out_path.name
        info["elapsed_sec"]   = round(elapsed, 2)
        manifest["components"].append(info)

    # Stage J: save manifest
    manifest_path = PANORAMA_DIR / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 70)
    print("  STEP 6 SUMMARY")
    print("=" * 70)
    print_step(f"Components produced: {len(components)}")
    for i, info in enumerate(manifest["components"]):
        cw, ch = info["canvas_size"]
        print_step(f"  Component {i}: {len(info['maps'])} maps -> "
                   f"{info['panorama_file']} ({cw} x {ch} px)")
    if missing:
        print_step(f"Maps NOT included (no homography): {missing}")

    print(f"\n  Output: {PANORAMA_DIR.resolve()}")
    print(f"\n  [Step 6] Status: SUCCESS")
    print("=" * 70 + "\n")
    return True


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--exclude", nargs="+", metavar="MAP", default=[],
                    help="Map numbers to skip, e.g. --exclude 50 51")
    args = ap.parse_args()
    ok = run(exclude=args.exclude)
    sys.exit(0 if ok else 1)
