"""
Render a panorama for a custom subset of maps.

Usage
-----
  python output/code/render_subset_panorama.py --maps 45 46 47 48 49
  python output/code/render_subset_panorama.py --maps 45 47 48 49 --out my_panorama.png

The script reads the existing homographies.json, keeps only the pairs where
BOTH maps are in the requested subset, then assembles and saves the panorama.

Output is saved to output/panorama/ (or the path given by --out).
"""

import argparse
import json
import time
import cv2
import numpy as np
from pathlib import Path
from collections import deque, defaultdict


PREPROCESSED_DIR = Path("output/preprocessed")
HOMOGRAPHY_DIR   = Path("output/homographies")
PANORAMA_DIR     = Path("output/panorama")
MAX_PANORAMA_PIXELS = 200_000_000
BLEND_FEATHER_PX = 80


def read_image(path):
    data = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Cannot read: {path}")
    return img


def save_image(path, img):
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, enc = cv2.imencode(path.suffix, img)
    if ok:
        enc.tofile(str(path))


def load_homographies(wanted_maps):
    json_path = HOMOGRAPHY_DIR / "homographies.json"
    with open(json_path, "r", encoding="utf-8") as f:
        all_data = json.load(f)

    pairs = {}
    for key, info in all_data.items():
        if "H" not in info:
            continue
        a, b = key.split("_")
        if a not in wanted_maps or b not in wanted_maps:
            continue
        H = np.array(info["H"], dtype=np.float64)
        if np.any(np.isnan(H)) or np.any(np.isinf(H)):
            print(f"  WARNING: skipping {key} — contains NaN/Inf")
            continue
        pairs[key] = H
        print(f"  Using pair {key}")
    return pairs


def build_graph(pairs):
    graph = defaultdict(list)
    for key, H_AB in pairs.items():
        a, b = key.split("_")
        graph[a].append((b, H_AB))
        graph[b].append((a, np.linalg.inv(H_AB)))
    return dict(graph)


def pick_anchor(graph):
    return max(graph, key=lambda n: len(graph[n]))


def compute_transforms(anchor, graph):
    transforms = {anchor: np.eye(3, dtype=np.float64)}
    queue = deque([anchor])
    while queue:
        cur = queue.popleft()
        for nb, H in graph[cur]:
            if nb not in transforms:
                transforms[nb] = transforms[cur] @ H
                queue.append(nb)
    return transforms


def make_feather_mask(h, w, f):
    fy = np.ones(h, dtype=np.float32)
    fx = np.ones(w, dtype=np.float32)
    if h > 2 * f:
        fy[:f] = np.linspace(0, 1, f)
        fy[-f:] = np.linspace(1, 0, f)
    else:
        fy = np.hanning(h).astype(np.float32) ** 0.5
    if w > 2 * f:
        fx[:f] = np.linspace(0, 1, f)
        fx[-f:] = np.linspace(1, 0, f)
    else:
        fx = np.hanning(w).astype(np.float32) ** 0.5
    return np.outer(fy, fx)


def assemble(transforms):
    maps = list(transforms.keys())

    # Compute bounding box
    all_corners = []
    for m in maps:
        img = read_image(PREPROCESSED_DIR / f"map_{m}_clean.png")
        h, w = img.shape
        corners = np.float32([[0,0],[w,0],[w,h],[0,h]]).reshape(-1,1,2)
        warped = cv2.perspectiveTransform(corners, transforms[m]).reshape(-1,2)
        all_corners.extend(warped.tolist())

    arr = np.array(all_corners)
    x_min = int(np.floor(arr[:,0].min()))
    y_min = int(np.floor(arr[:,1].min()))
    x_max = int(np.ceil(arr[:,0].max()))
    y_max = int(np.ceil(arr[:,1].max()))
    canvas_w = x_max - x_min
    canvas_h = y_max - y_min
    print(f"  Canvas: {canvas_w} x {canvas_h} px")

    pixels = canvas_w * canvas_h
    scale = 1.0
    if pixels > MAX_PANORAMA_PIXELS:
        scale = (MAX_PANORAMA_PIXELS / pixels) ** 0.5
        canvas_w = int(canvas_w * scale)
        canvas_h = int(canvas_h * scale)
        print(f"  Downscaled to: {canvas_w} x {canvas_h}")

    T_offset = np.array([
        [scale, 0,     -x_min * scale],
        [0,     scale, -y_min * scale],
        [0,     0,     1             ]
    ], dtype=np.float64)

    accum_img = np.zeros((canvas_h, canvas_w), dtype=np.float32)
    accum_w   = np.zeros((canvas_h, canvas_w), dtype=np.float32)

    for m in maps:
        print(f"  Warping map {m} ...")
        img = read_image(PREPROCESSED_DIR / f"map_{m}_clean.png")
        h, w = img.shape
        T_full = T_offset @ transforms[m]

        warped = cv2.warpPerspective(img, T_full, (canvas_w, canvas_h),
                                     flags=cv2.INTER_LINEAR,
                                     borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        cover = cv2.warpPerspective(np.ones((h, w), dtype=np.float32),
                                    T_full, (canvas_w, canvas_h),
                                    flags=cv2.INTER_LINEAR,
                                    borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        feather = make_feather_mask(h, w, BLEND_FEATHER_PX)
        feather_w = cv2.warpPerspective(feather, T_full, (canvas_w, canvas_h),
                                        flags=cv2.INTER_LINEAR,
                                        borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        weight = feather_w * cover
        accum_img += warped.astype(np.float32) * weight
        accum_w   += weight

    panorama = np.full((canvas_h, canvas_w), 255, dtype=np.uint8)
    covered = accum_w > 1e-3
    panorama[covered] = np.clip(accum_img[covered] / accum_w[covered], 0, 255).astype(np.uint8)

    # Fill interior white gaps with flat gray matching the map paper tone.
    # Flood-fill from all four corners to mark the outer background,
    # then anything still uncovered is an interior hole -> fill with gray.
    gap_mask = (~covered).astype(np.uint8) * 255
    flood = gap_mask.copy()
    for corner in [(0, 0), (canvas_w - 1, 0), (0, canvas_h - 1), (canvas_w - 1, canvas_h - 1)]:
        if flood[corner[1], corner[0]] == 255:
            cv2.floodFill(flood, None, corner, 0)
    interior_gaps = flood > 0
    if interior_gaps.any():
        paper_gray = int(np.median(panorama[covered]))
        panorama[interior_gaps] = paper_gray
        print(f"  Filled {interior_gaps.sum():,} interior gap pixels with gray={paper_gray}")

    return panorama


def main():
    parser = argparse.ArgumentParser(description="Panorama for a subset of maps")
    parser.add_argument("--maps", nargs="+", required=True,
                        help="Map numbers to include, e.g. --maps 45 46 47 48 49")
    parser.add_argument("--out", type=str, default=None,
                        help="Output filename (saved in output/panorama/). "
                             "Default: subset_<maps>.png")
    args = parser.parse_args()

    wanted = set(args.maps)
    label  = "_".join(sorted(args.maps, key=int))
    out_name = args.out or f"subset_{label}.png"
    out_path = PANORAMA_DIR / out_name

    print(f"\n  Maps requested : {sorted(wanted, key=int)}")

    pairs = load_homographies(wanted)
    if not pairs:
        print("ERROR: no homography pairs found for this map subset.")
        return

    graph = build_graph(pairs)
    reachable = set(graph.keys())
    missing = wanted - reachable
    if missing:
        print(f"  WARNING: maps {missing} have no homography — they will be skipped.")

    anchor = pick_anchor(graph)
    print(f"  Anchor: map {anchor}")

    transforms = compute_transforms(anchor, graph)
    print(f"  Maps in panorama: {sorted(transforms.keys(), key=int)}")

    t0 = time.time()
    panorama = assemble(transforms)
    elapsed = time.time() - t0

    save_image(out_path, panorama)
    print(f"\n  Saved: {out_path.resolve()}")
    print(f"  Size : {panorama.shape[1]} x {panorama.shape[0]} px")
    print(f"  Time : {elapsed:.1f}s\n")


if __name__ == "__main__":
    main()
