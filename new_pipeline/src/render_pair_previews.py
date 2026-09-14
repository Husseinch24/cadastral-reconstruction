"""
Render a pairwise stitched preview for every adjacent map pair using the
manual homographies saved in output/homographies/homographies.json.

Outputs one PNG per pair in output/pair_previews/.

Usage
-----
  python new_pipeline/src/render_pair_previews.py
  python new_pipeline/src/render_pair_previews.py --source binary
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

PREPROCESSED_DIR = Path("output/preprocessed")
HOMOGRAPHIES_FILE = Path("output/homographies/homographies.json")
OUTPUT_DIR       = Path("output/matches")

MAX_DIM = 8000   # cap the longer canvas dimension to keep file sizes reasonable


def read_image(path: Path) -> np.ndarray:
    data = np.fromfile(str(path), dtype=np.uint8)
    gray = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise FileNotFoundError(f"Cannot read: {path}")
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def save_image(path: Path, img: np.ndarray):
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, enc = cv2.imencode(path.suffix, img)
    if ok:
        enc.tofile(str(path))


def build_pair(pair_key: str, entry: dict, source: str) -> np.ndarray:
    map_a = entry["map_a"]
    map_b = entry["map_b"]
    H     = np.array(entry["H"], dtype=np.float64)

    img_a = read_image(PREPROCESSED_DIR / f"map_{map_a}_{source}.png")
    img_b = read_image(PREPROCESSED_DIR / f"map_{map_b}_{source}.png")

    h_a, w_a = img_a.shape[:2]
    h_b, w_b = img_b.shape[:2]

    # Warp B's corners through H to find the combined canvas bounds
    corners_b = np.float32(
        [[0, 0], [w_b, 0], [w_b, h_b], [0, h_b]]
    ).reshape(-1, 1, 2)
    warped_b_corners = cv2.perspectiveTransform(corners_b, H).reshape(-1, 2)

    corners_a = np.float32([[0, 0], [w_a, 0], [w_a, h_a], [0, h_a]])

    all_x = np.concatenate([corners_a[:, 0], warped_b_corners[:, 0]])
    all_y = np.concatenate([corners_a[:, 1], warped_b_corners[:, 1]])

    x_min = int(np.floor(all_x.min()))
    y_min = int(np.floor(all_y.min()))
    x_max = int(np.ceil(all_x.max()))
    y_max = int(np.ceil(all_y.max()))

    canvas_w = x_max - x_min
    canvas_h = y_max - y_min

    scale = min(1.0, MAX_DIM / max(canvas_w, canvas_h))
    out_w = int(canvas_w * scale)
    out_h = int(canvas_h * scale)

    # Translation matrix to shift everything into the positive quadrant + scale
    T = np.array(
        [[scale, 0,     -x_min * scale],
         [0,     scale, -y_min * scale],
         [0,     0,     1.0]],
        dtype=np.float64,
    )

    warped_a = cv2.warpPerspective(
        img_a, T, (out_w, out_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT, borderValue=255,
    )
    warped_b = cv2.warpPerspective(
        img_b, T @ H, (out_w, out_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT, borderValue=255,
    )

    # Content masks (anything not pure white)
    mask_a = np.any(warped_a < 240, axis=2)
    mask_b = np.any(warped_b < 240, axis=2)

    canvas = np.full((out_h, out_w, 3), 255, dtype=np.uint8)
    only_a  = mask_a & ~mask_b
    only_b  = ~mask_a & mask_b
    overlap = mask_a & mask_b

    canvas[only_a] = warped_a[only_a]
    canvas[only_b] = warped_b[only_b]
    if overlap.any():
        avg = (
            (warped_a[overlap].astype(np.uint16)
             + warped_b[overlap].astype(np.uint16)) // 2
        ).astype(np.uint8)
        canvas[overlap] = avg

    # Label bar
    label = f"Pair {pair_key}   map {map_a}  +  map {map_b}   ({source})"
    bar_h = max(40, int(out_h * 0.018))
    cv2.rectangle(canvas, (0, 0), (out_w, bar_h), (30, 30, 30), -1)
    font_scale = bar_h / 38.0
    cv2.putText(
        canvas, label, (12, int(bar_h * 0.78)),
        cv2.FONT_HERSHEY_SIMPLEX, font_scale,
        (255, 255, 255), max(1, int(font_scale * 1.5)), cv2.LINE_AA,
    )

    return canvas


def main():
    parser = argparse.ArgumentParser(
        description="Render one stitched preview image per adjacent map pair."
    )
    parser.add_argument(
        "--source", default="clean", choices=["clean", "binary"],
        help="Which preprocessed image version to use (default: clean)."
    )
    args = parser.parse_args()
    source = args.source

    if not HOMOGRAPHIES_FILE.exists():
        print(f"ERROR: {HOMOGRAPHIES_FILE} not found.")
        return

    data = json.load(open(HOMOGRAPHIES_FILE, encoding="utf-8"))
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\nRendering {len(data)} pair previews  (source={source})")
    print("=" * 60)

    for pair_key, entry in data.items():
        print(f"  Pair {pair_key} ...", end="  ", flush=True)
        try:
            pano = build_pair(pair_key, entry, source)
            out_path = OUTPUT_DIR / f"map_{pair_key.replace('_', '_')}_match.png"
            save_image(out_path, pano)
            print(f"saved  {pano.shape[1]}x{pano.shape[0]} px  →  {out_path}")
        except Exception as e:
            print(f"FAILED — {e}")

    print("=" * 60)
    print(f"Done.  Previews saved in:  {OUTPUT_DIR}/\n")


if __name__ == "__main__":
    main()
