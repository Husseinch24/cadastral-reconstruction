"""
OCR Method 2 — PaddleOCR
Masters Thesis - AI-Based Panoramic Cadastral Image Reconstruction
Author: Hussein Chalhoub

Attempts to read Arabic-Indic parcel numbers from Mask R-CNN parcel crops
using PaddleOCR 3.x with rotated text detection.

NOTE: This script crashes silently on Windows 11 due to a PaddlePaddle 3.x
PIR executor + oneDNN conflict. No results are produced on Windows.
Install: pip install paddlepaddle paddleocr

Usage
-----
  python new_pipeline/src/ocr_methods/ocr_2_paddleocr.py --map 45
  python new_pipeline/src/ocr_methods/ocr_2_paddleocr.py --all
"""

import argparse
import json
import os
from pathlib import Path

import cv2
import numpy as np

# Disable oneDNN before importing PaddleOCR (workaround attempt — does not fix crash)
os.environ["FLAGS_use_mkldnn"] = "0"
os.environ["PADDLE_DISABLE_MKLDNN"] = "1"
os.environ["FLAGS_enable_pir_api"] = "0"

PREPROCESSED_DIR = Path("output/preprocessed")
PREDICTIONS_DIR  = Path("new_pipeline/data/predictions")
OUTPUT_DIR       = Path("new_pipeline/data/parcel_numbers/paddleocr")

ROTATION_DEG     = 180       # maps are stored upside-down
MIN_NUMBER       = 2000
MAX_NUMBER       = 3999
CROP_MARGIN_PX   = 10
MIN_CROP_SIZE    = 32        # skip parcels smaller than this


def init_paddle():
    """Initialise PaddleOCR — this is where the Windows crash occurs."""
    from paddleocr import PaddleOCR
    attempts = [
        dict(use_textline_orientation=True, lang="ar", use_gpu=False, enable_mkldnn=False),
        dict(lang="ar", use_gpu=False, enable_mkldnn=False),
        dict(lang="en", use_gpu=False, enable_mkldnn=False),
    ]
    for kwargs in attempts:
        try:
            print(f"  Trying PaddleOCR({kwargs}) ...")
            ocr = PaddleOCR(**kwargs)
            print("  PaddleOCR initialised successfully.")
            return ocr
        except TypeError as e:
            print(f"  Init failed: {e}")
    raise RuntimeError("All PaddleOCR init attempts failed.")


def read_image(path: Path) -> np.ndarray:
    data = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)


def rotate_image(img: np.ndarray, deg: int) -> np.ndarray:
    if deg == 0:
        return img
    if deg == 90:
        return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    if deg == 180:
        return cv2.rotate(img, cv2.ROTATE_180)
    if deg == 270:
        return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    M = cv2.getRotationMatrix2D(
        (img.shape[1] / 2, img.shape[0] / 2), -deg, 1.0
    )
    return cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))


def parse_number(text: str) -> int | None:
    """Convert Arabic-Indic or Western digits to int, validate range."""
    arabic_indic = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")
    cleaned = text.translate(arabic_indic).strip()
    digits = "".join(c for c in cleaned if c.isdigit())
    if len(digits) == 4:
        n = int(digits)
        if MIN_NUMBER <= n <= MAX_NUMBER:
            return n
    return None


def process_map(map_num: str, ocr_engine):
    print(f"\n  Processing map {map_num} ...")

    img_path = PREPROCESSED_DIR / f"map_{map_num}_clean.png"
    parcels_path = PREDICTIONS_DIR / f"map_{map_num}_parcels.json"

    img = read_image(img_path)
    if img is None:
        print(f"  ERROR: cannot read {img_path}")
        return

    img = rotate_image(img, ROTATION_DEG)
    h, w = img.shape

    parcels = json.load(open(parcels_path, encoding="utf-8"))
    print(f"  {len(parcels)} parcels loaded")

    results = []
    for p in parcels:
        bx, by, bw, bh = p["bbox"]
        x1 = max(0, bx - CROP_MARGIN_PX)
        y1 = max(0, by - CROP_MARGIN_PX)
        x2 = min(w, bx + bw + CROP_MARGIN_PX)
        y2 = min(h, by + bh + CROP_MARGIN_PX)

        if (x2 - x1) < MIN_CROP_SIZE or (y2 - y1) < MIN_CROP_SIZE:
            continue

        crop = img[y1:y2, x1:x2]
        crop_bgr = cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR)

        try:
            paddle_result = ocr_engine.ocr(crop_bgr, cls=True)
        except Exception as e:
            print(f"    Parcel {p['parcel_id']}: OCR error — {e}")
            continue

        if not paddle_result or not paddle_result[0]:
            continue

        for line in paddle_result[0]:
            text = line[1][0]
            conf = float(line[1][1])
            number = parse_number(text)
            if number is not None:
                results.append({
                    "parcel_id":         p["parcel_id"],
                    "cx":                p["cx"],
                    "cy":                p["cy"],
                    "bbox":              p["bbox"],
                    "recognised_number": number,
                    "confidence":        conf,
                    "raw_text":          text,
                    "engine":            "paddleocr",
                })
                print(f"    Parcel {p['parcel_id']}: '{text}' → {number} (conf={conf:.2f})")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / f"map_{map_num}_numbers_paddle.json"
    json.dump(results, open(out_path, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print(f"  Saved {len(results)} readings → {out_path}")
    return results


def main():
    parser = argparse.ArgumentParser(
        description="OCR parcel numbers with PaddleOCR."
    )
    parser.add_argument("--map", type=str, default=None,
                        help="Single map number, e.g. --map 45")
    parser.add_argument("--all", action="store_true",
                        help="Run on all 13 maps")
    args = parser.parse_args()

    all_maps = ["43","44","45","46","47","48","49","50","51","52","53","54","55"]
    maps = all_maps if args.all else ([args.map] if args.map else None)
    if maps is None:
        parser.print_help()
        return

    print("\nInitialising PaddleOCR ...")
    try:
        ocr = init_paddle()
    except Exception as e:
        print(f"FATAL: PaddleOCR init failed — {e}")
        print("On Windows this is caused by a PIR+oneDNN conflict in PaddlePaddle 3.x.")
        return

    for m in maps:
        process_map(m, ocr)

    print("\nDone.")


if __name__ == "__main__":
    main()
