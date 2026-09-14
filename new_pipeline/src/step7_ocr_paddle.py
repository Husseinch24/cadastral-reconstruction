"""
=============================================================================
MILESTONE 5 — PADDLE OCR FOR PARCEL NUMBER RECOGNITION
=============================================================================
Masters Thesis - AI-Based Panoramic Cadastral Image Reconstruction
Author : Hussein Chalhoub

Purpose
--------
Read the 4-digit Arabic-Indic parcel numbers written inside each detected
parcel polygon, then cross-reference with the CSV ground truth so we know
which parcels are shared boundary parcels between adjacent map sheets.

This replaces the failed EasyOCR attempt. Key differences:
  - PaddleOCR's DBNet text detector is more robust on small numerals
  - Built-in angle classification handles 0/90/180/270 rotation natively
    (cadastral cartographers wrote each number along the parcel orientation)
  - We feed RAW grayscale, no Otsu binarisation, no contrast enhancement
    (binarisation destroyed the digits in the previous attempt)
  - Tight crops around each Mask R-CNN parcel bbox, NOT a 200 px expansion

Pipeline position
------------------
  step4 (Mask R-CNN)  -> per-map parcel bboxes + centroids
       |
  step7 (this file)   -> per-parcel OCR'd number  <-- you are here
       |
  step8               -> cross-reference with CSV, build control points
       |
  step9               -> homography from control points + contour refinement

Install
--------
PaddleOCR is not in requirements.txt yet. To install:

  .\venv_thesis\Scripts\Activate.ps1
  pip install paddlepaddle paddleocr

For GPU (only if your CUDA matches a published wheel; CUDA 13.0 currently
has no paddlepaddle-gpu wheel, so CPU is the safe default):

  pip install paddlepaddle-gpu paddleocr

Usage
------
  # Test mode: try the pair 45/47 only, looking for parcel 2580 on both maps
  python new_pipeline/src/step7_ocr_paddle.py --test-pair 45_47

  # Run OCR on a single map and save all detected numbers
  python new_pipeline/src/step7_ocr_paddle.py --map 45

  # Run OCR on all 13 maps
  python new_pipeline/src/step7_ocr_paddle.py --all

Output
-------
  new_pipeline/data/parcel_numbers/
    map_<N>_numbers.json     - every parcel's recognised number (or null)
    map_<N>_overlay.png      - visualisation: green dots for hits, red for miss

=============================================================================
"""

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# PADDLEPADDLE WINDOWS WORKAROUND
# ---------------------------------------------------------------------------
# PaddlePaddle 3.x on Windows + CPU has a bug where the new PIR executor
# combined with oneDNN crashes on certain ops with:
#   NotImplementedError: ConvertPirAttribute2RuntimeAttribute not support
#   [pir::ArrayAttribute<pir::DoubleAttribute>]
# Fix: disable oneDNN AND the PIR executor before importing paddle/paddleocr.
# These flags are read at framework init, so they MUST be set before import.
os.environ.setdefault("FLAGS_use_mkldnn",            "0")
os.environ.setdefault("FLAGS_enable_pir_api",        "0")
os.environ.setdefault("FLAGS_enable_pir_in_executor", "0")
os.environ.setdefault("FLAGS_enable_new_ir_in_executor", "0")

import cv2
import numpy as np
from tqdm import tqdm


# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------

PREPROCESSED_DIR = Path("output/preprocessed")
PREDICTIONS_DIR  = Path("new_pipeline/data/predictions")
OUTPUT_DIR       = Path("new_pipeline/data/parcel_numbers")

# CSV ground truth: parcel number -> list of maps it appears on
BOUNDARY_PARCELS_CSV = {
    2580: ["45", "47"],
    2616: ["45", "46"],
    2619: ["45", "46"],
    2749: ["47", "48"],
    2803: ["47", "48"],
    2814: ["47", "48"],
    2893: ["48", "49"],
    3022: ["49", "50"],
    3054: ["49", "50"],
    3068: ["50", "51"],
    3215: ["52", "53"],
    3216: ["52", "54"],
    3217: ["52", "55"],
    3338: ["54", "55"],
    3339: ["54", "55"],
    3345: ["54", "55"],
    3346: ["54", "55"],
    3813: ["50", "51"],
    3866: ["50", "51"],
}

# Per-map list of boundary parcel numbers (for highlighting in test mode)
MAP_BOUNDARY_PARCELS = {}
for parcel_num, maps in BOUNDARY_PARCELS_CSV.items():
    for m in maps:
        MAP_BOUNDARY_PARCELS.setdefault(m, []).append(parcel_num)

# Crop margin around each parcel bbox (px). The number is written inside
# the parcel; we add a small margin so the digit's strokes are not clipped
# at the bbox edge. Too large = neighbouring numbers leak in.
CROP_MARGIN_PX = 25

# Skip parcels whose bbox (after margin) is smaller than this — too small
# to plausibly contain a 4-digit number
MIN_CROP_SIDE_PX = 50

# Skip parcels whose bbox is larger than this — these are usually whole-row
# detection artefacts that contain too much noise to OCR reliably
MAX_CROP_AREA_PX = 1_500_000   # ~1224 x 1224

# Confidence threshold for accepting a detected number
MIN_OCR_CONFIDENCE = 0.30

# Arabic-Indic numerals -> Western digits
ARABIC_INDIC_TO_WESTERN = {
    "٠": "0", "١": "1", "٢": "2", "٣": "3", "٤": "4",
    "٥": "5", "٦": "6", "٧": "7", "٨": "8", "٩": "9",
    # Persian variants sometimes appear in PaddleOCR output
    "۰": "0", "۱": "1", "۲": "2", "۳": "3", "۴": "4",
    "۵": "5", "۶": "6", "۷": "7", "۸": "8", "۹": "9",
}


# ---------------------------------------------------------------------------
# UTILITIES
# ---------------------------------------------------------------------------

def read_image(path: Path, color: bool = False) -> np.ndarray:
    """Read an image, supporting Unicode paths (Windows OneDrive)."""
    data = np.fromfile(str(path), dtype=np.uint8)
    flag = cv2.IMREAD_COLOR if color else cv2.IMREAD_GRAYSCALE
    img = cv2.imdecode(data, flag)
    if img is None:
        raise FileNotFoundError(f"Cannot read: {path}")
    return img


def save_image(path: Path, img: np.ndarray):
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, enc = cv2.imencode(path.suffix, img)
    if ok:
        enc.tofile(str(path))


def normalise_to_western(text: str) -> str:
    """Convert Arabic-Indic / Persian digits to 0-9, drop everything else."""
    out = []
    for ch in text:
        if ch in ARABIC_INDIC_TO_WESTERN:
            out.append(ARABIC_INDIC_TO_WESTERN[ch])
        elif ch.isdigit():
            out.append(ch)
    return "".join(out)


def extract_4digit_numbers(text: str) -> list[int]:
    """Find all 4-digit integers in [2000, 9999] in OCR text."""
    norm = normalise_to_western(text)
    result = []
    for match in re.findall(r"\d{4}", norm):
        n = int(match)
        if 2000 <= n <= 9999:
            result.append(n)
    return result


# ---------------------------------------------------------------------------
# PADDLEOCR ENGINE (lazy-loaded)
# ---------------------------------------------------------------------------

_paddle_ocr = None
_paddle_api_v3 = False   # PaddleOCR 3.x uses predict(); 2.x uses ocr()


def get_paddle_ocr():
    """
    Lazy-load PaddleOCR. First call downloads ~300 MB of model files.

    Handles both PaddleOCR 2.x and 3.x init signatures, which renamed
    `use_angle_cls` to `use_textline_orientation` and removed `show_log`.
    """
    global _paddle_ocr, _paddle_api_v3
    if _paddle_ocr is not None:
        return _paddle_ocr

    try:
        from paddleocr import PaddleOCR
    except ImportError:
        print("\n  ERROR: PaddleOCR is not installed.")
        print("  Install with:")
        print("    pip install paddlepaddle paddleocr")
        sys.exit(1)

    print("  Loading PaddleOCR (first run downloads ~300 MB)...")

    # Try init signatures in order: 3.x first (current default), then 2.x.
    # The Arabic model handles Arabic-Indic digits.
    # NOTE: enable_mkldnn=False is REQUIRED on Windows + Paddle 3.x to avoid
    # the PIR-executor + oneDNN crash:
    #   ConvertPirAttribute2RuntimeAttribute not support
    #   [pir::ArrayAttribute<pir::DoubleAttribute>]
    init_attempts = [
        # PaddleOCR 3.x with oneDNN explicitly disabled
        (True,  dict(use_textline_orientation=True, lang="arabic",
                     enable_mkldnn=False)),
        (True,  dict(use_textline_orientation=True, lang="ar",
                     enable_mkldnn=False)),
        # PaddleOCR 3.x defaults (may crash on Windows)
        (True,  dict(use_textline_orientation=True, lang="arabic")),
        (True,  dict(use_textline_orientation=True, lang="ar")),
        # PaddleOCR 2.x
        (False, dict(use_angle_cls=True, lang="ar", enable_mkldnn=False,
                     show_log=False)),
        (False, dict(use_angle_cls=True, lang="ar", show_log=False)),
        (False, dict(use_angle_cls=True, lang="ar")),
    ]

    last_err = None
    for is_v3, kwargs in init_attempts:
        try:
            _paddle_ocr = PaddleOCR(**kwargs)
            _paddle_api_v3 = is_v3
            print(f"  PaddleOCR ready (api_v3={is_v3}, lang={kwargs['lang']})")
            return _paddle_ocr
        except (TypeError, ValueError) as e:
            last_err = e
            continue

    raise RuntimeError(f"Could not initialise PaddleOCR: {last_err}")


def _parse_v3_result(results) -> list[tuple[str, float]]:
    """
    PaddleOCR 3.x predict() returns a list whose items expose
    `rec_texts` and `rec_scores` (either as dict keys or attributes).
    """
    out = []
    if not results:
        return out
    for res in results:
        texts = None
        scores = None
        # Try dict-like access first
        try:
            texts  = res["rec_texts"]
            scores = res["rec_scores"]
        except (KeyError, TypeError, IndexError):
            pass
        # Fall back to attribute access
        if texts is None:
            texts  = getattr(res, "rec_texts",  None)
            scores = getattr(res, "rec_scores", None)
        if not texts or not scores:
            continue
        for text, conf in zip(texts, scores):
            try:
                conf = float(conf)
            except (TypeError, ValueError):
                continue
            text = str(text).strip()
            if conf >= MIN_OCR_CONFIDENCE and text:
                out.append((text, conf))
    return out


def _parse_v2_result(result) -> list[tuple[str, float]]:
    """PaddleOCR 2.x ocr() returns nested lists [[ [bbox, (text, conf)], ... ]]."""
    out = []
    if not result or not result[0]:
        return out
    for line in result[0]:
        try:
            text, conf = line[1][0], float(line[1][1])
        except (IndexError, TypeError, ValueError):
            continue
        text = str(text).strip()
        if conf >= MIN_OCR_CONFIDENCE and text:
            out.append((text, conf))
    return out


def ocr_crop(crop: np.ndarray) -> list[tuple[str, float]]:
    """
    Run PaddleOCR on a grayscale crop. Returns list of (text, confidence).

    PaddleOCR expects a 3-channel image. We replicate the grayscale channel
    rather than applying any preprocessing — PaddleOCR's pipeline handles
    its own normalisation.
    """
    ocr = get_paddle_ocr()

    if crop.ndim == 2:
        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_GRAY2RGB)
    else:
        crop_rgb = crop

    if _paddle_api_v3:
        try:
            results = ocr.predict(crop_rgb)
        except Exception:
            return []
        return _parse_v3_result(results)
    else:
        try:
            result = ocr.ocr(crop_rgb, cls=True)
        except TypeError:
            # Some 2.x builds dropped the cls kwarg
            try:
                result = ocr.ocr(crop_rgb)
            except Exception:
                return []
        except Exception:
            return []
        return _parse_v2_result(result)


# ---------------------------------------------------------------------------
# PER-PARCEL DETECTION
# ---------------------------------------------------------------------------

def crop_for_parcel(img: np.ndarray, parcel: dict) -> np.ndarray | None:
    """
    Crop a tight window around a parcel's bounding box plus a small margin.
    Returns None if the crop is too small or too large to be plausible.
    """
    H, W = img.shape[:2]
    x, y, w, h = parcel["bbox"]
    x0 = max(0, x - CROP_MARGIN_PX)
    y0 = max(0, y - CROP_MARGIN_PX)
    x1 = min(W, x + w + CROP_MARGIN_PX)
    y1 = min(H, y + h + CROP_MARGIN_PX)

    cw, ch = x1 - x0, y1 - y0
    if cw < MIN_CROP_SIDE_PX or ch < MIN_CROP_SIDE_PX:
        return None
    if cw * ch > MAX_CROP_AREA_PX:
        return None
    return img[y0:y1, x0:x1]


def detect_numbers_on_map(map_num: str,
                          target_set: set[int] | None = None,
                          max_parcels: int | None = None,
                          ) -> list[dict]:
    """
    Run OCR on every parcel of a map, return list of detection dicts.

    If `target_set` is provided, the function still OCRs every parcel but
    keeps a tally of which target numbers were hit. Targets are the CSV
    boundary parcels we expect to see on this map.
    """
    img_path  = PREPROCESSED_DIR / f"map_{map_num}_clean.png"
    pred_path = PREDICTIONS_DIR  / f"map_{map_num}_parcels.json"

    if not img_path.exists():
        raise FileNotFoundError(f"Map image not found: {img_path}")
    if not pred_path.exists():
        raise FileNotFoundError(
            f"Parcel predictions not found: {pred_path}\n"
            f"Run step4_segmentation.py --infer --all first."
        )

    img = read_image(img_path, color=False)
    with open(pred_path, "r", encoding="utf-8") as f:
        parcels = json.load(f)

    if max_parcels:
        parcels = parcels[:max_parcels]

    detections: list[dict] = []
    target_hits: dict[int, dict] = {}

    pbar = tqdm(parcels, desc=f"  Map {map_num}", leave=False)
    for p in pbar:
        crop = crop_for_parcel(img, p)
        if crop is None:
            continue

        ocr_results = ocr_crop(crop)
        best_for_parcel: tuple[int, float, str] | None = None

        for text, conf in ocr_results:
            for num in extract_4digit_numbers(text):
                if best_for_parcel is None or conf > best_for_parcel[1]:
                    best_for_parcel = (num, conf, text)

        if best_for_parcel is None:
            continue

        num, conf, raw = best_for_parcel
        det = {
            "parcel_id":         int(p["parcel_id"]),
            "cx":                float(p["cx"]),
            "cy":                float(p["cy"]),
            "bbox":              p["bbox"],
            "recognised_number": int(num),
            "confidence":        round(float(conf), 4),
            "raw_text":          raw,
            "is_csv_target":     bool(target_set and num in target_set),
        }
        detections.append(det)

        if target_set and num in target_set:
            if num not in target_hits or conf > target_hits[num]["confidence"]:
                target_hits[num] = det
                pbar.set_postfix_str(f"hit {num} (conf={conf:.2f})")

    return detections


# ---------------------------------------------------------------------------
# VISUALISATION
# ---------------------------------------------------------------------------

def visualise_detections(map_num: str,
                          detections: list[dict],
                          out_path: Path,
                          target_set: set[int] | None = None):
    """Render detected numbers on top of the original map (down-scaled)."""
    img_path = PREPROCESSED_DIR / f"map_{map_num}_clean.png"
    img = read_image(img_path, color=True)

    scale = min(1.0, 3000 / max(img.shape[:2]))
    if scale < 1.0:
        img = cv2.resize(
            img,
            (int(img.shape[1] * scale), int(img.shape[0] * scale)),
        )

    n_targets = 0
    for det in detections:
        cx = int(det["cx"] * scale)
        cy = int(det["cy"] * scale)
        is_target = bool(target_set and det["recognised_number"] in target_set)
        if is_target:
            colour = (0, 255, 0)   # bright green = CSV ground truth hit
            n_targets += 1
            cv2.circle(img, (cx, cy), 14, colour, -1)
        else:
            colour = (0, 165, 255)  # orange = recognised but not in CSV
            cv2.circle(img, (cx, cy), 6, colour, -1)
        cv2.putText(
            img, str(det["recognised_number"]),
            (cx + 12, cy + 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55, colour, 2, cv2.LINE_AA,
        )

    # Header
    cv2.rectangle(img, (0, 0), (img.shape[1], 50), (30, 30, 30), -1)
    header = (
        f"Map {map_num} | {len(detections)} numbers found | "
        f"{n_targets} CSV ground-truth hits"
    )
    cv2.putText(img, header, (10, 33),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (255, 255, 255), 2, cv2.LINE_AA)

    save_image(out_path, img)


# ---------------------------------------------------------------------------
# MAIN ROUTINES
# ---------------------------------------------------------------------------

def save_detections(map_num: str, detections: list[dict]):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / f"map_{map_num}_numbers.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(detections, f, indent=2, ensure_ascii=False)
    return out_path


def run_test_pair(pair_key: str):
    """
    Diagnostic: run OCR on the two maps of one pair and report whether the
    expected CSV-ground-truth boundary parcels were found on both.
    """
    a, b = pair_key.split("_")
    expected = [n for n, maps in BOUNDARY_PARCELS_CSV.items()
                if a in maps and b in maps]
    if not expected:
        print(f"  No CSV boundary parcels expected for pair {pair_key}.")
        return

    target_set = set(expected)
    print(f"\n  Test pair {pair_key}: expecting {expected} on both maps.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    print(f"\n  --- Map {a} ---")
    det_a = detect_numbers_on_map(a, target_set)
    save_detections(a, det_a)
    visualise_detections(a, det_a,
                         OUTPUT_DIR / f"map_{a}_overlay.png",
                         target_set)

    print(f"\n  --- Map {b} ---")
    det_b = detect_numbers_on_map(b, target_set)
    save_detections(b, det_b)
    visualise_detections(b, det_b,
                         OUTPUT_DIR / f"map_{b}_overlay.png",
                         target_set)

    # Report
    found_a = {d["recognised_number"] for d in det_a if d["is_csv_target"]}
    found_b = {d["recognised_number"] for d in det_b if d["is_csv_target"]}
    confirmed = found_a & found_b

    print("\n" + "=" * 70)
    print(f"  PAIR {pair_key} OCR RESULT  ({time.time() - t0:.1f}s)")
    print("=" * 70)
    print(f"  Expected boundary parcels   : {expected}")
    print(f"  Found on map {a:>2}              : {sorted(found_a) or '[]'}")
    print(f"  Found on map {b:>2}              : {sorted(found_b) or '[]'}")
    print(f"  Confirmed on BOTH (control points): {sorted(confirmed) or '[]'}")
    print(f"  Total numbers read on map {a}: {len(det_a)}")
    print(f"  Total numbers read on map {b}: {len(det_b)}")
    print(f"\n  Visualisations:")
    print(f"    {OUTPUT_DIR / f'map_{a}_overlay.png'}")
    print(f"    {OUTPUT_DIR / f'map_{b}_overlay.png'}")
    print("=" * 70)

    if confirmed:
        print("\n  PaddleOCR works on this pair — proceed with --all.")
    else:
        print("\n  PaddleOCR did NOT find any CSV-ground-truth match.")
        print("  Inspect the overlay PNGs above before running --all.")
        print("  If still failing, fall back to a Vision LLM (next step).")


def run_one_map(map_num: str):
    target_set = set(MAP_BOUNDARY_PARCELS.get(map_num, []))
    print(f"\n  Running PaddleOCR on map {map_num}")
    if target_set:
        print(f"  CSV expects boundary parcels: {sorted(target_set)}")
    t0 = time.time()
    detections = detect_numbers_on_map(map_num, target_set)
    save_detections(map_num, detections)
    visualise_detections(
        map_num, detections,
        OUTPUT_DIR / f"map_{map_num}_overlay.png",
        target_set,
    )
    hits = sum(1 for d in detections if d["is_csv_target"])
    print(f"\n  Map {map_num}: {len(detections)} numbers read, "
          f"{hits} CSV ground-truth hits  ({time.time() - t0:.1f}s)")


def run_all():
    print("\n" + "=" * 70)
    print("  STEP 7 — PADDLEOCR ON ALL 13 MAPS")
    print("=" * 70)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    summary = {}
    for n in range(43, 56):
        m = str(n)
        pred_path = PREDICTIONS_DIR / f"map_{m}_parcels.json"
        if not pred_path.exists():
            print(f"  Skipping map {m} (no Mask R-CNN predictions)")
            continue
        run_one_map(m)
        json_path = OUTPUT_DIR / f"map_{m}_numbers.json"
        with open(json_path, "r", encoding="utf-8") as f:
            dets = json.load(f)
        summary[m] = {
            "n_numbers_read": len(dets),
            "n_csv_hits":     sum(1 for d in dets if d["is_csv_target"]),
        }

    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    print(f"  {'Map':<5} {'Numbers':>8} {'CSV hits':>10}")
    for m, s in summary.items():
        print(f"  {m:<5} {s['n_numbers_read']:>8} {s['n_csv_hits']:>10}")
    print(f"\n  Total time: {time.time() - t0:.1f}s")
    print(f"  Output: {OUTPUT_DIR.resolve()}")
    print("\n  Next: step8 to cross-reference numbers and build control points")
    print("=" * 70 + "\n")


# ---------------------------------------------------------------------------
# DEBUG MODES
# ---------------------------------------------------------------------------

def run_debug_parcels(map_num: str, n_samples: int = 10):
    """
    Diagnostic: run OCR on the first N parcels and dump everything.

    Saves each crop to disk so you can visually confirm the parcels
    actually contain readable numbers, and prints the raw PaddleOCR
    result so we can tell whether the detector is silent or whether
    our parser is dropping valid output.
    """
    out_dir = OUTPUT_DIR / "debug" / f"map_{map_num}"
    out_dir.mkdir(parents=True, exist_ok=True)

    img_path  = PREPROCESSED_DIR / f"map_{map_num}_clean.png"
    pred_path = PREDICTIONS_DIR  / f"map_{map_num}_parcels.json"
    img = read_image(img_path, color=False)
    with open(pred_path, "r", encoding="utf-8") as f:
        parcels = json.load(f)

    # Force PaddleOCR to load before the loop, so the print order is sane
    ocr = get_paddle_ocr()

    samples = parcels[:n_samples]
    print(f"\n  Debugging {len(samples)} parcels of map {map_num}")
    print(f"  Crops saved under: {out_dir}\n")

    for p in samples:
        crop = crop_for_parcel(img, p)
        if crop is None:
            print(f"  parcel {p['parcel_id']}: SKIP "
                  f"(crop too small/large for bbox {p['bbox']})")
            continue

        crop_path = out_dir / f"parcel_{p['parcel_id']}.png"
        save_image(crop_path, crop)

        # Run OCR with no confidence filtering and dump RAW
        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_GRAY2RGB)
        try:
            if _paddle_api_v3:
                raw = ocr.predict(crop_rgb)
            else:
                raw = ocr.ocr(crop_rgb)
        except Exception as e:
            print(f"  parcel {p['parcel_id']}: OCR raised {type(e).__name__}: {e}")
            continue

        print(f"  parcel {p['parcel_id']:>4}  "
              f"bbox={p['bbox']}  crop={crop.shape[1]}x{crop.shape[0]}")
        print(f"    raw type: {type(raw).__name__}")

        if _paddle_api_v3:
            # PaddleOCR 3.x: raw is a list of OCRResult-like objects
            if not raw:
                print(f"    raw is empty list/None")
                continue
            for j, res in enumerate(raw):
                print(f"    [{j}] type={type(res).__name__}")
                # Try dict-style access first, then attributes, then dir()
                texts, scores, polys = None, None, None
                try:
                    texts  = res["rec_texts"]
                    scores = res["rec_scores"]
                    polys  = res.get("rec_polys") or res.get("dt_polys")
                except (KeyError, TypeError, IndexError, AttributeError):
                    texts  = getattr(res, "rec_texts",  None)
                    scores = getattr(res, "rec_scores", None)
                    polys  = getattr(res, "rec_polys", None) or \
                             getattr(res, "dt_polys",  None)
                print(f"        rec_texts:  {texts}")
                print(f"        rec_scores: {scores}")
                if polys is not None:
                    n_polys = len(polys) if hasattr(polys, "__len__") else "?"
                    print(f"        n_polys:    {n_polys}")
                if texts is None and scores is None:
                    # Fall back to dumping public attributes
                    pubs = [a for a in dir(res) if not a.startswith("_")]
                    print(f"        attrs: {pubs[:20]}")
        else:
            # 2.x: nested list of [bbox_quad, (text, conf)]
            if not raw or not raw[0]:
                print(f"    no detections")
                continue
            for line in raw[0]:
                print(f"    line: {line}")


def run_debug_full(map_num: str):
    """
    Run PaddleOCR on the FULL preprocessed map (downscaled to ~3000 px wide),
    not on per-parcel crops. Tells us whether PaddleOCR can find ANY text
    at all on the cadastral sheet — independent of our parcel cropping.
    """
    out_dir = OUTPUT_DIR / "debug"
    out_dir.mkdir(parents=True, exist_ok=True)

    img_path = PREPROCESSED_DIR / f"map_{map_num}_clean.png"
    img = read_image(img_path, color=False)
    h, w = img.shape

    target_w = 3000
    scale = min(1.0, target_w / max(h, w))
    if scale < 1.0:
        img = cv2.resize(img, (int(w * scale), int(h * scale)))
        print(f"  Downscaled map {map_num} by {scale:.2f}: "
              f"{img.shape[1]}x{img.shape[0]}")

    crop_path = out_dir / f"map_{map_num}_fulltest.png"
    save_image(crop_path, img)

    ocr = get_paddle_ocr()
    crop_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    print("  Running OCR on full map...")
    t0 = time.time()
    try:
        raw = ocr.predict(crop_rgb) if _paddle_api_v3 else ocr.ocr(crop_rgb)
    except Exception as e:
        print(f"  OCR raised {type(e).__name__}: {e}")
        return
    print(f"  Elapsed: {time.time() - t0:.1f}s")

    if _paddle_api_v3:
        if not raw:
            print("  No detections returned (raw is empty).")
            return
        for j, res in enumerate(raw):
            print(f"  [{j}] type={type(res).__name__}")
            try:
                texts  = res["rec_texts"]
                scores = res["rec_scores"]
            except (KeyError, TypeError, AttributeError):
                texts  = getattr(res, "rec_texts",  None)
                scores = getattr(res, "rec_scores", None)
            print(f"    n_texts: {len(texts) if texts else 0}")
            if texts:
                # show top 30 detections
                for k, (t, c) in enumerate(zip(texts[:30], scores[:30])):
                    norm = normalise_to_western(str(t))
                    print(f"    {k:>3}  conf={float(c):.3f}  "
                          f"text={t!r}  norm_digits={norm!r}")
                if len(texts) > 30:
                    print(f"    ... +{len(texts) - 30} more")
    else:
        if not raw or not raw[0]:
            print("  No detections returned.")
            return
        for k, line in enumerate(raw[0][:30]):
            print(f"    {k:>3}  {line}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="PaddleOCR-based parcel-number recognition"
    )
    parser.add_argument("--test-pair", type=str, default=None,
                        help="Run on one pair only, e.g. --test-pair 45_47")
    parser.add_argument("--map", type=str, default=None,
                        help="Run on a single map number, e.g. --map 45")
    parser.add_argument("--all", action="store_true",
                        help="Run on all 13 maps")
    parser.add_argument("--debug-parcels", type=str, default=None,
                        help="Diagnostic: dump first 10 parcel crops + raw "
                             "OCR for one map, e.g. --debug-parcels 45")
    parser.add_argument("--debug-full", type=str, default=None,
                        help="Diagnostic: run OCR on the full map "
                             "(downscaled), e.g. --debug-full 45")
    parser.add_argument("--n-samples", type=int, default=10,
                        help="Number of parcels for --debug-parcels")
    args = parser.parse_args()

    if args.debug_full:
        run_debug_full(args.debug_full)
    elif args.debug_parcels:
        run_debug_parcels(args.debug_parcels, args.n_samples)
    elif args.test_pair:
        run_test_pair(args.test_pair)
    elif args.map:
        run_one_map(args.map)
    elif args.all:
        run_all()
    else:
        print("Usage:")
        print("  python new_pipeline/src/step7_ocr_paddle.py --test-pair 45_47")
        print("  python new_pipeline/src/step7_ocr_paddle.py --map 45")
        print("  python new_pipeline/src/step7_ocr_paddle.py --all")
        print("  python new_pipeline/src/step7_ocr_paddle.py --debug-full 45")
        print("  python new_pipeline/src/step7_ocr_paddle.py --debug-parcels 45")
