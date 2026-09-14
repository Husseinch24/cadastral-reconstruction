"""
=============================================================================
STEP 5: PARCEL DETECTION & OCR NUMBER RECOGNITION
=============================================================================
Masters Thesis — AI-Based Panoramic Cadastral Image Reconstruction
Author  : Hussein Chalhoub
Purpose : Detect individual land parcels on each cadastral map and read
          their Arabic-Indic parcel numbers using OCR.  The output JSON
          files are consumed by Step 4 Stage 2 (parcel centroid refinement)
          to produce precise ground control point pairs for homography
          refinement.

Why this step is critical for the pipeline
-------------------------------------------
Step 4 Stage 1 uses border-line cross-correlation to estimate a rough
alignment between adjacent maps.  For some pairs the shift estimates are
unreliable because too few lines cross the map edge.  Step 4 Stage 2
corrects this by using PARCEL CENTROIDS as ground control points:

  Parcel 2580 appears on BOTH map 45 and map 47 (it straddles the boundary)
  → its centroid in map 45 pixels and in map 47 pixels are the same
    real-world location
  → this gives an exact correspondence pair for homography refinement

For this to work, Step 5 must:
  1. Find where each parcel is (centroid in pixel coordinates)
  2. Read which parcel number is written inside it

Pipeline inside this step
--------------------------
  Stage A : Load Step 1 preprocessed images (grayscale + binary)
  Stage B : Contour-based parcel segmentation
  Stage C : Per-parcel image preprocessing for OCR
            (padding, upscaling, CLAHE, denoising)
  Stage D : Tesseract OCR (digits + Arabic-Indic numerals)
  Stage E : EasyOCR (deep-learning, better on handwritten numerals)
  Stage F : Result fusion (agree = high confidence; one engine = lower)
  Stage G : PDF-index cross-validation
            (check detected number falls in expected range for this map)
  Stage H : Save JSON (for Step 4 Stage 2) + CSV + annotated image

Output files (per map N)
-------------------------
  output/detected/map_<N>_parcels.json    - parcel list with centroids + numbers
  output/detected/map_<N>_ocr.csv         - per-parcel OCR results table
  output/detected/map_<N>_annotated.png   - colour-coded detection image

JSON format (what Step 4 Stage 2 reads)
----------------------------------------
  [
    {
      "parcel_id":        0,
      "detected_number":  2580,
      "cx":               1234.5,
      "cy":               678.9,
      "confidence":       0.85,
      "validation":       "VALID",
      "area_px":          12400.0,
      "bbox":             [x, y, w, h]
    },
    ...
  ]

=============================================================================
"""

import sys
import time
import json
import csv
import cv2
import numpy as np
from pathlib import Path

# OCR engines imported conditionally
try:
    import pytesseract
    # Windows installation path — update if installed elsewhere
    pytesseract.pytesseract.tesseract_cmd = \
        r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    TESSERACT_OK = True
except ImportError:
    TESSERACT_OK = False

try:
    import easyocr
    EASYOCR_OK = True
except ImportError:
    EASYOCR_OK = False


# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------

PREPROCESSED_DIR = Path("output/preprocessed")
DETECTED_DIR     = Path("output/detected")
DEBUG_DIR        = Path("output/debug/step5")

MAP_NUMBERS = [str(n) for n in range(43, 56)]

# Parcel segmentation
MIN_PARCEL_AREA      = 800
MAX_PARCEL_AREA_FRAC = 0.80
MIN_OCR_DIM          = 25

# OCR preprocessing
OCR_PAD_PX     = 15
OCR_UPSCALE    = 3
OCR_CLAHE_CLIP = 3.0
OCR_CLAHE_TILE = (4, 4)

# OCR fusion weights
TESS_WEIGHT    = 0.40
EASYOCR_WEIGHT = 0.60
SINGLE_PENALTY = 0.70

# Tesseract config
TESS_CONFIG = (
    "--psm 6 --oem 3 "
    "-c tessedit_char_whitelist=0123456789\u0660\u0661\u0662\u0663\u0664"
    "\u0665\u0666\u0667\u0668\u0669"
)

EASYOCR_LANGS = ['ar', 'en']

# PDF index: expected parcel number ranges per map
# BORDER parcels (within ±50 of range boundary) are the ones that appear
# on two sheets and are the key inputs for Step 4 Stage 2 refinement
PARCEL_RANGE_PER_MAP = {
    "43": (2477, 2500),
    "44": (2451, 2476),
    "45": (2501, 2750),
    "46": (2751, 3000),
    "47": (2701, 2750),
    "48": (2751, 3000),
    "49": (2951, 3250),
    "50": (3201, 3500),
    "51": (3051, 3250),
    "52": (3201, 3500),
    "53": (3251, 3500),
    "54": (3301, 3500),
    "55": (3351, 3500),
}
BORDER_TOLERANCE = 50


# ---------------------------------------------------------------------------
# ARABIC-INDIC NORMALISATION
# ---------------------------------------------------------------------------

ARABIC_INDIC_MAP = str.maketrans(
    "\u0660\u0661\u0662\u0663\u0664\u0665\u0666\u0667\u0668\u0669",
    "0123456789"
)


def normalise_number(text: str) -> int | None:
    """
    Convert OCR output to an integer parcel number.
    Handles Arabic-Indic numerals, Western numerals, and mixed strings.
    Returns None if no valid 4-5 digit number found.
    """
    if not text:
        return None
    translated = text.translate(ARABIC_INDIC_MAP)
    digits = ''.join(c for c in translated if c.isdigit())
    if not digits:
        return None
    try:
        val = int(digits)
        if 100 <= val <= 99999:
            return val
    except ValueError:
        pass
    return None


# ---------------------------------------------------------------------------
# UTILITIES
# ---------------------------------------------------------------------------

def print_step(msg: str):
    print(f"  [Step 5] {msg}")


def make_dirs():
    for d in [DETECTED_DIR, DEBUG_DIR]:
        d.mkdir(parents=True, exist_ok=True)
    print_step("Output directories ready")


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


def load_map_images(map_num: str) -> tuple[np.ndarray, np.ndarray]:
    clean_path  = PREPROCESSED_DIR / f"map_{map_num}_clean.png"
    binary_path = PREPROCESSED_DIR / f"map_{map_num}_binary.png"

    if not clean_path.exists():
        raise FileNotFoundError(
            f"Preprocessed image not found for map {map_num}. "
            f"Run step1_preprocessing.py first."
        )

    gray = read_image(clean_path, grayscale=True)

    if binary_path.exists():
        binary = read_image(binary_path, grayscale=True)
    else:
        print_step(f"WARNING: binary not found for map {map_num} — "
                   f"computing from grayscale")
        _, binary = cv2.threshold(
            gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

    return gray, binary


# ---------------------------------------------------------------------------
# STAGE B: PARCEL SEGMENTATION
# ---------------------------------------------------------------------------

def segment_parcels(binary: np.ndarray, map_num: str) -> list[dict]:
    """
    Detect parcel regions as closed contours in the binary image.
    Binary convention: ink=0 (black), paper=255 (white).
    We invert first so parcel interiors become white blobs.
    RETR_CCOMP returns two-level hierarchy — we keep only top-level
    contours (outer boundaries, not holes).
    """
    H, W = binary.shape
    max_area = MAX_PARCEL_AREA_FRAC * H * W
    inverted = cv2.bitwise_not(binary)

    contours, hierarchy = cv2.findContours(
        inverted, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE
    )

    if contours is None or len(contours) == 0:
        print_step(f"  Map {map_num}: no contours found")
        return []

    parcels = []
    pid = 0

    for i, contour in enumerate(contours):
        if hierarchy[0][i][3] != -1:   # skip holes
            continue

        area = cv2.contourArea(contour)
        if area < MIN_PARCEL_AREA or area > max_area:
            continue

        M = cv2.moments(contour)
        if M["m00"] == 0:
            continue

        cx = float(M["m10"] / M["m00"])
        cy = float(M["m01"] / M["m00"])
        x, y, bw, bh = cv2.boundingRect(contour)

        if bw < MIN_OCR_DIM or bh < MIN_OCR_DIM:
            continue

        parcels.append({
            "id":      pid,
            "area_px": float(area),
            "cx":      cx,
            "cy":      cy,
            "bbox":    [int(x), int(y), int(bw), int(bh)],
            "contour": contour,
        })
        pid += 1

    print_step(f"  Map {map_num}: {len(parcels)} parcel regions detected")
    return parcels


# ---------------------------------------------------------------------------
# STAGE C: OCR PREPROCESSING
# ---------------------------------------------------------------------------

def preprocess_for_ocr(gray: np.ndarray,
                        bbox: list[int]) -> np.ndarray | None:
    """
    Crop a parcel region and enhance for OCR:
    1. Padded crop
    2. 3x upscale (OCR engines need text >= 20px tall)
    3. CLAHE contrast enhancement
    4. Light Gaussian denoise
    """
    H, W = gray.shape
    x, y, bw, bh = bbox

    x1 = max(0, x - OCR_PAD_PX)
    y1 = max(0, y - OCR_PAD_PX)
    x2 = min(W, x + bw + OCR_PAD_PX)
    y2 = min(H, y + bh + OCR_PAD_PX)

    crop = gray[y1:y2, x1:x2]
    if crop.size == 0:
        return None

    up_w = int(crop.shape[1] * OCR_UPSCALE)
    up_h = int(crop.shape[0] * OCR_UPSCALE)
    if up_w < 10 or up_h < 10:
        return None

    upscaled = cv2.resize(crop, (up_w, up_h), interpolation=cv2.INTER_CUBIC)
    clahe    = cv2.createCLAHE(clipLimit=OCR_CLAHE_CLIP,
                                tileGridSize=OCR_CLAHE_TILE)
    enhanced = clahe.apply(upscaled)
    denoised = cv2.GaussianBlur(enhanced, (3, 3), sigmaX=0.5)

    return denoised


# ---------------------------------------------------------------------------
# STAGE D: TESSERACT
# ---------------------------------------------------------------------------

def run_tesseract(crop: np.ndarray) -> tuple[int | None, float]:
    if not TESSERACT_OK or crop is None:
        return None, 0.0
    try:
        data = pytesseract.image_to_data(
            crop,
            config=TESS_CONFIG,
            lang='ara+eng',
            output_type=pytesseract.Output.DICT
        )
        best_text = ""
        best_conf = 0.0
        for text, conf in zip(data['text'], data['conf']):
            try:
                conf_f = float(conf)
            except (ValueError, TypeError):
                continue
            if conf_f > best_conf and text.strip():
                best_conf = conf_f
                best_text = text.strip()
        return normalise_number(best_text), best_conf / 100.0
    except Exception:
        return None, 0.0


# ---------------------------------------------------------------------------
# STAGE E: EASYOCR
# ---------------------------------------------------------------------------

_easyocr_reader = None


def get_easyocr_reader():
    global _easyocr_reader
    if _easyocr_reader is None and EASYOCR_OK:
        print_step("  Initialising EasyOCR reader (first time ~5s)...")
        _easyocr_reader = easyocr.Reader(
            EASYOCR_LANGS, gpu=False, verbose=False
        )
    return _easyocr_reader


def run_easyocr(crop: np.ndarray) -> tuple[int | None, float]:
    if not EASYOCR_OK or crop is None:
        return None, 0.0
    reader = get_easyocr_reader()
    if reader is None:
        return None, 0.0
    try:
        results = reader.readtext(
            crop, detail=1, min_size=10,
            text_threshold=0.4, low_text=0.3
        )
        best_number = None
        best_conf   = 0.0
        for (_, text, conf) in results:
            number = normalise_number(text)
            if number is not None and float(conf) > best_conf:
                best_number = number
                best_conf   = float(conf)
        return best_number, best_conf
    except Exception:
        return None, 0.0


# ---------------------------------------------------------------------------
# STAGE F: FUSION
# ---------------------------------------------------------------------------

def fuse_ocr_results(tess_num, tess_conf,
                      easy_num, easy_conf) -> tuple[int | None, float]:
    """
    Fuse Tesseract and EasyOCR into a single (number, confidence).
    Both agree -> high confidence.
    One engine  -> result with penalty.
    Disagree    -> higher-weighted engine wins with extra penalty.
    """
    if tess_num is None and easy_num is None:
        return None, 0.0
    if tess_num is None:
        return easy_num, easy_conf * SINGLE_PENALTY
    if easy_num is None:
        return tess_num, tess_conf * SINGLE_PENALTY
    if tess_num == easy_num:
        return tess_num, min(
            TESS_WEIGHT * tess_conf + EASYOCR_WEIGHT * easy_conf, 1.0
        )
    # Disagreement
    if EASYOCR_WEIGHT * easy_conf >= TESS_WEIGHT * tess_conf:
        return easy_num, easy_conf * SINGLE_PENALTY * 0.8
    return tess_num, tess_conf * SINGLE_PENALTY * 0.8


# ---------------------------------------------------------------------------
# STAGE G: VALIDATION
# ---------------------------------------------------------------------------

def validate_parcel_number(number: int | None, map_num: str) -> str:
    """
    Cross-check detected number against the known range for this map.
    VALID   -> within expected range
    BORDER  -> within +-BORDER_TOLERANCE of range (shared boundary parcel)
    INVALID -> outside range (likely OCR error)
    """
    if number is None:
        return "NO_NUMBER"
    if map_num not in PARCEL_RANGE_PER_MAP:
        return "UNKNOWN"
    lo, hi = PARCEL_RANGE_PER_MAP[map_num]
    if lo <= number <= hi:
        return "VALID"
    if (lo - BORDER_TOLERANCE) <= number <= (hi + BORDER_TOLERANCE):
        return "BORDER"
    return "INVALID"


# ---------------------------------------------------------------------------
# STAGE H: SAVE OUTPUTS
# ---------------------------------------------------------------------------

def save_json(map_num: str, parcels: list[dict]):
    """
    Save parcel list as JSON for Step 4 Stage 2.
    The fields detected_number, cx, cy are what Step 4 reads.
    """
    output = []
    for p in parcels:
        output.append({
            "parcel_id":       int(p["id"]),
            "detected_number": p.get("detected_number"),
            "cx":              round(p["cx"], 2),
            "cy":              round(p["cy"], 2),
            "confidence":      round(p.get("confidence", 0.0), 4),
            "validation":      p.get("validation", "UNKNOWN"),
            "area_px":         round(p["area_px"], 1),
            "bbox":            p["bbox"],
        })
    path = DETECTED_DIR / f"map_{map_num}_parcels.json"
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print_step(f"  Saved: {path.name}")


def save_csv(map_num: str, parcels: list[dict]):
    path = DETECTED_DIR / f"map_{map_num}_ocr.csv"
    fields = ["parcel_id", "detected_number", "cx", "cy",
              "confidence", "validation", "area_px",
              "tess_number", "tess_conf", "easy_number", "easy_conf"]
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        w.writeheader()
        for p in parcels:
            w.writerow({k: p.get(k, '') for k in fields})
    print_step(f"  Saved: {path.name}")


def save_annotated(map_num: str, gray: np.ndarray, parcels: list[dict]):
    """
    Annotated image:
    Green  = VALID parcel
    Cyan   = BORDER parcel (key for Step 4 Stage 2 — straddles sheet edge)
    Red    = INVALID (OCR error suspected)
    Grey   = no number detected
    """
    annotated = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    colour_map = {
        "VALID":     (0, 200, 0),
        "BORDER":    (0, 220, 255),
        "INVALID":   (0, 0, 220),
        "NO_NUMBER": (150, 150, 150),
        "UNKNOWN":   (150, 150, 150),
    }

    for p in parcels:
        if p.get("detected_number") is None:
            continue
        colour = colour_map.get(p.get("validation", "UNKNOWN"), (150, 150, 150))
        x, y, bw, bh = p["bbox"]
        cv2.rectangle(annotated, (x, y), (x + bw, y + bh), colour, 2)
        label = str(p["detected_number"])
        cx, cy = int(p["cx"]), int(p["cy"])
        cv2.putText(annotated, label, (cx - 20, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255), 3, cv2.LINE_AA)
        cv2.putText(annotated, label, (cx - 20, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    colour, 1, cv2.LINE_AA)

    # Legend
    y_pos = 40
    for status, col in colour_map.items():
        cv2.putText(annotated, status, (20, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (255, 255, 255), 3, cv2.LINE_AA)
        cv2.putText(annotated, status, (20, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    col, 1, cv2.LINE_AA)
        y_pos += 30

    save_image(DETECTED_DIR / f"map_{map_num}_annotated.png", annotated)
    print_step(f"  Saved: map_{map_num}_annotated.png")


# ---------------------------------------------------------------------------
# MAIN: PROCESS ONE MAP
# ---------------------------------------------------------------------------

def process_map(map_num: str) -> dict:
    print(f"\n  {'='*60}")
    print(f"  Map {map_num}")
    print(f"  {'='*60}")
    t0 = time.time()

    gray, binary = load_map_images(map_num)
    print_step(f"  Loaded: {gray.shape}")

    parcels = segment_parcels(binary, map_num)
    if not parcels:
        print_step("  No parcels found — skipping OCR")
        return {"map": map_num, "n_parcels": 0,
                "n_valid": 0, "n_border": 0, "elapsed": 0.0}

    n_valid = n_border = n_invalid = n_no_number = n_ocr_ok = 0

    for p in parcels:
        crop = preprocess_for_ocr(gray, p["bbox"])

        tess_num, tess_conf = run_tesseract(crop)
        easy_num, easy_conf = run_easyocr(crop)
        fused_num, fused_conf = fuse_ocr_results(
            tess_num, tess_conf, easy_num, easy_conf
        )
        validation = validate_parcel_number(fused_num, map_num)

        p["detected_number"] = fused_num
        p["confidence"]      = fused_conf
        p["validation"]      = validation
        p["tess_number"]     = tess_num
        p["tess_conf"]       = round(tess_conf, 4)
        p["easy_number"]     = easy_num
        p["easy_conf"]       = round(easy_conf, 4)

        if fused_num is not None:
            n_ocr_ok += 1
        if   validation == "VALID":     n_valid     += 1
        elif validation == "BORDER":    n_border    += 1
        elif validation == "INVALID":   n_invalid   += 1
        else:                           n_no_number += 1

    save_json(map_num, parcels)
    save_csv(map_num, parcels)
    save_annotated(map_num, gray, parcels)

    elapsed = time.time() - t0

    border_list = [
        p["detected_number"]
        for p in parcels if p.get("validation") == "BORDER"
    ]

    print_step(f"  {len(parcels)} parcels | OCR ok={n_ocr_ok} | "
               f"VALID={n_valid} | BORDER={n_border} | "
               f"INVALID={n_invalid} | NO_NUM={n_no_number} | {elapsed:.1f}s")
    if border_list:
        print_step(f"  BORDER parcels (for Step 4 Stage 2): {border_list}")

    return {
        "map":           map_num,
        "n_parcels":     len(parcels),
        "n_ocr_success": n_ocr_ok,
        "n_valid":       n_valid,
        "n_border":      n_border,
        "n_invalid":     n_invalid,
        "n_no_number":   n_no_number,
        "border_list":   border_list,
        "elapsed":       round(elapsed, 2),
    }


# ---------------------------------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------------------------------

def run():
    print("\n" + "="*70)
    print("  STEP 5 — PARCEL DETECTION & OCR NUMBER RECOGNITION")
    print("  Cadastral Panoramic Reconstruction — Masters Thesis")
    print("="*70)

    make_dirs()

    print_step(f"Tesseract : {'YES' if TESSERACT_OK else 'NO  (pip install pytesseract)'}")
    print_step(f"EasyOCR   : {'YES' if EASYOCR_OK  else 'NO  (pip install easyocr)'}")

    if not TESSERACT_OK and not EASYOCR_OK:
        print_step("ERROR: No OCR engine available.")
        print_step("  Install at least one:")
        print_step("  pip install pytesseract   + tesseract binary")
        print_step("  pip install easyocr")
        return False

    all_results = []
    missing     = []

    for num in MAP_NUMBERS:
        if not (PREPROCESSED_DIR / f"map_{num}_clean.png").exists():
            print_step(f"WARNING: map {num} not found — skipping")
            missing.append(num)
            continue
        result = process_map(num)
        all_results.append(result)

    # ── Summary ───────────────────────────────────────────────────────────
    print("\n" + "="*70)
    print("  STEP 5 SUMMARY")
    print("="*70)

    if missing:
        print_step(f"Maps skipped : {', '.join(missing)}")

    print(f"\n  {'Map':>4} | {'Parcels':>7} | {'OCR OK':>6} | "
          f"{'Valid':>6} | {'Border':>7} | {'Invalid':>8} | "
          f"{'NoNum':>6} | {'Time':>6}")
    print("  " + "-"*72)

    total_parcels = total_valid = total_border = 0
    for r in all_results:
        total_parcels += r.get("n_parcels", 0)
        total_valid   += r.get("n_valid", 0)
        total_border  += r.get("n_border", 0)
        print(f"  {r['map']:>4} | "
              f"{r.get('n_parcels', 0):>7} | "
              f"{r.get('n_ocr_success', 0):>6} | "
              f"{r.get('n_valid', 0):>6} | "
              f"{r.get('n_border', 0):>7} | "
              f"{r.get('n_invalid', 0):>8} | "
              f"{r.get('n_no_number', 0):>6} | "
              f"{r.get('elapsed', 0):>5.1f}s")

    total_time = sum(r.get("elapsed", 0) for r in all_results)
    print(f"\n  Total parcels detected : {total_parcels}")
    print(f"  Total VALID            : {total_valid}")
    print(f"  Total BORDER           : {total_border}  <- used by Step 4 Stage 2")
    print(f"  Total time             : {total_time:.1f}s")
    print(f"\n  Output : {DETECTED_DIR.resolve()}")
    print(f"\n  Next steps:")
    print(f"    1. Re-run step4_matching.py  (Stage 2 activates automatically)")
    print(f"    2. Run step6_panorama.py     (final stitched panorama)")
    print(f"\n  [Step 5] Status: SUCCESS")
    print("="*70 + "\n")
    return True


if __name__ == "__main__":
    ok = run()
    sys.exit(0 if ok else 1)
