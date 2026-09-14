"""
=============================================================================
MILESTONE 5 (alt) — CLAUDE VISION OCR FOR PARCEL NUMBER RECOGNITION
=============================================================================
Masters Thesis - AI-Based Panoramic Cadastral Image Reconstruction
Author : Hussein Chalhoub

Purpose
--------
Read the 4-digit Arabic-Indic parcel numbers written inside each detected
parcel polygon by sending the cropped image to Claude's vision API.

This file is the fallback for step7_ocr_paddle.py, which proved unusable on
this machine: PaddlePaddle 3.x crashes (PIR + oneDNN) on Windows + CPU.

Why a vision LLM?
  - Robust on handwritten / aged / rotated numerals where classical OCR fails
  - No model files to download, no native code that crashes
  - Output is constrained: we ask for "the 4-digit number, in Western digits,
    or NONE" — no parsing of free-form text

Pipeline position
------------------
  step4 (Mask R-CNN)  -> per-map parcel bboxes + centroids
       |
  step7 (this file)   -> per-parcel OCR'd number (via Claude Vision)
       |
  step8               -> cross-reference with CSV, build control points
       |
  step9               -> homography + parcel-contour refinement

Setup
------
  pip install anthropic
  $env:ANTHROPIC_API_KEY = "sk-ant-..."

Cost (Haiku 4.5 is the default; very cheap for this task)
  ~ $0.0005 per parcel crop → ~ $5-7 for all 11,400 parcels.

Usage
------
  # 1) Cheap diagnostic: send only the boundary parcels (CSV ground truth) of
  #    one pair through the API. About 20 calls, < $0.05, finishes in seconds.
  python new_pipeline/src/step7_ocr_vision.py --test-pair 45_47

  # 2) Full run on a single map (uses async batches for speed, resumable).
  python new_pipeline/src/step7_ocr_vision.py --map 45

  # 3) All 13 maps.
  python new_pipeline/src/step7_ocr_vision.py --all

  # Optional: pick a model
  python new_pipeline/src/step7_ocr_vision.py --map 45 --model sonnet

Resume
-------
Per-parcel responses are cached to:
  new_pipeline/data/parcel_numbers/cache/map_<N>.jsonl
A re-run skips parcels that already have a cached response. Delete the cache
file to force a re-run for that map.

Output (same as step7_ocr_paddle.py)
-------------------------------------
  new_pipeline/data/parcel_numbers/
    map_<N>_numbers.json     - every parcel's recognised number (or null)
    map_<N>_overlay.png      - visualisation: bright green = CSV hit

=============================================================================
"""

import argparse
import base64
import io
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------

PREPROCESSED_DIR = Path("output/preprocessed")
PREDICTIONS_DIR  = Path("new_pipeline/data/predictions")
OUTPUT_DIR       = Path("new_pipeline/data/parcel_numbers")
# Cache and per-map JSON filenames are namespaced by --source ("clean" or
# "binary") so re-running on a different source doesn't trash earlier
# results. Set by argparse at the bottom of this file.
SOURCE_KIND      = "clean"
CACHE_DIR        = OUTPUT_DIR / "cache" / SOURCE_KIND


def _set_source(kind: str):
    """Switch the source preprocessed image type and namespace outputs."""
    global SOURCE_KIND, CACHE_DIR
    if kind not in ("clean", "binary"):
        raise ValueError(f"--source must be 'clean' or 'binary', got {kind!r}")
    SOURCE_KIND = kind
    CACHE_DIR = OUTPUT_DIR / "cache" / SOURCE_KIND


def _image_path(map_num: str) -> Path:
    return PREPROCESSED_DIR / f"map_{map_num}_{SOURCE_KIND}.png"


def _numbers_path(map_num: str) -> Path:
    return OUTPUT_DIR / f"map_{map_num}_numbers_{SOURCE_KIND}.json"

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

MAP_BOUNDARY_PARCELS = {}
for parcel_num, maps in BOUNDARY_PARCELS_CSV.items():
    for m in maps:
        MAP_BOUNDARY_PARCELS.setdefault(m, []).append(parcel_num)

# Crop margin around each parcel bbox (px). The number sits inside the parcel;
# a small margin guarantees the strokes are not clipped at the bbox edge.
CROP_MARGIN_PX = 25

# Skip parcels whose bbox is too small or too big to plausibly hold a 4-digit
# parcel number. These bounds are deliberately loose.
MIN_CROP_SIDE_PX = 50
MAX_CROP_SIDE_PX = 1500

# To keep API payloads small, downscale each crop so its longest side is at
# most this many pixels before sending. Bumped from 600 to 1024 because the
# handwritten 40-80 px digits get blurry below that and Haiku/Sonnet need
# enough pixels to disambiguate Arabic-Indic numerals.
MAX_API_SIDE_PX = 1024

# Pre-rotate each crop by this many degrees before sending. The Lebanese
# cadastral sheets in this project are scanned upside-down: setting 180 makes
# all handwritten text upright before OCR. Override per-run with --rotation.
DEFAULT_ROTATION = 180

# Concurrent API requests. Stay well below tier-1 rate limits.
DEFAULT_CONCURRENCY = 6

# Model selection
MODEL_BY_KEY = {
    "haiku":  "claude-haiku-4-5-20251001",   # default, cheap, fast
    "sonnet": "claude-sonnet-4-6",
    "opus":   "claude-opus-4-7",
}

SYSTEM_PROMPT = (
    "You are an OCR assistant for old Lebanese cadastral blueprint maps.\n"
    "I will show you an image cropped from one such map. Inside that crop\n"
    "there is normally a 4-digit parcel number written by hand in\n"
    "Arabic-Indic numerals (٠١٢٣٤٥٦٧٨٩):\n"
    "  ٠=0  ١=1  ٢=2  ٣=3  ٤=4  ٥=5  ٦=6  ٧=7  ٨=8  ٩=9\n"
    "\n"
    "The number is between 2000 and 3999 (rarely up to 4999).\n"
    "Numbers are READ FROM LEFT TO RIGHT (the first digit is the leftmost,\n"
    "the last digit is the rightmost), the same order as Western numerals.\n"
    "\n"
    "The text may be rotated 0, 90, 180, or 270 degrees, and may be\n"
    "partially faded, smudged, or written by hand in a non-uniform style.\n"
    "Edges of neighbouring parcels' numbers may bleed into the crop —\n"
    "focus on the most central, clearly written number.\n"
    "\n"
    "Be willing to make a best-effort read. If 3 of the 4 digits are\n"
    "clear and the 4th is ambiguous, give your best guess.\n"
    "Only answer NONE if the crop genuinely contains no parcel number\n"
    "(e.g. roads, blank ground, or only neighbouring parcels' digits).\n"
    "\n"
    "Reply with EXACTLY one line:\n"
    "  - the 4-digit number in Western digits (e.g. \"2580\"), OR\n"
    "  - \"NONE\" only if no number is plausibly readable.\n"
    "\n"
    "No other text. No prefix. No explanation."
)


# ---------------------------------------------------------------------------
# UTILITIES
# ---------------------------------------------------------------------------

def read_image(path: Path, color: bool = False) -> np.ndarray:
    """Unicode-safe image read (Windows OneDrive paths)."""
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


def crop_for_parcel(img: np.ndarray, parcel: dict) -> np.ndarray | None:
    """Tight bbox + small margin. Returns None for implausible crops."""
    H, W = img.shape[:2]
    x, y, w, h = parcel["bbox"]
    x0 = max(0, x - CROP_MARGIN_PX)
    y0 = max(0, y - CROP_MARGIN_PX)
    x1 = min(W, x + w + CROP_MARGIN_PX)
    y1 = min(H, y + h + CROP_MARGIN_PX)
    cw, ch = x1 - x0, y1 - y0
    if cw < MIN_CROP_SIDE_PX or ch < MIN_CROP_SIDE_PX:
        return None
    if cw > MAX_CROP_SIDE_PX or ch > MAX_CROP_SIDE_PX:
        return None
    return img[y0:y1, x0:x1]


def encode_crop_for_api(crop: np.ndarray, rotation: int = 0) -> str:
    """
    Downscale (if needed), pre-rotate, and base64-encode as PNG for Anthropic.

    rotation: 0, 90, 180, or 270. The crop is rotated counter-clockwise by
    this many degrees before encoding. We use 180 for the Lebanese sheets
    that are scanned upside-down so the digits become upright.
    """
    if crop.ndim == 2:
        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR)
    else:
        crop_rgb = crop

    if rotation == 90:
        crop_rgb = cv2.rotate(crop_rgb, cv2.ROTATE_90_COUNTERCLOCKWISE)
    elif rotation == 180:
        crop_rgb = cv2.rotate(crop_rgb, cv2.ROTATE_180)
    elif rotation == 270:
        crop_rgb = cv2.rotate(crop_rgb, cv2.ROTATE_90_CLOCKWISE)
    elif rotation != 0:
        raise ValueError(f"rotation must be 0/90/180/270, got {rotation}")

    h, w = crop_rgb.shape[:2]
    scale = min(1.0, MAX_API_SIDE_PX / max(h, w))
    if scale < 1.0:
        crop_rgb = cv2.resize(
            crop_rgb,
            (int(w * scale), int(h * scale)),
            interpolation=cv2.INTER_AREA,
        )

    ok, buf = cv2.imencode(".png", crop_rgb)
    if not ok:
        raise RuntimeError("PNG encode failed")
    return base64.standard_b64encode(buf.tobytes()).decode("ascii")


def parse_model_response(text: str) -> int | None:
    """
    The model is instructed to return either '2580' or 'NONE'. Be defensive:
    extract any 4-digit number in [2000, 9999] from the response.
    """
    if not text:
        return None
    if text.strip().upper().startswith("NONE"):
        return None
    m = re.search(r"\b([2-9]\d{3})\b", text)
    if not m:
        return None
    n = int(m.group(1))
    if 2000 <= n <= 9999:
        return n
    return None


# ---------------------------------------------------------------------------
# ANTHROPIC CLIENT
# ---------------------------------------------------------------------------

_client = None


def verify_setup():
    """
    Check anthropic SDK + API key on the MAIN thread before any work starts.
    Prints one clear error and exits the process. Must be called from the
    main thread — sys.exit() from a worker just kills that thread.
    """
    try:
        import anthropic   # noqa: F401
    except ImportError:
        print("\n  ERROR: anthropic SDK not installed.")
        print("  Install with:")
        print("    pip install anthropic")
        sys.exit(1)
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("\n  ERROR: ANTHROPIC_API_KEY not set.")
        print('  In PowerShell:  $env:ANTHROPIC_API_KEY = "sk-ant-..."')
        print("  Then re-run the script in the SAME PowerShell window.")
        sys.exit(1)


def get_client():
    """
    Return a (cached) Anthropic client. Assumes verify_setup() has already
    been called on the main thread, so the import + key check have passed.
    """
    global _client
    if _client is not None:
        return _client
    import anthropic   # safe: verify_setup() proved it imports
    _client = anthropic.Anthropic()
    return _client


def call_claude_for_crop(crop: np.ndarray, model: str,
                          rotation: int = 0,
                          max_retries: int = 3) -> dict:
    """
    Send one crop to Claude and parse the answer.
    Returns dict: {number: int|None, raw: str, error: str|None,
                   input_tokens, output_tokens}
    """
    client = get_client()
    b64 = encode_crop_for_api(crop, rotation=rotation)

    last_err = None
    for attempt in range(max_retries):
        try:
            resp = client.messages.create(
                model=model,
                max_tokens=20,
                system=[{
                    "type": "text",
                    "text": SYSTEM_PROMPT,
                    "cache_control": {"type": "ephemeral"},
                }],
                messages=[{
                    "role": "user",
                    "content": [{
                        "type": "image",
                        "source": {
                            "type":       "base64",
                            "media_type": "image/png",
                            "data":        b64,
                        },
                    }],
                }],
            )
            text = ""
            if resp.content:
                first = resp.content[0]
                text  = getattr(first, "text", "") or ""
            return {
                "number":         parse_model_response(text),
                "raw":            text.strip(),
                "error":          None,
                "input_tokens":   getattr(resp.usage, "input_tokens", 0),
                "output_tokens":  getattr(resp.usage, "output_tokens", 0),
            }
        except Exception as e:
            last_err = e
            # Exponential back-off on rate-limit / transient errors
            time.sleep(min(8.0, 1.0 * (2 ** attempt)))

    return {"number": None, "raw": "",
            "error": f"{type(last_err).__name__}: {last_err}",
            "input_tokens": 0, "output_tokens": 0}


# ---------------------------------------------------------------------------
# CACHE (resume support)
# ---------------------------------------------------------------------------

def load_cache(map_num: str) -> dict[int, dict]:
    """JSONL cache: one record per parcel_id."""
    cache_path = CACHE_DIR / f"map_{map_num}.jsonl"
    if not cache_path.exists():
        return {}
    out = {}
    with open(cache_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "parcel_id" in rec:
                out[int(rec["parcel_id"])] = rec
    return out


def append_cache(map_num: str, record: dict):
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = CACHE_DIR / f"map_{map_num}.jsonl"
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# PER-MAP DETECTION
# ---------------------------------------------------------------------------

def detect_numbers_on_map(map_num: str,
                          model_key: str = "haiku",
                          parcels_filter: list[int] | None = None,
                          concurrency: int = DEFAULT_CONCURRENCY,
                          rotation: int = DEFAULT_ROTATION,
                          force: bool = False,
                          ) -> list[dict]:
    """
    Read every (or selected) parcel of `map_num` through Claude vision.

    `parcels_filter`: if given, only run on parcels with these IDs (used
    by --test-pair to restrict to a small handful of crops).
    """
    if model_key not in MODEL_BY_KEY:
        raise ValueError(f"Unknown model key: {model_key}. "
                         f"Choose from: {list(MODEL_BY_KEY)}")
    model = MODEL_BY_KEY[model_key]

    img_path  = _image_path(map_num)
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

    if parcels_filter is not None:
        wanted = set(int(p) for p in parcels_filter)
        parcels = [p for p in parcels if int(p["parcel_id"]) in wanted]

    cache = {} if force else load_cache(map_num)

    # Pre-build the work list: (parcel, crop) pairs that need an API call
    to_call: list[tuple[dict, np.ndarray]] = []
    skipped_no_crop  = 0
    skipped_cached   = 0
    cached_results: list[dict] = []

    for p in parcels:
        pid = int(p["parcel_id"])
        if pid in cache:
            cached = cache[pid]
            cached_rotation = cached.get("rotation", 0)
            # Re-call if:
            # - the cached record was an error (e.g. account had no credit), OR
            # - the cached rotation differs from the current run
            is_error = bool(cached.get("error")) and (
                cached.get("number") is None
                and cached.get("raw") not in ("SKIP_CROP", "NONE")
            )
            rotation_mismatch = (
                cached.get("raw") != "SKIP_CROP"
                and cached_rotation != rotation
            )
            if not is_error and not rotation_mismatch:
                skipped_cached += 1
                if cached.get("number") is not None:
                    cached_results.append({
                        "parcel_id":  pid,
                        "cx":         float(p["cx"]),
                        "cy":         float(p["cy"]),
                        "bbox":       p["bbox"],
                        "recognised_number": int(cached["number"]),
                        "confidence": 1.0,
                        "raw_text":   cached.get("raw", ""),
                        "model":      cached.get("model", model),
                        "rotation":   cached_rotation,
                    })
                continue
            # else: fall through to re-call this parcel
        crop = crop_for_parcel(img, p)
        if crop is None:
            skipped_no_crop += 1
            # Cache the skip so we don't recompute on resume
            append_cache(map_num, {
                "parcel_id": pid,
                "number":    None,
                "raw":       "SKIP_CROP",
                "model":     model,
            })
            continue
        to_call.append((p, crop))

    print(f"  Map {map_num}: {len(parcels)} parcels"
          f"  ({skipped_cached} cached, {skipped_no_crop} no-crop, "
          f"{len(to_call)} to call)  rotation={rotation}deg"
          f"{'  [FORCED]' if force else ''}")

    detections: list[dict] = list(cached_results)
    total_in_tokens  = 0
    total_out_tokens = 0

    if not to_call:
        return detections

    target_set = set(MAP_BOUNDARY_PARCELS.get(map_num, []))

    pbar = tqdm(total=len(to_call), desc=f"  Map {map_num} -> Claude",
                leave=False)

    def _work(item):
        p, crop = item
        return p, call_claude_for_crop(crop, model, rotation=rotation)

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [pool.submit(_work, item) for item in to_call]
        for fut in as_completed(futures):
            try:
                p, result = fut.result()
            except Exception as e:
                pbar.update(1)
                pbar.set_postfix_str(f"err {type(e).__name__}")
                continue

            pid = int(p["parcel_id"])
            total_in_tokens  += result.get("input_tokens",  0)
            total_out_tokens += result.get("output_tokens", 0)

            # Persist to cache regardless of outcome
            append_cache(map_num, {
                "parcel_id": pid,
                "number":    result["number"],
                "raw":       result["raw"],
                "error":     result.get("error"),
                "model":     model,
                "rotation":  rotation,
            })

            if result["number"] is not None:
                num = int(result["number"])
                detections.append({
                    "parcel_id":         pid,
                    "cx":                float(p["cx"]),
                    "cy":                float(p["cy"]),
                    "bbox":              p["bbox"],
                    "recognised_number": num,
                    "confidence":        1.0,
                    "raw_text":          result["raw"],
                    "model":             model,
                    "rotation":          rotation,
                })
                if num in target_set:
                    pbar.set_postfix_str(f"hit {num}")

            pbar.update(1)
    pbar.close()

    # Annotate every detection with whether it's a CSV target hit (for the
    # overlay PNG)
    for d in detections:
        d["is_csv_target"] = d["recognised_number"] in target_set

    # Rough cost estimate (Haiku 4.5 published price as of writing)
    if model_key == "haiku":
        cost = (total_in_tokens / 1e6) * 1.0 + (total_out_tokens / 1e6) * 5.0
        print(f"  Tokens: {total_in_tokens} in, {total_out_tokens} out  "
              f"(~${cost:.3f} on Haiku 4.5)")
    else:
        print(f"  Tokens: {total_in_tokens} in, {total_out_tokens} out")

    return detections


# ---------------------------------------------------------------------------
# OUTPUT / VISUALISATION
# ---------------------------------------------------------------------------

def save_detections(map_num: str, detections: list[dict]) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = _numbers_path(map_num)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(detections, f, indent=2, ensure_ascii=False)
    return out_path


def visualise_detections(map_num: str,
                          detections: list[dict],
                          out_path: Path,
                          target_set: set[int] | None = None):
    """Render detected numbers on top of the original map (down-scaled)."""
    img_path = _image_path(map_num)
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
            colour = (0, 255, 0)
            n_targets += 1
            cv2.circle(img, (cx, cy), 14, colour, -1)
        else:
            colour = (0, 165, 255)
            cv2.circle(img, (cx, cy), 6, colour, -1)
        cv2.putText(
            img, str(det["recognised_number"]),
            (cx + 12, cy + 6),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, colour, 2, cv2.LINE_AA,
        )

    cv2.rectangle(img, (0, 0), (img.shape[1], 50), (30, 30, 30), -1)
    header = (
        f"Map {map_num} | {len(detections)} numbers found | "
        f"{n_targets} CSV hits  (Claude Vision)"
    )
    cv2.putText(img, header, (10, 33),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (255, 255, 255), 2, cv2.LINE_AA)

    save_image(out_path, img)


# ---------------------------------------------------------------------------
# RUN MODES
# ---------------------------------------------------------------------------

def parcel_ids_near_boundary_csv_targets(map_num: str,
                                          target_numbers: list[int]
                                          ) -> list[int]:
    """
    For --test-pair: we don't know which parcel_id holds which number, so we
    just send EVERY parcel of the map through Claude. That is how the real
    pipeline would work anyway. The target_numbers list is used only for
    reporting (which CSV ground-truth numbers were hit on this map).
    """
    pred_path = PREDICTIONS_DIR / f"map_{map_num}_parcels.json"
    with open(pred_path, "r", encoding="utf-8") as f:
        parcels = json.load(f)
    return [int(p["parcel_id"]) for p in parcels]


def run_test_pair(pair_key: str, model_key: str = "haiku",
                  concurrency: int = DEFAULT_CONCURRENCY,
                  rotation: int = DEFAULT_ROTATION,
                  force: bool = False):
    """
    Diagnostic: OCR every parcel of both maps, then check whether the CSV
    ground-truth boundary parcels were correctly identified on both sides.
    """
    a, b = pair_key.split("_")
    expected = [n for n, maps in BOUNDARY_PARCELS_CSV.items()
                if a in maps and b in maps]
    if not expected:
        print(f"  No CSV boundary parcels expected for pair {pair_key}.")
        return

    target_set = set(expected)
    print(f"\n  Test pair {pair_key}: expecting {expected} on both maps.")
    print(f"  Model: {MODEL_BY_KEY[model_key]}  rotation: {rotation}deg")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    print(f"\n  --- Map {a} ---")
    det_a = detect_numbers_on_map(a, model_key=model_key,
                                   concurrency=concurrency,
                                   rotation=rotation, force=force)
    save_detections(a, det_a)
    visualise_detections(a, det_a,
                         OUTPUT_DIR / f"map_{a}_overlay.png", target_set)

    print(f"\n  --- Map {b} ---")
    det_b = detect_numbers_on_map(b, model_key=model_key,
                                   concurrency=concurrency,
                                   rotation=rotation, force=force)
    save_detections(b, det_b)
    visualise_detections(b, det_b,
                         OUTPUT_DIR / f"map_{b}_overlay.png", target_set)

    found_a = {d["recognised_number"] for d in det_a if d["is_csv_target"]}
    found_b = {d["recognised_number"] for d in det_b if d["is_csv_target"]}
    confirmed = found_a & found_b

    print("\n" + "=" * 70)
    print(f"  PAIR {pair_key} OCR RESULT  ({time.time() - t0:.1f}s)")
    print("=" * 70)
    print(f"  Expected boundary parcels    : {expected}")
    print(f"  Found on map {a:>2}               : {sorted(found_a) or '[]'}")
    print(f"  Found on map {b:>2}               : {sorted(found_b) or '[]'}")
    print(f"  Confirmed on BOTH (control points): {sorted(confirmed) or '[]'}")
    print(f"  Total numbers read on map {a}: {len(det_a)}")
    print(f"  Total numbers read on map {b}: {len(det_b)}")
    print(f"\n  Visualisations:")
    print(f"    {OUTPUT_DIR / f'map_{a}_overlay.png'}")
    print(f"    {OUTPUT_DIR / f'map_{b}_overlay.png'}")
    print("=" * 70)

    if confirmed:
        print("\n  Vision OCR works on this pair — proceed with --all.")
    else:
        print("\n  Vision OCR did not match any CSV ground-truth pair.")
        print("  Inspect the overlay PNGs above to see what was read.")


def run_one_map(map_num: str, model_key: str = "haiku",
                 concurrency: int = DEFAULT_CONCURRENCY,
                 rotation: int = DEFAULT_ROTATION,
                 force: bool = False):
    target_set = set(MAP_BOUNDARY_PARCELS.get(map_num, []))
    print(f"\n  Map {map_num}  | Model: {MODEL_BY_KEY[model_key]}  "
          f"rotation: {rotation}deg")
    if target_set:
        print(f"  CSV expects boundary parcels: {sorted(target_set)}")
    t0 = time.time()
    detections = detect_numbers_on_map(
        map_num, model_key=model_key, concurrency=concurrency,
        rotation=rotation, force=force,
    )
    save_detections(map_num, detections)
    visualise_detections(
        map_num, detections,
        OUTPUT_DIR / f"map_{map_num}_overlay.png",
        target_set,
    )
    hits = sum(1 for d in detections if d["is_csv_target"])
    print(f"\n  Map {map_num}: {len(detections)} numbers read, "
          f"{hits} CSV ground-truth hits  ({time.time() - t0:.1f}s)")


def run_all(model_key: str = "haiku",
            concurrency: int = DEFAULT_CONCURRENCY,
            rotation: int = DEFAULT_ROTATION,
            force: bool = False):
    print("\n" + "=" * 70)
    print(f"  STEP 7 — CLAUDE VISION OCR ON ALL 13 MAPS")
    print(f"  Model: {MODEL_BY_KEY[model_key]}  rotation: {rotation}deg")
    print("=" * 70)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    summary = {}
    for n in range(43, 56):
        m = str(n)
        if not (PREDICTIONS_DIR / f"map_{m}_parcels.json").exists():
            print(f"  Skipping map {m} (no Mask R-CNN predictions)")
            continue
        run_one_map(m, model_key=model_key, concurrency=concurrency,
                    rotation=rotation, force=force)
        with open(_numbers_path(m), "r", encoding="utf-8") as f:
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
# DIAGNOSTIC: read the cache file and summarise what came back from Claude
# ---------------------------------------------------------------------------

def report_cache(map_num: str):
    """
    Summarise what's in `cache/map_<N>.jsonl`:
      - total records
      - how many parsed to a number
      - how many were Claude saying "NONE"
      - how many were errors
      - how many were SKIP_CROP (bbox too small/large)
    Also lists the actual numbers read so we can sanity-check them.
    """
    cache_path = CACHE_DIR / f"map_{map_num}.jsonl"
    if not cache_path.exists():
        print(f"  No cache for map {map_num}: {cache_path}")
        return

    n_total = n_got = n_none = n_err = n_skip = 0
    numbers: list[int] = []
    raws_for_none: list[str] = []
    sample_errors: list[str] = []

    with open(cache_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            n_total += 1
            if rec.get("error"):
                n_err += 1
                if len(sample_errors) < 3:
                    sample_errors.append(rec["error"][:140])
                continue
            raw = rec.get("raw", "")
            if raw == "SKIP_CROP":
                n_skip += 1
                continue
            if rec.get("number") is not None:
                n_got += 1
                numbers.append(int(rec["number"]))
            else:
                n_none += 1
                if raw and raw != "NONE":
                    raws_for_none.append(raw)

    print(f"\n  Map {map_num} cache report ({cache_path.name})")
    print(f"  ------------------------------------------------")
    print(f"  total records       : {n_total}")
    print(f"  read a number       : {n_got}")
    print(f"  Claude said NONE    : {n_none}")
    print(f"  errors              : {n_err}")
    print(f"  skipped (no crop)   : {n_skip}")

    expected = set(MAP_BOUNDARY_PARCELS.get(map_num, []))
    if expected:
        hits = sorted(set(numbers) & expected)
        misses = sorted(expected - set(numbers))
        print(f"\n  CSV expects on this map  : {sorted(expected)}")
        print(f"  CSV found via Claude     : {hits}")
        print(f"  CSV missed by Claude     : {misses}")

    if numbers:
        # Show all numbers read (sorted), highlight CSV hits
        unique_nums = sorted(set(numbers))
        print(f"\n  Distinct numbers read ({len(unique_nums)}):")
        line = []
        for n in unique_nums:
            tag = "*" if n in expected else " "
            line.append(f"{tag}{n}")
        # 8 numbers per line
        for i in range(0, len(line), 8):
            print("    " + "  ".join(line[i:i + 8]))
        print("    (* = matches a CSV ground-truth boundary parcel)")

    if raws_for_none:
        print(f"\n  Sample non-empty 'NONE'-class raw texts (first 5):")
        for r in raws_for_none[:5]:
            print(f"    {r!r}")

    if sample_errors:
        print(f"\n  Sample errors (first 3):")
        for e in sample_errors:
            print(f"    {e}")
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Claude Vision parcel-number recognition"
    )
    parser.add_argument("--test-pair", type=str, default=None,
                        help="One pair only, e.g. --test-pair 45_47")
    parser.add_argument("--map", type=str, default=None,
                        help="A single map number, e.g. --map 45")
    parser.add_argument("--all", action="store_true",
                        help="All 13 maps")
    parser.add_argument("--model", type=str, default="haiku",
                        choices=list(MODEL_BY_KEY),
                        help="haiku (default, cheap), sonnet, or opus")
    parser.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY,
                        help="Concurrent API requests (default 6)")
    parser.add_argument("--rotation", type=int, default=DEFAULT_ROTATION,
                        choices=[0, 90, 180, 270],
                        help=f"Rotate each crop by this many degrees before "
                             f"sending to the model. Default {DEFAULT_ROTATION} "
                             f"(matches the upside-down scans).")
    parser.add_argument("--force", action="store_true",
                        help="Ignore cached results and re-call every parcel "
                             "(useful after changing rotation or prompt).")
    parser.add_argument("--report", type=str, default=None,
                        help="Diagnostic: summarise cache for one map "
                             "(reads only, no API calls). e.g. --report 45")
    parser.add_argument("--source", type=str, default="clean",
                        choices=["clean", "binary"],
                        help="Which preprocessed image to OCR. 'clean' = "
                             "grayscale (default); 'binary' = thresholded. "
                             "Each source has its own cache + output JSON, "
                             "so you can A/B-test without losing results.")
    args = parser.parse_args()

    _set_source(args.source)

    if args.report:
        report_cache(args.report)
        sys.exit(0)

    if args.test_pair or args.map or args.all:
        verify_setup()

    if args.test_pair:
        run_test_pair(args.test_pair, args.model, args.concurrency,
                      args.rotation, args.force)
    elif args.map:
        run_one_map(args.map, args.model, args.concurrency,
                    args.rotation, args.force)
    elif args.all:
        run_all(args.model, args.concurrency, args.rotation, args.force)
    else:
        print("Usage:")
        print("  python new_pipeline/src/step7_ocr_vision.py --test-pair 45_47")
        print("  python new_pipeline/src/step7_ocr_vision.py --map 45")
        print("  python new_pipeline/src/step7_ocr_vision.py --all")
        print("  --model haiku|sonnet|opus  (default haiku)")
        print("  --concurrency N            (default 6)")
