"""
=============================================================================
MILESTONE 6 — INTERACTIVE PARCEL-PAIR CLICK TOOL
=============================================================================
Masters Thesis - AI-Based Panoramic Cadastral Image Reconstruction
Author : Hussein Chalhoub

Purpose
--------
Pop up a two-pane viewer of two adjacent map sheets (e.g. 45 and 47).
You click corresponding parcels: first on Map A, then on Map B. The tool
snaps each click to the nearest Mask R-CNN parcel centroid, highlights
the polygon, and records (parcel_a_id, parcel_b_id) pairs.

This replaces (for now) automatic OCR-based parcel identification. The
clicks are ONLY parcel identity hints — the actual stitching homography
will be derived automatically by SIFT correspondences computed inside
the matched polygons (step 9), not from where you click.

Why click instead of OCR?
  - OCR on handwritten Arabic-Indic on aged blueprints is unreliable.
  - We have ~1-5 boundary parcels per pair; clicking is a one-time
    operation of a few seconds per pair.
  - The thesis pipeline still uses SIFT + segmentation + homography
    automatically; clicks are just the supervised "which parcel = which
    parcel" labels that OCR was supposed to provide.

Controls
---------
  Left click on Map A panel  : select a parcel on A
  Left click on Map B panel  : pair it with the previously-selected A
  Press 'u'                  : undo (cancel pending A pick, or remove
                                last completed pair)
  Press 's'                  : save and exit
  Press 'q'                  : quit without saving

Usage
------
  python new_pipeline/src/step8_click_pairs.py --pair 45_47

Output
-------
  new_pipeline/data/control_points/pair_<A>_<B>_clicks.json

  {
    "map_a": "45",
    "map_b": "47",
    "pairs": [
      {
        "pair_index": 0,
        "parcel_a": 234,            // parcel_id from step4 prediction
        "parcel_b": 478,
        "cx_a": 5234.0, "cy_a": 3120.0,
        "cx_b": 1067.0, "cy_b": 4580.0,
        "bbox_a": [...], "bbox_b": [...],
        "polygon_a": [[...]], "polygon_b": [[...]]
      }
    ]
  }

The next step (step9_sift_match.py) reads this file and computes the
SIFT correspondences inside each matched polygon.
=============================================================================
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import matplotlib
matplotlib.use("TkAgg")   # interactive backend; required for click events
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon


# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------

PREDICTIONS_DIR  = Path("new_pipeline/data/predictions")
PREPROCESSED_DIR = Path("output/preprocessed")
OUTPUT_DIR       = Path("new_pipeline/data/control_points")

# Downscale each displayed map so total UI fits without lagging matplotlib.
DISPLAY_MAX_PX = 2000

# Cycle of bright colours used to label successive pairs.
PAIR_COLOURS = [
    "#00ff66", "#ff3030", "#3399ff", "#ffaa00",
    "#ff00ff", "#00ffff", "#ffff00", "#ff5500",
    "#aa00ff", "#00aa44",
]


# ---------------------------------------------------------------------------
# IO HELPERS
# ---------------------------------------------------------------------------

def load_color(map_num: str) -> np.ndarray:
    p = PREPROCESSED_DIR / f"map_{map_num}_clean.png"
    data = np.fromfile(str(p), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Cannot read map {map_num}: {p}")
    return img


def load_parcels(map_num: str) -> list[dict]:
    p = PREDICTIONS_DIR / f"map_{map_num}_parcels.json"
    if not p.exists():
        raise FileNotFoundError(
            f"Parcel JSON missing: {p}\n"
            f"Run step4_segmentation.py --infer --maps {map_num} first."
        )
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def downscale_for_display(img: np.ndarray) -> tuple[np.ndarray, float]:
    h, w = img.shape[:2]
    scale = min(1.0, DISPLAY_MAX_PX / max(h, w))
    if scale < 1.0:
        img = cv2.resize(img, (int(w * scale), int(h * scale)),
                          interpolation=cv2.INTER_AREA)
    return img, scale


def find_nearest_parcel(parcels: list[dict],
                         x_native: float, y_native: float) -> dict | None:
    if not parcels:
        return None
    best = None
    best_d = float("inf")
    for p in parcels:
        d = (p["cx"] - x_native) ** 2 + (p["cy"] - y_native) ** 2
        if d < best_d:
            best_d = d
            best = p
    return best


# ---------------------------------------------------------------------------
# CLICK TOOL
# ---------------------------------------------------------------------------

class ClickTool:
    def __init__(self, map_a: str, map_b: str):
        self.map_a = map_a
        self.map_b = map_b
        self.parcels_a = load_parcels(map_a)
        self.parcels_b = load_parcels(map_b)

        img_a_native = load_color(map_a)
        img_b_native = load_color(map_b)
        img_a_disp, self.scale_a = downscale_for_display(img_a_native)
        img_b_disp, self.scale_b = downscale_for_display(img_b_native)
        # matplotlib expects RGB
        self.img_a_disp = cv2.cvtColor(img_a_disp, cv2.COLOR_BGR2RGB)
        self.img_b_disp = cv2.cvtColor(img_b_disp, cv2.COLOR_BGR2RGB)

        self.fig, (self.ax_a, self.ax_b) = plt.subplots(
            1, 2, figsize=(18, 9)
        )
        self.fig.canvas.manager.set_window_title(
            f"Click matching parcels: {map_a} <-> {map_b}"
        )
        self._draw_base()

        self.pending_a: dict | None = None        # parcel selected on A awaiting B
        self.pairs: list[dict] = []
        self.fig.canvas.mpl_connect("button_press_event", self._on_click)
        self.fig.canvas.mpl_connect("key_press_event",     self._on_key)

        self.status = self.fig.text(
            0.5, 0.02, "", ha="center", va="bottom", fontsize=11,
            family="monospace",
        )
        self._update_status()

    # ── drawing ─────────────────────────────────────────────────────────
    def _draw_base(self):
        self.ax_a.clear()
        self.ax_b.clear()
        self.ax_a.imshow(self.img_a_disp)
        self.ax_b.imshow(self.img_b_disp)
        self.ax_a.set_title(f"Map {self.map_a}  (click 1st)")
        self.ax_b.set_title(f"Map {self.map_b}  (click 2nd)")
        for ax in (self.ax_a, self.ax_b):
            ax.set_xticks([])
            ax.set_yticks([])

    def _highlight_parcel(self, ax, parcel: dict, scale: float, colour: str):
        poly = parcel.get("polygon") or []
        if len(poly) >= 3:
            disp_poly = [(p[0] * scale, p[1] * scale) for p in poly]
            patch = MplPolygon(
                disp_poly, closed=True, fill=True,
                facecolor=colour, edgecolor=colour,
                linewidth=2, alpha=0.35,
            )
            ax.add_patch(patch)
        else:
            x, y, w, h = parcel["bbox"]
            patch = mpatches.Rectangle(
                (x * scale, y * scale), w * scale, h * scale,
                fill=True, facecolor=colour, edgecolor=colour,
                linewidth=2, alpha=0.35,
            )
            ax.add_patch(patch)
        ax.plot(parcel["cx"] * scale, parcel["cy"] * scale,
                marker="o", color=colour, markersize=8,
                markeredgecolor="black", markeredgewidth=1)

    def _label_parcel(self, ax, parcel: dict, scale: float,
                      colour: str, idx: int):
        ax.text(
            parcel["cx"] * scale + 14, parcel["cy"] * scale,
            str(idx + 1),
            color=colour, fontsize=14, fontweight="bold",
            path_effects=None,
            bbox=dict(boxstyle="round,pad=0.15",
                      facecolor="white", edgecolor=colour, alpha=0.85),
        )

    def _redraw_all(self):
        self._draw_base()
        for i, pair in enumerate(self.pairs):
            colour = PAIR_COLOURS[i % len(PAIR_COLOURS)]
            pa = self._lookup(self.parcels_a, pair["parcel_a"])
            pb = self._lookup(self.parcels_b, pair["parcel_b"])
            if pa is not None:
                self._highlight_parcel(self.ax_a, pa, self.scale_a, colour)
                self._label_parcel(self.ax_a, pa, self.scale_a, colour, i)
            if pb is not None:
                self._highlight_parcel(self.ax_b, pb, self.scale_b, colour)
                self._label_parcel(self.ax_b, pb, self.scale_b, colour, i)
        if self.pending_a is not None:
            colour = PAIR_COLOURS[len(self.pairs) % len(PAIR_COLOURS)]
            self._highlight_parcel(self.ax_a, self.pending_a,
                                    self.scale_a, colour)
        self.fig.canvas.draw_idle()

    def _update_status(self):
        if self.pending_a is None:
            msg = (f"  Pairs collected: {len(self.pairs)}.  "
                   f"Click a parcel on Map {self.map_a}.   "
                   f"[u]=undo  [s]=save&exit  [q]=quit  ")
        else:
            msg = (f"  Pending: parcel {self.pending_a['parcel_id']} on "
                   f"Map {self.map_a}.  Now click matching parcel on "
                   f"Map {self.map_b}.   [u]=cancel  [q]=quit  ")
        self.status.set_text(msg)
        self.fig.canvas.draw_idle()

    # ── helpers ─────────────────────────────────────────────────────────
    @staticmethod
    def _lookup(parcels: list[dict], pid: int) -> dict | None:
        for p in parcels:
            if int(p["parcel_id"]) == int(pid):
                return p
        return None

    # ── event handlers ──────────────────────────────────────────────────
    def _on_click(self, event):
        if event.xdata is None or event.ydata is None:
            return
        if event.button != 1:
            return

        if event.inaxes is self.ax_a:
            x_native = event.xdata / self.scale_a
            y_native = event.ydata / self.scale_a
            picked = find_nearest_parcel(self.parcels_a, x_native, y_native)
            if picked is None:
                return
            self.pending_a = picked
            self._redraw_all()
            self._update_status()

        elif event.inaxes is self.ax_b:
            if self.pending_a is None:
                return
            x_native = event.xdata / self.scale_b
            y_native = event.ydata / self.scale_b
            picked = find_nearest_parcel(self.parcels_b, x_native, y_native)
            if picked is None:
                return
            self.pairs.append({
                "pair_index": len(self.pairs),
                "parcel_a":   int(self.pending_a["parcel_id"]),
                "parcel_b":   int(picked["parcel_id"]),
                "cx_a":       float(self.pending_a["cx"]),
                "cy_a":       float(self.pending_a["cy"]),
                "cx_b":       float(picked["cx"]),
                "cy_b":       float(picked["cy"]),
                "bbox_a":     self.pending_a["bbox"],
                "bbox_b":     picked["bbox"],
                "polygon_a":  self.pending_a.get("polygon", []),
                "polygon_b":  picked.get("polygon", []),
            })
            self.pending_a = None
            self._redraw_all()
            self._update_status()

    def _on_key(self, event):
        key = (event.key or "").lower()
        if key == "u":
            if self.pending_a is not None:
                self.pending_a = None
            elif self.pairs:
                self.pairs.pop()
            self._redraw_all()
            self._update_status()
        elif key == "s":
            self._save()
            plt.close(self.fig)
        elif key == "q":
            print(f"\n  Quit without saving "
                  f"({len(self.pairs)} pair(s) discarded).")
            plt.close(self.fig)

    # ── save ────────────────────────────────────────────────────────────
    def _save(self):
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        out = {
            "map_a": self.map_a,
            "map_b": self.map_b,
            "pairs": self.pairs,
        }
        out_path = (
            OUTPUT_DIR / f"pair_{self.map_a}_{self.map_b}_clicks.json"
        )
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)
        print(f"\n  Saved {len(self.pairs)} pair(s) to {out_path}")
        print(f"  Next: python new_pipeline/src/step9_sift_match.py "
              f"--pair {self.map_a}_{self.map_b}")

    # ── main ────────────────────────────────────────────────────────────
    def run(self):
        plt.tight_layout(rect=(0, 0.04, 1, 1))
        plt.show()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Interactive click tool for matching parcel pairs."
    )
    parser.add_argument("--pair", type=str, required=True,
                        help="Adjacent pair, e.g. --pair 45_47")
    args = parser.parse_args()

    a, b = args.pair.split("_")
    tool = ClickTool(a, b)
    tool.run()
