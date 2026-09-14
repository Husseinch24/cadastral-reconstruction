"""
=============================================================================
CADASTRAL PANORAMA BUILDER — GUI
=============================================================================
Run from the thesis root directory:
    python new_pipeline/src/panorama_ui.py

Select the maps you want, click GENERATE PANORAMA.
Output is always seamlessly blended (no visible seams).
=============================================================================
"""

import os
import sys
import threading
import tkinter as tk
from tkinter import filedialog, messagebox
from pathlib import Path

# Set working directory to thesis root so relative paths resolve correctly
os.chdir(Path(__file__).parent.parent.parent)
sys.path.insert(0, str(Path(__file__).parent))

from PIL import Image, ImageTk
import cv2

from panorama_all import run as run_panorama, MAPS, OUT_PATH
from edge_orientation_finder import PREPROCESSED_DIR, auto_crop, read_image

# ---------------------------------------------------------------------------
# Visual constants
# ---------------------------------------------------------------------------
THUMB_W  = 130
THUMB_H  = 165
COLS     = 4
BG       = "#141414"
CARD_OFF = "#1e1e1e"
CARD_ON  = "#17301f"
FG       = "#d8d8d8"
FG_DIM   = "#777777"
ACCENT   = "#46d46a"
HL_ON    = "#46d46a"
HL_OFF   = "#3a3a3a"
BTN_BG   = "#1a3d26"
BTN_FG   = "#46d46a"
BTN_ACT  = "#235233"
LOG_BG   = "#0d0d0d"
LOG_FG   = "#999999"


# ---------------------------------------------------------------------------
# stdout redirect → tkinter Text widget
# ---------------------------------------------------------------------------
class _LogRedirect:
    def __init__(self, callback):
        self._cb = callback
    def write(self, text):
        if text:
            self._cb(text)
    def flush(self):
        pass


# ---------------------------------------------------------------------------
# Main application
# ---------------------------------------------------------------------------
class PanoramaUI(tk.Tk):

    def __init__(self):
        super().__init__()
        self.title("Cadastral Panorama Builder")
        self.configure(bg=BG)
        self.minsize(680, 620)
        self.resizable(True, True)

        self._vars   = {}   # map_num -> BooleanVar
        self._cards  = {}   # map_num -> tk.Frame
        self._thumbs = {}   # keep PhotoImage references alive
        self._all_var = tk.BooleanVar(value=True)
        self._running = False

        self._build_ui()
        self._load_thumbnails()
        self.mainloop()

    # ------------------------------------------------------------------ #
    # UI construction                                                       #
    # ------------------------------------------------------------------ #
    def _build_ui(self):
        # ── title ──────────────────────────────────────────────────────
        tk.Label(self,
                 text="Cadastral Panorama Builder",
                 bg=BG, fg=ACCENT,
                 font=("Helvetica", 15, "bold"),
                 pady=12).pack(fill="x")
        tk.Frame(self, bg="#2e2e2e", height=1).pack(fill="x")

        # ── top bar: Select All + count ────────────────────────────────
        top = tk.Frame(self, bg=BG, padx=16, pady=8)
        top.pack(fill="x")

        tk.Checkbutton(
            top,
            text="Select All",
            variable=self._all_var,
            bg=BG, fg=FG,
            activebackground=BG, activeforeground=ACCENT,
            selectcolor="#2a2a2a",
            font=("Helvetica", 10, "bold"),
            command=self._toggle_all,
        ).pack(side="left")

        self._lbl_count = tk.Label(top, text="", bg=BG, fg=FG_DIM,
                                   font=("Helvetica", 9))
        self._lbl_count.pack(side="left", padx=14)

        # ── map grid ───────────────────────────────────────────────────
        grid_wrap = tk.Frame(self, bg=BG, padx=14, pady=6)
        grid_wrap.pack(fill="both", expand=True)

        for col in range(COLS):
            grid_wrap.columnconfigure(col, weight=1)

        for i, m in enumerate(MAPS):
            var = tk.BooleanVar(value=True)
            var.trace_add("write", lambda *_: self._update_count())
            self._vars[m] = var

            card = tk.Frame(
                grid_wrap, bg=CARD_ON,
                highlightthickness=2,
                highlightbackground=HL_ON,
                padx=5, pady=6,
                cursor="hand2",
            )
            card.grid(row=i // COLS, column=i % COLS,
                      padx=5, pady=5, sticky="nsew")
            self._cards[m] = card

            # thumbnail label (filled in later)
            img_lbl = tk.Label(card, bg=CARD_ON,
                               width=THUMB_W, height=THUMB_H)
            img_lbl.pack()
            card._img_lbl = img_lbl

            # map number
            tk.Label(card, text=f"Map {m}", bg=CARD_ON, fg=FG,
                     font=("Helvetica", 9, "bold")).pack(pady=(2, 0))

            # checkbox
            cb = tk.Checkbutton(
                card, variable=var,
                bg=CARD_ON, activebackground=CARD_ON,
                selectcolor="#2a2a2a",
                command=lambda mm=m: self._on_card_toggle(mm),
            )
            cb.pack()
            card._cb = cb

            # clicking anywhere on the card toggles the checkbox
            for widget in (card, img_lbl):
                widget.bind("<Button-1>", lambda e, mm=m: self._click_card(mm))

        # ── separator ──────────────────────────────────────────────────
        tk.Frame(self, bg="#2e2e2e", height=1).pack(fill="x", padx=14, pady=(6, 0))

        # ── output path row ────────────────────────────────────────────
        out_row = tk.Frame(self, bg=BG, padx=16, pady=6)
        out_row.pack(fill="x")

        tk.Label(out_row, text="Output file:", bg=BG, fg=FG_DIM,
                 font=("Helvetica", 9)).pack(side="left")

        self._out_var = tk.StringVar(value=str(OUT_PATH))
        tk.Entry(
            out_row,
            textvariable=self._out_var,
            bg="#222", fg=FG, insertbackground=FG,
            relief="flat", font=("Courier", 8), width=52,
        ).pack(side="left", padx=8)

        tk.Button(
            out_row, text="Browse…",
            bg="#252525", fg=FG, activebackground="#333",
            relief="flat", font=("Helvetica", 8),
            command=self._browse,
        ).pack(side="left")

        # ── generate button ────────────────────────────────────────────
        btn_frame = tk.Frame(self, bg=BG, pady=10)
        btn_frame.pack()

        self._gen_btn = tk.Button(
            btn_frame,
            text="GENERATE PANORAMA",
            bg=BTN_BG, fg=BTN_FG,
            activebackground=BTN_ACT, activeforeground=BTN_FG,
            font=("Helvetica", 12, "bold"),
            relief="flat", padx=30, pady=10,
            command=self._generate,
        )
        self._gen_btn.pack()

        # ── status + log ───────────────────────────────────────────────
        bot = tk.Frame(self, bg=BG, padx=16, pady=4)
        bot.pack(fill="both")

        self._status = tk.Label(bot, text="Ready — select maps and click Generate",
                                bg=BG, fg=FG_DIM,
                                font=("Helvetica", 9), anchor="w")
        self._status.pack(fill="x")

        self._log = tk.Text(
            bot, height=5,
            bg=LOG_BG, fg=LOG_FG,
            font=("Courier", 8),
            relief="flat", state="disabled",
        )
        self._log.pack(fill="x", pady=(2, 8))

        self._update_count()

    # ------------------------------------------------------------------ #
    # Thumbnail loading                                                     #
    # ------------------------------------------------------------------ #
    def _load_thumbnails(self):
        for m in MAPS:
            path = PREPROCESSED_DIR / f"map_{m}_clean.png"
            if not path.exists():
                continue
            try:
                gray  = auto_crop(read_image(path))
                h, w  = gray.shape
                scale = min(THUMB_W / w, THUMB_H / h)
                nw, nh = int(w * scale), int(h * scale)
                gray  = cv2.resize(gray, (nw, nh), interpolation=cv2.INTER_AREA)

                # center on a fixed-size canvas
                canvas = Image.new("L", (THUMB_W, THUMB_H), 230)
                canvas.paste(Image.fromarray(gray),
                             ((THUMB_W - nw) // 2, (THUMB_H - nh) // 2))

                photo = ImageTk.PhotoImage(canvas)
                self._thumbs[m] = photo
                self._cards[m]._img_lbl.configure(image=photo,
                                                   width=THUMB_W, height=THUMB_H)
            except Exception:
                pass

    # ------------------------------------------------------------------ #
    # Interactions                                                          #
    # ------------------------------------------------------------------ #
    def _click_card(self, m):
        self._vars[m].set(not self._vars[m].get())
        self._on_card_toggle(m)

    def _toggle_all(self):
        v = self._all_var.get()
        for var in self._vars.values():
            var.set(v)
        for m in MAPS:
            self._style_card(m)

    def _on_card_toggle(self, m):
        self._style_card(m)
        n_sel = sum(v.get() for v in self._vars.values())
        self._all_var.set(n_sel == len(MAPS))

    def _style_card(self, m):
        on  = self._vars[m].get()
        bg  = CARD_ON  if on else CARD_OFF
        hl  = HL_ON    if on else HL_OFF
        card = self._cards[m]
        card.configure(bg=bg, highlightbackground=hl)
        for w in card.winfo_children():
            try:
                w.configure(bg=bg)
            except tk.TclError:
                pass

    def _update_count(self):
        n = sum(v.get() for v in self._vars.values())
        self._lbl_count.configure(text=f"{n} / {len(MAPS)} maps selected")

    def _browse(self):
        path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG image", "*.png")],
            initialfile=Path(self._out_var.get()).name,
        )
        if path:
            self._out_var.set(path)

    # ------------------------------------------------------------------ #
    # Panorama generation                                                   #
    # ------------------------------------------------------------------ #
    def _generate(self):
        if self._running:
            return

        selected = [m for m, v in self._vars.items() if v.get()]
        if len(selected) < 2:
            messagebox.showwarning("Too few maps",
                                   "Select at least 2 maps to build a panorama.")
            return

        exclude  = [m for m in MAPS if m not in selected]
        out_path = Path(self._out_var.get())

        self._running = True
        self._gen_btn.configure(state="disabled", text="Generating…")
        self._log_clear()
        self._set_status(f"Building panorama from {len(selected)} maps…")

        def worker():
            old_stdout = sys.stdout
            try:
                sys.stdout = _LogRedirect(
                    lambda t: self.after(0, self._append_log, t)
                )
                run_panorama(exclude=exclude, out_path=out_path)
                sys.stdout = old_stdout
                self.after(0, self._done_ok, str(out_path))
            except Exception as exc:
                sys.stdout = old_stdout
                self.after(0, self._done_err, str(exc))

        threading.Thread(target=worker, daemon=True).start()

    def _done_ok(self, path):
        self._running = False
        self._gen_btn.configure(state="normal", text="GENERATE PANORAMA")
        self._set_status(f"Saved → {path}")

    def _done_err(self, msg):
        self._running = False
        self._gen_btn.configure(state="normal", text="GENERATE PANORAMA")
        self._set_status(f"Error: {msg}")
        messagebox.showerror("Generation failed", msg)

    # ------------------------------------------------------------------ #
    # Log helpers                                                           #
    # ------------------------------------------------------------------ #
    def _set_status(self, text):
        self._status.configure(text=text)

    def _append_log(self, text):
        self._log.configure(state="normal")
        self._log.insert("end", text)
        self._log.see("end")
        self._log.configure(state="disabled")

    def _log_clear(self):
        self._log.configure(state="normal")
        self._log.delete("1.0", "end")
        self._log.configure(state="disabled")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    PanoramaUI()
