"""
EfficientNet-B0 ImageNet Classifier Demo
=========================================
• Load an image from disk  OR  capture from webcam
• Runs EfficientNet-B0 via ONNX Runtime
• Displays top-5 predictions with confidence bars
"""

import os
import sys
import threading
import time
import tkinter as tk
from tkinter import filedialog, ttk
import numpy as np
import cv2
import onnxruntime as ort
from PIL import Image, ImageTk

# ─── Paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH  = os.path.join(SCRIPT_DIR, "efficientnet_b0.onnx")
LABELS_PATH = os.path.join(SCRIPT_DIR, "labels.txt")

INPUT_SIZE   = (224, 224)   # EfficientNet-B0 expects 224×224
INPUT_NAME   = "image_tensor"
OUTPUT_NAME  = "class_logits"

# ─── Colour palette ────────────────────────────────────────────────────────────
BG_DARK   = "#0f1117"
BG_CARD   = "#1a1d27"
BG_CARD2  = "#22263a"
ACCENT    = "#6c63ff"
ACCENT2   = "#00d4aa"
TEXT_PRI  = "#f0f0f5"
TEXT_SEC  = "#8b8fa8"
BAR_COLORS = ["#6c63ff", "#5a9cff", "#00d4aa", "#f59e0b", "#ef4444"]


# ══════════════════════════════════════════════════════════════════════════════
#  Model helpers
# ══════════════════════════════════════════════════════════════════════════════

def load_labels(path: str) -> list[str]:
    with open(path, "r") as f:
        return [line.strip() for line in f.readlines()]


def softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - np.max(x))
    return e / e.sum()


def preprocess(img_bgr: np.ndarray) -> np.ndarray:
    """BGR uint8 → NCHW float32 [0, 1]"""
    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, INPUT_SIZE, interpolation=cv2.INTER_LINEAR)
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))          # HWC → CHW
    img = np.expand_dims(img, axis=0)            # CHW → NCHW
    return img


def run_inference(session: ort.InferenceSession, img_bgr: np.ndarray):
    tensor = preprocess(img_bgr)
    logits = session.run([OUTPUT_NAME], {INPUT_NAME: tensor})[0][0]
    probs  = softmax(logits)
    top5_idx = np.argsort(probs)[::-1][:5]
    return [(int(i), float(probs[i])) for i in top5_idx]


# ══════════════════════════════════════════════════════════════════════════════
#  Custom Widgets
# ══════════════════════════════════════════════════════════════════════════════

class ConfidenceBar(tk.Canvas):
    """Animated horizontal confidence bar."""

    def __init__(self, parent, color=ACCENT, **kwargs):
        kwargs.setdefault("height", 8)
        kwargs.setdefault("bg", BG_CARD2)
        kwargs.setdefault("highlightthickness", 0)
        super().__init__(parent, **kwargs)
        self._color = color
        self._target = 0.0
        self._current = 0.0
        self._rect = None
        self.bind("<Configure>", self._on_resize)

    def _on_resize(self, _event=None):
        self._draw(self._current)

    def _draw(self, frac: float):
        w = self.winfo_width()
        h = self.winfo_height()
        if w < 2 or h < 2:
            return
        self.delete("all")
        # Background track
        self.create_rectangle(0, 0, w, h, fill="#2e3250", outline="")
        # Filled portion
        fill_w = max(0, int(frac * w))
        if fill_w > 0:
            self.create_rectangle(0, 0, fill_w, h, fill=self._color, outline="")
        self._current = frac

    def animate_to(self, target: float, steps=20, delay=12):
        self._target = target
        step_size = (target - self._current) / max(steps, 1)

        def _tick(remaining):
            if remaining <= 0:
                self._draw(self._target)
                return
            self._current += step_size
            self._draw(max(0, min(1, self._current)))
            self.after(delay, lambda: _tick(remaining - 1))

        _tick(steps)


class PredictionRow(tk.Frame):
    """One prediction row: rank label + class name + bar + percentage."""

    def __init__(self, parent, rank: int, color: str):
        super().__init__(parent, bg=BG_CARD2)
        self._bar_color = color

        # Rank badge
        badge = tk.Label(self, text=f"#{rank}", font=("Segoe UI", 10, "bold"),
                         fg=color, bg=BG_CARD2, width=3, anchor="center")
        badge.pack(side="left", padx=(8, 4), pady=6)

        # Class name
        self._name_lbl = tk.Label(self, text="—", font=("Segoe UI", 10),
                                  fg=TEXT_PRI, bg=BG_CARD2, anchor="w",
                                  width=26)
        self._name_lbl.pack(side="left", padx=(0, 8))

        # Bar
        self._bar = ConfidenceBar(self, color=color, width=200)
        self._bar.pack(side="left", fill="x", expand=True, padx=(0, 8))

        # Percentage
        self._pct_lbl = tk.Label(self, text="—", font=("Segoe UI", 10, "bold"),
                                 fg=color, bg=BG_CARD2, width=7, anchor="e")
        self._pct_lbl.pack(side="right", padx=(0, 10))

    def update_prediction(self, label: str, prob: float):
        self._name_lbl.config(text=label[:30])
        self._pct_lbl.config(text=f"{prob*100:.2f}%")
        self._bar.animate_to(prob)

    def clear(self):
        self._name_lbl.config(text="—")
        self._pct_lbl.config(text="—")
        self._bar.animate_to(0)


# ══════════════════════════════════════════════════════════════════════════════
#  Main Application
# ══════════════════════════════════════════════════════════════════════════════

class EfficientNetApp(tk.Tk):

    def __init__(self):
        super().__init__()
        self.title("EfficientNet-B0 · ImageNet Classifier")
        self.configure(bg=BG_DARK)
        self.resizable(True, True)
        self.minsize(820, 600)

        # State
        self._session: ort.InferenceSession | None = None
        self._labels: list[str] = []
        self._cam_running = False
        self._cam_thread: threading.Thread | None = None
        self._cap: cv2.VideoCapture | None = None
        self._current_frame: np.ndarray | None = None
        self._last_infer_time = 0.0
        self._infer_interval = 0.5   # seconds between live inferences

        self._build_ui()
        self._load_model_async()

    # ── UI Construction ───────────────────────────────────────────────────────

    def _build_ui(self):
        # Header
        hdr = tk.Frame(self, bg=BG_DARK)
        hdr.pack(fill="x", pady=(18, 0), padx=20)

        tk.Label(hdr, text="⚡ EfficientNet-B0",
                 font=("Segoe UI", 22, "bold"), fg=ACCENT, bg=BG_DARK
                 ).pack(side="left")
        tk.Label(hdr, text="  ImageNet Classifier  ·  ONNX Runtime",
                 font=("Segoe UI", 11), fg=TEXT_SEC, bg=BG_DARK
                 ).pack(side="left", pady=(6, 0))

        self._status_lbl = tk.Label(hdr, text="⏳ Loading model…",
                                    font=("Segoe UI", 10), fg=ACCENT2, bg=BG_DARK)
        self._status_lbl.pack(side="right")

        sep = tk.Frame(self, height=1, bg=BG_CARD2)
        sep.pack(fill="x", padx=20, pady=10)

        # Main body
        body = tk.Frame(self, bg=BG_DARK)
        body.pack(fill="both", expand=True, padx=20, pady=(0, 20))

        # Left panel ─ image preview
        left = tk.Frame(body, bg=BG_CARD, bd=0)
        left.pack(side="left", fill="both", expand=True, padx=(0, 10))

        self._preview = tk.Label(left, bg=BG_CARD, text="No image loaded",
                                 fg=TEXT_SEC, font=("Segoe UI", 12))
        self._preview.pack(fill="both", expand=True, padx=4, pady=4)

        # Button bar below preview
        btn_bar = tk.Frame(left, bg=BG_CARD)
        btn_bar.pack(fill="x", pady=(0, 8), padx=8)

        self._btn_open = self._make_button(btn_bar, "📂  Open Image",
                                           self._open_image, ACCENT)
        self._btn_open.pack(side="left", padx=(0, 6))

        self._btn_cam = self._make_button(btn_bar, "📷  Start Webcam",
                                          self._toggle_webcam, ACCENT2)
        self._btn_cam.pack(side="left", padx=(0, 6))

        self._btn_snap = self._make_button(btn_bar, "🔍  Run Inference",
                                           self._infer_current, "#f59e0b")
        self._btn_snap.pack(side="left")
        self._btn_snap.config(state="disabled")

        # Right panel ─ results
        right = tk.Frame(body, bg=BG_CARD, width=340)
        right.pack(side="right", fill="y")
        right.pack_propagate(False)

        tk.Label(right, text="Top-5 Predictions",
                 font=("Segoe UI", 13, "bold"), fg=TEXT_PRI, bg=BG_CARD
                 ).pack(pady=(14, 6), padx=14, anchor="w")

        self._pred_rows: list[PredictionRow] = []
        for i in range(5):
            row = PredictionRow(right, rank=i+1, color=BAR_COLORS[i])
            row.pack(fill="x", padx=10, pady=4)
            self._pred_rows.append(row)

        # Inference info
        info_card = tk.Frame(right, bg=BG_CARD2)
        info_card.pack(fill="x", padx=10, pady=(10, 6))
        self._inf_time_lbl = tk.Label(info_card, text="Inference: —",
                                      font=("Segoe UI", 9), fg=TEXT_SEC,
                                      bg=BG_CARD2)
        self._inf_time_lbl.pack(anchor="w", padx=10, pady=4)

        # Mode indicator
        self._mode_lbl = tk.Label(right, text="Mode: Idle",
                                  font=("Segoe UI", 9), fg=TEXT_SEC, bg=BG_CARD)
        self._mode_lbl.pack(pady=(4, 14), padx=14, anchor="w")

    def _make_button(self, parent, text, command, color):
        btn = tk.Button(parent, text=text, command=command,
                        font=("Segoe UI", 10, "bold"),
                        fg="white", bg=color,
                        activebackground=color, activeforeground="white",
                        relief="flat", bd=0, padx=12, pady=6,
                        cursor="hand2")
        # Hover effect
        btn.bind("<Enter>", lambda e, b=btn, c=color: b.config(bg=_lighten(c)))
        btn.bind("<Leave>", lambda e, b=btn, c=color: b.config(bg=c))
        return btn

    # ── Model Loading ─────────────────────────────────────────────────────────

    def _load_model_async(self):
        def _load():
            try:
                self._labels = load_labels(LABELS_PATH)
                opts = ort.SessionOptions()
                opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                self._session = ort.InferenceSession(MODEL_PATH, sess_options=opts,
                    providers=["CPUExecutionProvider"])
                self.after(0, self._on_model_ready)
            except Exception as exc:
                self.after(0, lambda: self._status_lbl.config(
                    text=f"❌ Model load failed: {exc}", fg="#ef4444"))

        threading.Thread(target=_load, daemon=True).start()

    def _on_model_ready(self):
        self._status_lbl.config(text="✅ Model ready", fg=ACCENT2)
        self._btn_snap.config(state="normal")
        self._mode_lbl.config(text="Mode: Ready")

    # ── Image Loading ─────────────────────────────────────────────────────────

    def _open_image(self):
        path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.webp *.tiff"),
                       ("All files", "*.*")])
        if not path:
            return
        self._stop_webcam()
        img_bgr = cv2.imread(path)
        if img_bgr is None:
            self._status_lbl.config(text="❌ Could not read image", fg="#ef4444")
            return
        self._current_frame = img_bgr
        self._show_frame(img_bgr)
        self._mode_lbl.config(text="Mode: Static Image")
        self._status_lbl.config(text=f"📄 {os.path.basename(path)}", fg=TEXT_SEC)
        self._run_inference_on(img_bgr)

    # ── Webcam ────────────────────────────────────────────────────────────────

    def _toggle_webcam(self):
        if self._cam_running:
            self._stop_webcam()
        else:
            self._start_webcam()

    def _start_webcam(self):
        self._cap = cv2.VideoCapture(1)
        if not self._cap.isOpened():
            self._status_lbl.config(text="❌ No webcam found", fg="#ef4444")
            return
        self._cam_running = True
        self._btn_cam.config(text="⏹  Stop Webcam")
        self._mode_lbl.config(text="Mode: Live Webcam")
        self._status_lbl.config(text="📷 Webcam active", fg=ACCENT2)
        self._cam_thread = threading.Thread(target=self._webcam_loop, daemon=True)
        self._cam_thread.start()

    def _stop_webcam(self):
        self._cam_running = False
        if self._cap:
            self._cap.release()
            self._cap = None
        self._btn_cam.config(text="📷  Start Webcam")
        if not self._current_frame is None:
            pass  # keep last frame visible
        else:
            self._mode_lbl.config(text="Mode: Idle")

    def _webcam_loop(self):
        while self._cam_running:
            ret, frame = self._cap.read()
            if not ret:
                break
            self._current_frame = frame
            self.after(0, lambda f=frame.copy(): self._show_frame(f))
            # Periodic live inference
            now = time.time()
            if self._session and now - self._last_infer_time >= self._infer_interval:
                self._last_infer_time = now
                self.after(0, lambda f=frame.copy(): self._run_inference_on(f))
            time.sleep(0.03)  # ~30 fps display

    # ── Frame Display ─────────────────────────────────────────────────────────

    def _show_frame(self, img_bgr: np.ndarray):
        w = max(self._preview.winfo_width(), 400)
        h = max(self._preview.winfo_height(), 340)
        ih, iw = img_bgr.shape[:2]
        scale = min(w / iw, h / ih, 1.0)
        nw, nh = int(iw * scale), int(ih * scale)
        resized = cv2.resize(img_bgr, (nw, nh))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        photo = ImageTk.PhotoImage(pil)
        self._preview.config(image=photo, text="")
        self._preview._photo = photo  # prevent GC

    # ── Inference ─────────────────────────────────────────────────────────────

    def _infer_current(self):
        if self._current_frame is not None:
            self._run_inference_on(self._current_frame)

    def _run_inference_on(self, img_bgr: np.ndarray):
        if self._session is None:
            return

        def _infer():
            t0 = time.perf_counter()
            results = run_inference(self._session, img_bgr)
            elapsed = (time.perf_counter() - t0) * 1000
            self.after(0, lambda: self._update_results(results, elapsed))

        threading.Thread(target=_infer, daemon=True).start()

    def _update_results(self, results: list[tuple[int, float]], elapsed_ms: float):
        for i, row in enumerate(self._pred_rows):
            if i < len(results):
                idx, prob = results[i]
                label = self._labels[idx] if idx < len(self._labels) else f"class_{idx}"
                row.update_prediction(label, prob)
            else:
                row.clear()
        self._inf_time_lbl.config(text=f"Inference: {elapsed_ms:.1f} ms")


# ══════════════════════════════════════════════════════════════════════════════
#  Utility
# ══════════════════════════════════════════════════════════════════════════════

def _lighten(hex_color: str, amount=30) -> str:
    """Return a slightly lighter version of a hex colour."""
    hex_color = hex_color.lstrip("#")
    r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
    r = min(255, r + amount)
    g = min(255, g + amount)
    b = min(255, b + amount)
    return f"#{r:02x}{g:02x}{b:02x}"


# ══════════════════════════════════════════════════════════════════════════════
#  Entry point
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    app = EfficientNetApp()
    # Centre window on screen
    app.update_idletasks()
    W, H = 900, 620
    sw = app.winfo_screenwidth()
    sh = app.winfo_screenheight()
    app.geometry(f"{W}x{H}+{(sw-W)//2}+{(sh-H)//2}")
    app.mainloop()
