"""
EdgeTAM (Lightweight SAM2) Interactive Segmentation Demo
========================================================
• Load an image from disk  OR  capture from webcam
• Click on the image to add positive (Left Click) or negative (Right Click) prompts
• Runs EdgeTAM via ONNX Runtime for instant segmentation
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
ENCODER_PATH = os.path.join(SCRIPT_DIR, "edgetam", "encoder.onnx")
DECODER_PATH = os.path.join(SCRIPT_DIR, "edgetam", "decoder.onnx")

# ─── Colour palette ────────────────────────────────────────────────────────────
BG_DARK   = "#0f1117"
BG_CARD   = "#1a1d27"
BG_CARD2  = "#22263a"
ACCENT    = "#6c63ff"
ACCENT2   = "#00d4aa"
TEXT_PRI  = "#f0f0f5"
TEXT_SEC  = "#8b8fa8"

# ══════════════════════════════════════════════════════════════════════════════
#  Model Logic (EdgeTAM / SAM2)
# ══════════════════════════════════════════════════════════════════════════════

class EdgeTAMModel:
    def __init__(self, encoder_path, decoder_path):
        providers = ['CPUExecutionProvider']
        self.encoder_session = ort.InferenceSession(encoder_path, providers=providers)
        self.decoder_session = ort.InferenceSession(decoder_path, providers=providers)
        
        self.input_tensor = None
        self.scale = 1.0
        self.new_h = 0
        self.new_w = 0
        self.img_h = 0
        self.img_w = 0

    def prepare_image(self, image):
        """Preprocesses the image for model inference."""
        self.img_h, self.img_w = image.shape[:2]
        self.scale = 1024.0 / max(self.img_h, self.img_w)
        self.new_h = int(self.img_h * self.scale + 0.5)
        self.new_w = int(self.img_w * self.scale + 0.5)
        
        rgb_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        resized_img = cv2.resize(rgb_img, (self.new_w, self.new_h), interpolation=cv2.INTER_LINEAR)
        
        padded_img = np.zeros((1024, 1024, 3), dtype=np.float32)
        padded_img[:self.new_h, :self.new_w, :] = resized_img.astype(np.float32) / 255.0
        chw_img = np.transpose(padded_img, (2, 0, 1))
        self.input_tensor = np.expand_dims(chw_img, axis=0)

    def predict(self, points):
        """
        points: list of (x, y, label)
        label: 1.0 (pos), 0.0 (neg)
        """
        if not points:
            return None, 0.0

        mapped_coords = []
        mapped_labels = []
        
        # Take up to last 2 points (as per existing demo logic)
        for pt in points[-2:]:
            mapped_coords.append([float(pt[0] * self.scale), float(pt[1] * self.scale)])
            mapped_labels.append(pt[2])
            
        while len(mapped_coords) < 2:
            mapped_coords.append([0.0, 0.0])
            mapped_labels.append(-1.0)
            
        unnorm_coords = np.array([mapped_coords], dtype=np.float32)
        labels = np.array([mapped_labels], dtype=np.float32)
        
        # 1. Encoder
        encoder_outputs = self.encoder_session.run(None, {
            "image": self.input_tensor,
            "unnorm_coords": unnorm_coords,
            "labels": labels
        })
        img_embeds, hr_feat1, hr_feat2, sparse_embed = encoder_outputs
        
        # 2. Decoder
        decoder_outputs = self.decoder_session.run(None, {
            "image_embeddings": img_embeds,
            "high_res_features1": hr_feat1,
            "high_res_features2": hr_feat2,
            "sparse_embedding": sparse_embed
        })
        masks, scores = decoder_outputs
        
        mask = masks[0, 0]  # (256, 256)
        score = float(scores[0, 0])
        
        # Postprocess
        binary_mask = (mask < 0.0).astype(np.uint8)
        full_padded_mask = cv2.resize(binary_mask, (1024, 1024), interpolation=cv2.INTER_NEAREST)
        valid_mask = full_padded_mask[:self.new_h, :self.new_w]
        full_mask = cv2.resize(valid_mask, (self.img_w, self.img_h), interpolation=cv2.INTER_NEAREST)
        
        return full_mask, score

# ══════════════════════════════════════════════════════════════════════════════
#  Main Application
# ══════════════════════════════════════════════════════════════════════════════

class EdgeTAMApp(tk.Tk):

    def __init__(self):
        super().__init__()
        self.title("EdgeTAM · SAM2 Interactive Segmentation")
        self.configure(bg=BG_DARK)
        self.resizable(True, True)
        self.minsize(1000, 750)

        # State
        self._model: EdgeTAMModel | None = None
        self._cam_running = False
        self._cap: cv2.VideoCapture | None = None
        self._current_frame: np.ndarray | None = None
        self._orig_image: np.ndarray | None = None
        self._points = [] # list of (x, y, label)
        self._mask = None
        self._score = 0.0
        self._is_webcam_mode = False

        self._build_ui()
        self._load_model_async()

    def _build_ui(self):
        # Header
        hdr = tk.Frame(self, bg=BG_DARK)
        hdr.pack(fill="x", pady=(18, 0), padx=20)

        tk.Label(hdr, text="🎨 EdgeTAM (SAM2)",
                 font=("Segoe UI", 22, "bold"), fg=ACCENT, bg=BG_DARK
                 ).pack(side="left")
        tk.Label(hdr, text="  Interactive Segmentation  ·  Real-time",
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

        # Left panel ─ interactive preview
        left = tk.Frame(body, bg=BG_CARD, bd=0)
        left.pack(side="left", fill="both", expand=True, padx=(0, 10))

        self._canvas = tk.Label(left, bg=BG_CARD, text="No image loaded",
                                fg=TEXT_SEC, font=("Segoe UI", 12), cursor="crosshair")
        self._canvas.pack(fill="both", expand=True, padx=4, pady=4)
        
        self._canvas.bind("<Button-1>", self._on_left_click)
        self._canvas.bind("<Button-3>", self._on_right_click)

        # Controls below preview
        ctrl_bar = tk.Frame(left, bg=BG_CARD)
        ctrl_bar.pack(fill="x", pady=(0, 8), padx=8)

        self._btn_open = self._make_button(ctrl_bar, "📂  Open Image", self._open_image, ACCENT)
        self._btn_open.pack(side="left", padx=(0, 6))

        self._btn_cam = self._make_button(ctrl_bar, "📷  Webcam Capture", self._toggle_webcam, ACCENT2)
        self._btn_cam.pack(side="left", padx=(0, 6))

        self._btn_clear = self._make_button(ctrl_bar, "🧹  Clear Points", self._clear_points, "#ef4444")
        self._btn_clear.pack(side="left", padx=(0, 6))

        # Right panel ─ info
        right = tk.Frame(body, bg=BG_CARD, width=280)
        right.pack(side="right", fill="y")
        right.pack_propagate(False)

        tk.Label(right, text="Instructions",
                 font=("Segoe UI", 13, "bold"), fg=TEXT_PRI, bg=BG_CARD
                 ).pack(pady=(14, 6), padx=14, anchor="w")

        instr = (
            "• Left Click: Positive prompt\n"
            "  (Include object)\n\n"
            "• Right Click: Negative prompt\n"
            "  (Exclude area)\n\n"
            "• Clear: Reset all points\n\n"
            "Tip: Click multiple times to\n"
            "refine the selection."
        )
        tk.Label(right, text=instr, font=("Segoe UI", 10), fg=TEXT_SEC, 
                 bg=BG_CARD, justify="left", anchor="nw").pack(padx=14, pady=5, fill="x")

        sep2 = tk.Frame(right, height=1, bg=BG_CARD2)
        sep2.pack(fill="x", padx=10, pady=10)

        tk.Label(right, text="Performance",
                 font=("Segoe UI", 11, "bold"), fg=TEXT_PRI, bg=BG_CARD
                 ).pack(padx=14, anchor="w")

        self._inf_time_lbl = tk.Label(right, text="Inference: —",
                                      font=("Segoe UI", 9), fg=TEXT_SEC, bg=BG_CARD)
        self._inf_time_lbl.pack(anchor="w", padx=14, pady=2)

        self._score_lbl = tk.Label(right, text="Conf Score: —",
                                   font=("Segoe UI", 9), fg=TEXT_SEC, bg=BG_CARD)
        self._score_lbl.pack(anchor="w", padx=14, pady=2)

        self._mode_lbl = tk.Label(right, text="Mode: Idle",
                                  font=("Segoe UI", 9), fg=TEXT_SEC, bg=BG_CARD)
        self._mode_lbl.pack(side="bottom", pady=14, padx=14, anchor="w")

    def _make_button(self, parent, text, command, color):
        btn = tk.Button(parent, text=text, command=command,
                        font=("Segoe UI", 10, "bold"),
                        fg="white", bg=color,
                        activebackground=color, activeforeground="white",
                        relief="flat", bd=0, padx=12, pady=6,
                        cursor="hand2")
        btn.bind("<Enter>", lambda e, b=btn, c=color: b.config(bg=_lighten(c)))
        btn.bind("<Leave>", lambda e, b=btn, c=color: b.config(bg=c))
        return btn

    # ── Model Loading ─────────────────────────────────────────────────────────

    def _load_model_async(self):
        def _load():
            try:
                self._model = EdgeTAMModel(ENCODER_PATH, DECODER_PATH)
                self.after(0, self._on_model_ready)
            except Exception as exc:
                self.after(0, lambda: self._status_lbl.config(
                    text=f"❌ Model load failed: {exc}", fg="#ef4444"))

        threading.Thread(target=_load, daemon=True).start()

    def _on_model_ready(self):
        self._status_lbl.config(text="✅ Model ready", fg=ACCENT2)
        self._mode_lbl.config(text="Mode: Ready")

    # ── Image Loading ─────────────────────────────────────────────────────────

    def _open_image(self):
        path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.webp"), ("All files", "*.*")])
        if not path: return
        self._is_webcam_mode = False
        self._stop_webcam()
        img_bgr = cv2.imread(path)
        if img_bgr is None: return
        self._set_new_image(img_bgr)
        self._status_lbl.config(text=f"📄 {os.path.basename(path)}", fg=TEXT_SEC)
        self._mode_lbl.config(text="Mode: Static Image")

    def _set_new_image(self, img_bgr):
        self._orig_image = img_bgr.copy()
        self._current_frame = img_bgr.copy()
        self._points = []
        self._mask = None
        self._score = 0.0
        self._model.prepare_image(img_bgr)
        self._update_display()

    # ── Webcam ────────────────────────────────────────────────────────────────

    def _toggle_webcam(self):
        if self._cam_running:
            self._stop_webcam()
        else:
            self._start_webcam()

    def _start_webcam(self):
        self._cap = cv2.VideoCapture(1)
        if not self._cap.isOpened(): self._cap = cv2.VideoCapture(0)
        if not self._cap.isOpened(): return
        
        self._cam_running = True
        self._is_webcam_mode = True
        self._btn_cam.config(text="⏹  Freeze Frame")
        self._mode_lbl.config(text="Mode: Live Webcam")
        
        def loop():
            while self._cam_running:
                ret, frame = self._cap.read()
                if not ret: break
                self._current_frame = frame
                self.after(0, lambda f=frame.copy(): self._show_frame(f))
                time.sleep(0.03)
        
        threading.Thread(target=loop, daemon=True).start()

    def _stop_webcam(self):
        self._cam_running = False
        if self._cap:
            self._cap.release()
            self._cap = None
        self._btn_cam.config(text="📷  Webcam Capture")
        if self._is_webcam_mode and self._current_frame is not None:
            # "Freeze" the last frame for interaction
            self._set_new_image(self._current_frame)
            self._mode_lbl.config(text="Mode: Captured Frame")

    # ── Interaction ───────────────────────────────────────────────────────────

    def _on_left_click(self, event):
        self._add_point(event.x, event.y, 1.0)

    def _on_right_click(self, event):
        self._add_point(event.x, event.y, 0.0)

    def _add_point(self, vx, vy, label):
        if self._orig_image is None or self._cam_running: return
        
        # Map view coordinates to image coordinates
        cw = self._canvas.winfo_width()
        ch = self._canvas.winfo_height()
        ih, iw = self._orig_image.shape[:2]
        
        scale = min(cw / iw, ch / ih, 1.0)
        nw, nh = int(iw * scale), int(ih * scale)
        
        # Offset for centering
        dx = (cw - nw) // 2
        dy = (ch - nh) // 2
        
        ix = int((vx - dx) / scale)
        iy = int((vy - dy) / scale)
        
        if 0 <= ix < iw and 0 <= iy < ih:
            self._points.append((ix, iy, label))
            # Keep up to 2 points for this specific EdgeTAM implementation
            if len(self._points) > 2:
                self._points.pop(0)
            self._run_inference()

    def _clear_points(self):
        self._points = []
        self._mask = None
        self._score = 0.0
        self._update_display()

    # ── Inference ─────────────────────────────────────────────────────────────

    def _run_inference(self):
        if not self._model or self._orig_image is None or not self._points:
            return

        def task():
            t0 = time.perf_counter()
            mask, score = self._model.predict(self._points)
            elapsed = (time.perf_counter() - t0) * 1000
            self.after(0, lambda: self._on_inference_done(mask, score, elapsed))

        threading.Thread(target=task, daemon=True).start()

    def _on_inference_done(self, mask, score, elapsed_ms):
        self._mask = mask
        self._score = score
        self._inf_time_lbl.config(text=f"Inference: {elapsed_ms:.1f} ms")
        self._score_lbl.config(text=f"Conf Score: {score:.3f}")
        self._update_display()

    # ── Display ───────────────────────────────────────────────────────────────

    def _update_display(self):
        if self._orig_image is None: return
        
        display_img = self._orig_image.copy()
        
        # Apply mask
        if self._mask is not None:
            color = np.array([255, 144, 30], dtype=np.uint8) # Elegant Blue
            mask_bool = self._mask == 1
            display_img[mask_bool] = display_img[mask_bool] * 0.4 + color * 0.6
            
            contours, _ = cv2.findContours(self._mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(display_img, contours, -1, (255, 255, 255), 2)

        # Draw points
        for pt in self._points:
            ix, iy, label = pt
            c = (0, 220, 0) if label == 1.0 else (0, 0, 220)
            cv2.circle(display_img, (ix, iy), 8, (255, 255, 255), -1)
            cv2.circle(display_img, (ix, iy), 6, c, -1)

        self._show_frame(display_img)

    def _show_frame(self, img_bgr):
        cw = max(self._canvas.winfo_width(), 400)
        ch = max(self._canvas.winfo_height(), 340)
        ih, iw = img_bgr.shape[:2]
        scale = min(cw / iw, ch / ih, 1.0)
        nw, nh = int(iw * scale), int(ih * scale)
        
        resized = cv2.resize(img_bgr, (nw, nh))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        photo = ImageTk.PhotoImage(Image.fromarray(rgb))
        self._canvas.config(image=photo, text="")
        self._canvas._photo = photo

# ══════════════════════════════════════════════════════════════════════════════
#  Utility
# ══════════════════════════════════════════════════════════════════════════════

def _lighten(hex_color: str, amount=30) -> str:
    hex_color = hex_color.lstrip("#")
    r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
    r = min(255, r + amount); g = min(255, g + amount); b = min(255, b + amount)
    return f"#{r:02x}{g:02x}{b:02x}"

if __name__ == "__main__":
    app = EdgeTAMApp()
    app.update_idletasks()
    W, H = 1000, 750
    sw = app.winfo_screenwidth()
    sh = app.winfo_screenheight()
    app.geometry(f"{W}x{H}+{(sw-W)//2}+{(sh-H)//2}")
    app.mainloop()
