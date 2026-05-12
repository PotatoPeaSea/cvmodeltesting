"""
Conditional-DETR Object Detection Demo
======================================
• Load an image from disk  OR  capture from webcam
• Runs Conditional-DETR-ResNet50 via ONNX Runtime
• Displays detected objects with bounding boxes and labels
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
MODEL_PATH  = os.path.join(SCRIPT_DIR, "conditional-detr", "conditional_detr_resnet50.onnx")

INPUT_SIZE   = (640, 640)
INPUT_NAME   = "image"
OUTPUT_NAMES = ["boxes", "logits", "classes"]

# ─── COCO Labels (80 Classes) ──────────────────────────────────────────────────
COCO_CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
    'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
    'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
    'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
    'hair drier', 'toothbrush'
]

# ─── Colour palette ────────────────────────────────────────────────────────────
BG_DARK   = "#0f1117"
BG_CARD   = "#1a1d27"
BG_CARD2  = "#22263a"
ACCENT    = "#6c63ff"
ACCENT2   = "#00d4aa"
TEXT_PRI  = "#f0f0f5"
TEXT_SEC  = "#8b8fa8"

# ══════════════════════════════════════════════════════════════════════════════
#  Model helpers
# ══════════════════════════════════════════════════════════════════════════════

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def preprocess(img_bgr: np.ndarray) -> np.ndarray:
    """BGR uint8 → NCHW float32 [0, 1]"""
    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, INPUT_SIZE, interpolation=cv2.INTER_LINEAR)
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))          # HWC → CHW
    img = np.expand_dims(img, axis=0)            # CHW → NCHW
    return img

def run_inference(session: ort.InferenceSession, img_bgr: np.ndarray, threshold=0.5):
    tensor = preprocess(img_bgr)
    outputs = session.run(None, {INPUT_NAME: tensor})
    
    # Based on metadata: 
    # boxes: [1, 300, 4]
    # logits: [1, 300]
    # classes: [1, 300]
    
    out_boxes = outputs[0][0]   # [300, 4]
    out_logits = outputs[1][0]  # [300]
    out_classes = outputs[2][0] # [300]
    
    # The 'logits' might be raw logits or scores. 
    # Usually if it's [1, 300] it's already a confidence score.
    # If values are > 1, it might be raw logits needing sigmoid.
    scores = out_logits
    if np.max(scores) > 1.0 or np.min(scores) < 0.0:
        scores = sigmoid(out_logits)
        
    detections = []
    for i in range(len(scores)):
        score = float(scores[i])
        if score < threshold:
            continue
            
        class_id = int(out_classes[i])
        box = out_boxes[i] # cx, cy, w, h normalized
        
        detections.append({
            'class_id': class_id,
            'score': score,
            'box': box # [cx, cy, w, h]
        })
        
    return detections

# ══════════════════════════════════════════════════════════════════════════════
#  Main Application
# ══════════════════════════════════════════════════════════════════════════════

class ConditionalDetrApp(tk.Tk):

    def __init__(self):
        super().__init__()
        self.title("Conditional-DETR · Object Detection")
        self.configure(bg=BG_DARK)
        self.resizable(True, True)
        self.minsize(900, 700)

        # State
        self._session: ort.InferenceSession | None = None
        self._cam_running = False
        self._cam_thread: threading.Thread | None = None
        self._cap: cv2.VideoCapture | None = None
        self._current_frame: np.ndarray | None = None
        self._last_infer_time = 0.0
        self._infer_interval = 0.3   # seconds between live inferences
        self._threshold = 0.5
        self._detections = []

        self._build_ui()
        self._load_model_async()

    def _build_ui(self):
        # Header
        hdr = tk.Frame(self, bg=BG_DARK)
        hdr.pack(fill="x", pady=(18, 0), padx=20)

        tk.Label(hdr, text="🎯 Conditional-DETR",
                 font=("Segoe UI", 22, "bold"), fg=ACCENT, bg=BG_DARK
                 ).pack(side="left")
        tk.Label(hdr, text="  Object Detection  ·  ResNet50",
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

        # Controls below preview
        ctrl_bar = tk.Frame(left, bg=BG_CARD)
        ctrl_bar.pack(fill="x", pady=(0, 8), padx=8)

        self._btn_open = self._make_button(ctrl_bar, "📂  Open Image",
                                           self._open_image, ACCENT)
        self._btn_open.pack(side="left", padx=(0, 6))

        self._btn_cam = self._make_button(ctrl_bar, "📷  Start Webcam",
                                          self._toggle_webcam, ACCENT2)
        self._btn_cam.pack(side="left", padx=(0, 6))

        # Threshold slider
        tk.Label(ctrl_bar, text="Threshold:", fg=TEXT_SEC, bg=BG_CARD, font=("Segoe UI", 9)).pack(side="left", padx=(10, 2))
        self._thresh_val = tk.DoubleVar(value=0.5)
        thresh_scale = ttk.Scale(ctrl_bar, from_=0.1, to=0.9, variable=self._thresh_val, orient="horizontal", length=100)
        thresh_scale.pack(side="left", padx=5)
        
        # Right panel ─ stats/info
        right = tk.Frame(body, bg=BG_CARD, width=280)
        right.pack(side="right", fill="y")
        right.pack_propagate(False)

        tk.Label(right, text="Detection Info",
                 font=("Segoe UI", 13, "bold"), fg=TEXT_PRI, bg=BG_CARD
                 ).pack(pady=(14, 6), padx=14, anchor="w")

        self._info_text = tk.Text(right, bg=BG_CARD2, fg=TEXT_PRI, font=("Consolas", 9),
                                  bd=0, padx=10, pady=10, height=20)
        self._info_text.pack(fill="both", expand=True, padx=10, pady=10)

        self._inf_time_lbl = tk.Label(right, text="Inference: —",
                                      font=("Segoe UI", 9), fg=TEXT_SEC, bg=BG_CARD)
        self._inf_time_lbl.pack(anchor="w", padx=14, pady=2)

        self._mode_lbl = tk.Label(right, text="Mode: Idle",
                                  font=("Segoe UI", 9), fg=TEXT_SEC, bg=BG_CARD)
        self._mode_lbl.pack(pady=(2, 14), padx=14, anchor="w")

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
                opts = ort.SessionOptions()
                self._session = ort.InferenceSession(MODEL_PATH, sess_options=opts,
                    providers=["CPUExecutionProvider"])
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
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.webp"),
                       ("All files", "*.*")])
        if not path:
            return
        self._stop_webcam()
        img_bgr = cv2.imread(path)
        if img_bgr is None:
            self._status_lbl.config(text="❌ Could not read image", fg="#ef4444")
            return
        self._current_frame = img_bgr
        self._run_inference_on(img_bgr)
        self._mode_lbl.config(text="Mode: Static Image")
        self._status_lbl.config(text=f"📄 {os.path.basename(path)}", fg=TEXT_SEC)

    # ── Webcam ────────────────────────────────────────────────────────────────

    def _toggle_webcam(self):
        if self._cam_running:
            self._stop_webcam()
        else:
            self._start_webcam()

    def _start_webcam(self):
        # The user changed the webcam index to 1 in the previous file, so I'll try 1 first, then 0.
        self._cap = cv2.VideoCapture(1)
        if not self._cap.isOpened():
            self._cap = cv2.VideoCapture(0)
            
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

    def _webcam_loop(self):
        while self._cam_running:
            ret, frame = self._cap.read()
            if not ret:
                break
            
            self._current_frame = frame
            
            # Periodic live inference
            now = time.time()
            if self._session and now - self._last_infer_time >= self._infer_interval:
                self._last_infer_time = now
                self._run_inference_on(frame.copy(), is_webcam=True)
            elif not self._session:
                self.after(0, lambda f=frame.copy(): self._show_frame(f, []))
                
            time.sleep(0.03)

    # ── Inference ─────────────────────────────────────────────────────────────

    def _run_inference_on(self, img_bgr: np.ndarray, is_webcam=False):
        if self._session is None:
            return

        def _infer():
            t0 = time.perf_counter()
            thresh = self._thresh_val.get()
            detections = run_inference(self._session, img_bgr, threshold=thresh)
            elapsed = (time.perf_counter() - t0) * 1000
            
            self.after(0, lambda: self._update_ui(img_bgr, detections, elapsed))

        threading.Thread(target=_infer, daemon=True).start()

    def _update_ui(self, img_bgr, detections, elapsed_ms):
        self._inf_time_lbl.config(text=f"Inference: {elapsed_ms:.1f} ms")
        self._show_frame(img_bgr, detections)
        
        # Update info text
        self._info_text.delete("1.0", tk.END)
        if not detections:
            self._info_text.insert(tk.END, "No objects detected.")
        else:
            self._info_text.insert(tk.END, f"Detected {len(detections)} objects:\n\n")
            for det in detections:
                cls_name = COCO_CLASSES[det['class_id']] if det['class_id'] < len(COCO_CLASSES) else f"ID:{det['class_id']}"
                self._info_text.insert(tk.END, f"• {cls_name:<12} {det['score']*100:>5.1f}%\n")

    # ── Frame Display with Bounding Boxes ─────────────────────────────────────

    def _show_frame(self, img_bgr: np.ndarray, detections):
        # Draw detections
        ih, iw = img_bgr.shape[:2]
        display_img = img_bgr.copy()
        
        for det in detections:
            cx, cy, w, h = det['box']
            # Normalized to pixel coords
            x1 = int((cx - w/2) * iw)
            y1 = int((cy - h/2) * ih)
            x2 = int((cx + w/2) * iw)
            y2 = int((cy + h/2) * ih)
            
            # Clip to image boundaries
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(iw, x2), min(ih, y2)
            
            color = (108, 99, 255) # ACCENT hex #6c63ff in BGR
            cv2.rectangle(display_img, (x1, y1), (x2, y2), color, 2)
            
            cls_name = COCO_CLASSES[det['class_id']] if det['class_id'] < len(COCO_CLASSES) else f"ID:{det['class_id']}"
            label = f"{cls_name} {det['score']:.2f}"
            
            # Draw label background
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(display_img, (x1, y1 - th - 5), (x1 + tw, y1), color, -1)
            cv2.putText(display_img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Scale for preview
        pw = max(self._preview.winfo_width(), 400)
        ph = max(self._preview.winfo_height(), 340)
        ih, iw = display_img.shape[:2]
        scale = min(pw / iw, ph / ih, 1.0)
        nw, nh = int(iw * scale), int(ih * scale)
        
        resized = cv2.resize(display_img, (nw, nh))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        photo = ImageTk.PhotoImage(pil)
        self._preview.config(image=photo, text="")
        self._preview._photo = photo

# ══════════════════════════════════════════════════════════════════════════════
#  Utility
# ══════════════════════════════════════════════════════════════════════════════

def _lighten(hex_color: str, amount=30) -> str:
    hex_color = hex_color.lstrip("#")
    r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
    r = min(255, r + amount); g = min(255, g + amount); b = min(255, b + amount)
    return f"#{r:02x}{g:02x}{b:02x}"

if __name__ == "__main__":
    app = ConditionalDetrApp()
    app.update_idletasks()
    W, H = 1000, 750
    sw = app.winfo_screenwidth()
    sh = app.winfo_screenheight()
    app.geometry(f"{W}x{H}+{(sw-W)//2}+{(sh-H)//2}")
    app.mainloop()
