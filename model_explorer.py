"""
Universal Model Explorer
========================
A generalized interface that adapts to Classification, Detection, and 
Interactive Segmentation models based on metadata.json definitions.
"""

import os
import sys
import threading
import time
import json
import tkinter as tk
from tkinter import filedialog, ttk
import numpy as np
import cv2
import onnxruntime as ort
from PIL import Image, ImageTk

# ─── Colour palette ────────────────────────────────────────────────────────────
BG_DARK   = "#0f1117"
BG_CARD   = "#1a1d27"
BG_CARD2  = "#22263a"
ACCENT    = "#6c63ff"
ACCENT2   = "#00d4aa"
TEXT_PRI  = "#f0f0f5"
TEXT_SEC  = "#8b8fa8"

# ─── Constants ─────────────────────────────────────────────────────────────────
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

# ══════════════════════════════════════════════════════════════════════════════
#  Generalized Model Processors
# ══════════════════════════════════════════════════════════════════════════════

class BaseModelProcessor:
    def __init__(self, metadata, base_path):
        self.metadata = metadata
        self.base_path = base_path
        self.sessions = {}
        self.labels = []
        self._load_labels()

    def _load_labels(self):
        # Try to find a labels file mentioned in metadata or local labels.txt
        labels_file = "labels.txt"
        # Check metadata for labels_file hint
        for model_info in self.metadata.get("model_files", {}).values():
            for out_info in model_info.get("outputs", {}).values():
                if "labels_file" in out_info:
                    labels_file = out_info["labels_file"]
                    break
        
        path = os.path.join(self.base_path, labels_file)
        if not os.path.exists(path):
            path = os.path.join(os.path.dirname(self.base_path), labels_file) # check root too

        if os.path.exists(path):
            with open(path, "r") as f:
                self.labels = [line.strip() for line in f.readlines()]
        else:
            # Fallback to COCO if it looks like a 80-class model
            self.labels = COCO_CLASSES

    def load_models(self):
        providers = ['CPUExecutionProvider']
        model_files = self.metadata.get("model_files", {})
        for filename, info in model_files.items():
            path = os.path.join(self.base_path, filename)
            if not os.path.exists(path):
                # Check root if not in subdir
                path = os.path.join(os.path.dirname(self.base_path), filename)
            self.sessions[filename] = ort.InferenceSession(path, providers=providers)

    def preprocess(self, img_bgr, input_info):
        shape = input_info["shape"] # [1, 3, H, W]
        target_h, target_w = shape[2], shape[3]
        
        img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))
        return np.expand_dims(img, axis=0)

    def run(self, img_bgr, **kwargs):
        raise NotImplementedError()

class ClassificationProcessor(BaseModelProcessor):
    def run(self, img_bgr, **kwargs):
        # Assume first model, first input
        model_name = list(self.sessions.keys())[0]
        sess = self.sessions[model_name]
        input_name = sess.get_inputs()[0].name
        model_info = self.metadata["model_files"][model_name]
        input_info = model_info["inputs"][input_name]
        
        tensor = self.preprocess(img_bgr, input_info)
        logits = sess.run(None, {input_name: tensor})[0][0]
        
        # Softmax
        e = np.exp(logits - np.max(logits))
        probs = e / e.sum()
        
        top5_idx = np.argsort(probs)[::-1][:5]
        results = []
        for i in top5_idx:
            label = self.labels[i] if i < len(self.labels) else f"Class {i}"
            results.append({"label": label, "score": float(probs[i])})
        return {"type": "classification", "data": results}

class DetectionProcessor(BaseModelProcessor):
    def run(self, img_bgr, threshold=0.5, **kwargs):
        model_name = list(self.sessions.keys())[0]
        sess = self.sessions[model_name]
        input_name = sess.get_inputs()[0].name
        model_info = self.metadata["model_files"][model_name]
        input_info = model_info["inputs"][input_name]
        
        tensor = self.preprocess(img_bgr, input_info)
        outputs = sess.run(None, {input_name: tensor})
        
        # Mapping outputs by order in metadata or name
        # We'll assume the standard [boxes, logits, classes] order for Conditional-DETR
        # or similar.
        out_boxes = outputs[0][0]
        out_scores = outputs[1][0]
        out_classes = outputs[2][0]
        
        if np.max(out_scores) > 1.0 or np.min(out_scores) < 0.0:
            out_scores = 1 / (1 + np.exp(-out_scores)) # Sigmoid if logits
            
        detections = []
        for i in range(len(out_scores)):
            score = float(out_scores[i])
            if score < threshold: continue
            
            detections.append({
                "class_id": int(out_classes[i]),
                "label": self.labels[int(out_classes[i])] if int(out_classes[i]) < len(self.labels) else f"ID:{out_classes[i]}",
                "score": score,
                "box": out_boxes[i] # cx, cy, w, h
            })
        return {"type": "detection", "data": detections}

class SegmentationProcessor(BaseModelProcessor):
    def __init__(self, metadata, base_path):
        super().__init__(metadata, base_path)
        self.input_tensor = None
        self.scale = 1.0
        self.new_h = 0
        self.new_w = 0
        self.img_h = 0
        self.img_w = 0

    def prepare(self, img_bgr):
        self.img_h, self.img_w = img_bgr.shape[:2]
        self.scale = 1024.0 / max(self.img_h, self.img_w)
        self.new_h = int(self.img_h * self.scale + 0.5)
        self.new_w = int(self.img_w * self.scale + 0.5)
        
        rgb_img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        resized_img = cv2.resize(rgb_img, (self.new_w, self.new_h), interpolation=cv2.INTER_LINEAR)
        
        padded_img = np.zeros((1024, 1024, 3), dtype=np.float32)
        padded_img[:self.new_h, :self.new_w, :] = resized_img.astype(np.float32) / 255.0
        chw_img = np.transpose(padded_img, (2, 0, 1))
        self.input_tensor = np.expand_dims(chw_img, axis=0)

    def run(self, img_bgr, points=None, **kwargs):
        if not points: return None
        
        mapped_coords = []
        mapped_labels = []
        for pt in points[-2:]:
            mapped_coords.append([float(pt[0] * self.scale), float(pt[1] * self.scale)])
            mapped_labels.append(pt[2])
        while len(mapped_coords) < 2:
            mapped_coords.append([0.0, 0.0])
            mapped_labels.append(-1.0)
            
        # Encoder
        enc_sess = self.sessions["encoder.onnx"]
        enc_out = enc_sess.run(None, {
            "image": self.input_tensor,
            "unnorm_coords": np.array([mapped_coords], dtype=np.float32),
            "labels": np.array([mapped_labels], dtype=np.float32)
        })
        
        # Decoder
        dec_sess = self.sessions["decoder.onnx"]
        masks, scores = dec_sess.run(None, {
            "image_embeddings": enc_out[0],
            "high_res_features1": enc_out[1],
            "high_res_features2": enc_out[2],
            "sparse_embedding": enc_out[3]
        })
        
        # Postprocess
        mask = (masks[0, 0] < 0.0).astype(np.uint8)
        full_padded_mask = cv2.resize(mask, (1024, 1024), interpolation=cv2.INTER_NEAREST)
        valid_mask = full_padded_mask[:self.new_h, :self.new_w]
        full_mask = cv2.resize(valid_mask, (self.img_w, self.img_h), interpolation=cv2.INTER_NEAREST)
        
        return {"type": "segmentation", "data": {"mask": full_mask, "score": float(scores[0, 0])}}

# ══════════════════════════════════════════════════════════════════════════════
#  UI Widgets
# ══════════════════════════════════════════════════════════════════════════════

class ConfidenceBar(tk.Frame):
    def __init__(self, parent, label, score, color):
        super().__init__(parent, bg=BG_CARD2)
        tk.Label(self, text=label[:20], font=("Segoe UI", 9), fg=TEXT_PRI, bg=BG_CARD2, width=20, anchor="w").pack(side="left", padx=5)
        self.canvas = tk.Canvas(self, height=8, bg="#2e3250", highlightthickness=0)
        self.canvas.pack(side="left", fill="x", expand=True, padx=5)
        self.color = color
        self.after(100, lambda: self._draw(score))
        tk.Label(self, text=f"{score*100:.1f}%", font=("Segoe UI", 9, "bold"), fg=color, bg=BG_CARD2, width=6).pack(side="right", padx=5)

    def _draw(self, score):
        w = self.canvas.winfo_width()
        self.canvas.delete("all")
        self.canvas.create_rectangle(0, 0, int(w * score), 8, fill=self.color, outline="")

# ══════════════════════════════════════════════════════════════════════════════
#  Main Application
# ══════════════════════════════════════════════════════════════════════════════

class UniversalModelApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Universal Vision Model Explorer")
        self.configure(bg=BG_DARK)
        self.geometry("1100x800")
        
        self.current_processor = None
        self.models_found = []
        self.orig_image = None
        self.points = []
        self.cam_running = False
        self.cap = None

        self._build_ui()
        self._discover_models()

    def _build_ui(self):
        # Sidebar
        side = tk.Frame(self, bg=BG_CARD, width=280)
        side.pack(side="left", fill="y")
        side.pack_propagate(False)

        tk.Label(side, text="Models", font=("Segoe UI", 14, "bold"), fg=TEXT_PRI, bg=BG_CARD).pack(pady=20, padx=20, anchor="w")
        
        self.model_list = ttk.Combobox(side, state="readonly")
        self.model_list.pack(fill="x", padx=20)
        self.model_list.bind("<<ComboboxSelected>>", self._on_model_selected)

        tk.Label(side, text="Controls", font=("Segoe UI", 11, "bold"), fg=TEXT_PRI, bg=BG_CARD).pack(pady=(30, 5), padx=20, anchor="w")
        
        btn_frame = tk.Frame(side, bg=BG_CARD)
        btn_frame.pack(fill="x", padx=20)
        
        self._make_btn(btn_frame, "📂 Open Image", self._open_image, ACCENT).pack(fill="x", pady=5)
        self.cam_btn = self._make_btn(btn_frame, "📷 Start Webcam", self._toggle_webcam, ACCENT2)
        self.cam_btn.pack(fill="x", pady=5)
        
        self.thresh_var = tk.DoubleVar(value=0.5)
        tk.Label(side, text="Threshold:", fg=TEXT_SEC, bg=BG_CARD, font=("Segoe UI", 9)).pack(pady=(20, 0), padx=20, anchor="w")
        ttk.Scale(side, from_=0.1, to=0.9, variable=self.thresh_var, orient="horizontal").pack(fill="x", padx=20, pady=5)

        # Main Area
        main = tk.Frame(self, bg=BG_DARK)
        main.pack(side="right", fill="both", expand=True)
        
        # Display
        self.display = tk.Label(main, bg=BG_DARK, text="Select a model to begin", fg=TEXT_SEC, cursor="crosshair")
        self.display.pack(fill="both", expand=True, padx=20, pady=20)
        self.display.bind("<Button-1>", lambda e: self._on_click(e, 1.0))
        self.display.bind("<Button-3>", lambda e: self._on_click(e, 0.0))

        # Bottom Results Area
        self.results_pane = tk.Frame(main, bg=BG_CARD, height=200)
        self.results_pane.pack(side="bottom", fill="x", padx=20, pady=(0, 20))
        self.results_pane.pack_propagate(False)
        
        self.info_lbl = tk.Label(self.results_pane, text="Ready", fg=TEXT_SEC, bg=BG_CARD, font=("Segoe UI", 9))
        self.info_lbl.pack(side="bottom", pady=5)

    def _make_btn(self, parent, text, cmd, color):
        return tk.Button(parent, text=text, command=cmd, bg=color, fg="white", font=("Segoe UI", 9, "bold"), relief="flat", padx=10, pady=8, cursor="hand2")

    def _discover_models(self):
        root = os.path.dirname(os.path.abspath(__file__))
        candidates = [root] + [os.path.join(root, d) for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))]
        
        for path in candidates:
            meta_path = os.path.join(path, "metadata.json")
            if os.path.exists(meta_path):
                with open(meta_path, "r") as f:
                    meta = json.load(f)
                    self.models_found.append({"path": path, "metadata": meta})
        
        self.model_list["values"] = [m["metadata"]["model_name"] for m in self.models_found]
        if self.models_found:
            self.model_list.current(0)
            self._on_model_selected()

    def _on_model_selected(self, _=None):
        idx = self.model_list.current()
        model = self.models_found[idx]
        meta = model["metadata"]
        path = model["path"]
        
        # Determine task type
        task = "classification"
        for m_file in meta.get("model_files", {}).values():
            for out in m_file.get("outputs", {}).values():
                if out.get("io_type") == "bbox": task = "detection"
        if "encoder.onnx" in meta.get("model_files", {}): task = "segmentation"
        
        # Load processor
        self.info_lbl.config(text=f"Loading {meta['model_name']}...")
        if task == "classification": self.current_processor = ClassificationProcessor(meta, path)
        elif task == "detection": self.current_processor = DetectionProcessor(meta, path)
        elif task == "segmentation": self.current_processor = SegmentationProcessor(meta, path)
        
        threading.Thread(target=self.current_processor.load_models, daemon=True).start()
        self.info_lbl.config(text=f"{meta['model_name']} loaded.")
        self.points = []

    def _open_image(self):
        p = filedialog.askopenfilename()
        if not p: return
        self.cam_running = False
        img = cv2.imread(p)
        if img is not None:
            self.orig_image = img
            if isinstance(self.current_processor, SegmentationProcessor):
                self.current_processor.prepare(img)
            self._run_inference()

    def _toggle_webcam(self):
        if self.cam_running:
            self.cam_running = False
            if self.cap: self.cap.release()
            if self.orig_image is not None and isinstance(self.current_processor, SegmentationProcessor):
                self.current_processor.prepare(self.orig_image)
        else:
            self.cap = cv2.VideoCapture(1)
            if not self.cap.isOpened(): self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened(): return
            self.cam_running = True
            def loop():
                while self.cam_running:
                    ret, frame = self.cap.read()
                    if not ret: break
                    self.orig_image = frame
                    self.after(0, self._run_inference)
                    time.sleep(0.05)
            threading.Thread(target=loop, daemon=True).start()

    def _on_click(self, e, label):
        if not isinstance(self.current_processor, SegmentationProcessor) or self.orig_image is None: return
        cw, ch = self.display.winfo_width(), self.display.winfo_height()
        ih, iw = self.orig_image.shape[:2]
        s = min(cw/iw, ch/ih, 1.0)
        ix = int((e.x - (cw - iw*s)//2) / s)
        iy = int((e.y - (ch - ih*s)//2) / s)
        if 0 <= ix < iw and 0 <= iy < ih:
            self.points.append((ix, iy, label))
            if len(self.points) > 2: self.points.pop(0)
            self._run_inference()

    def _run_inference(self):
        if self.orig_image is None or not self.current_processor: return
        def task():
            t0 = time.perf_counter()
            res = self.current_processor.run(self.orig_image, threshold=self.thresh_var.get(), points=self.points)
            dt = (time.perf_counter() - t0) * 1000
            self.after(0, lambda: self._update_ui(res, dt))
        threading.Thread(target=task, daemon=True).start()

    def _update_ui(self, res, dt):
        self.info_lbl.config(text=f"Inference: {dt:.1f}ms")
        for w in self.results_pane.winfo_children():
            if w != self.info_lbl: w.destroy()
        
        display_img = self.orig_image.copy()
        if res:
            if res["type"] == "classification":
                for i, item in enumerate(res["data"]):
                    ConfidenceBar(self.results_pane, item["label"], item["score"], ACCENT if i==0 else TEXT_SEC).pack(fill="x", pady=2)
            elif res["type"] == "detection":
                ih, iw = display_img.shape[:2]
                for det in res["data"]:
                    cx, cy, w, h = det["box"]
                    x1, y1 = int((cx-w/2)*iw), int((cy-h/2)*ih)
                    x2, y2 = int((cx+w/2)*iw), int((cy+h/2)*ih)
                    cv2.rectangle(display_img, (x1, y1), (x2, y2), (108, 99, 255), 2)
                    cv2.putText(display_img, f"{det['label']} {det['score']:.2f}", (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
            elif res["type"] == "segmentation":
                mask = res["data"]["mask"]
                display_img[mask == 1] = display_img[mask == 1] * 0.4 + np.array([255, 144, 30]) * 0.6
                for pt in self.points:
                    cv2.circle(display_img, (pt[0], pt[1]), 5, (0,255,0) if pt[2]==1 else (0,0,255), -1)

        # Scale and show
        cw, ch = self.display.winfo_width(), self.display.winfo_height()
        ih, iw = display_img.shape[:2]
        s = min(cw/iw, ch/ih, 1.0)
        resized = cv2.resize(display_img, (int(iw*s), int(ih*s)))
        photo = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)))
        self.display.config(image=photo, text="")
        self.display._photo = photo

if __name__ == "__main__":
    app = UniversalModelApp()
    app.mainloop()
