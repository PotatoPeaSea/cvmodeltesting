# CV Model Testing Playground

A Python-based collection of interactive computer vision demos focused on running ONNX models locally with ONNX Runtime.

This repository includes:
- **Image classification** demos (EfficientNet-B0, BEiT)
- **Object detection** demo (Conditional-DETR)
- **Interactive segmentation** demos (EdgeTAM / SAM2-style)
- A **Universal Model Explorer** that auto-discovers models from `metadata.json`

Most demos provide a desktop UI (Tkinter + OpenCV preview), support image upload and webcam input, and display real-time inference results.

---

## Repository Structure

```text
cvmodeltesting/
├── README.md
├── model_explorer.py
├── efficientnet_demo.py
├── beit/
│   └── beit_demo.py
├── conditional-detr/
│   └── conditional_detr_demo.py
├── edgetam/
│   ├── edgetam_demo.py
│   ├── sam2_opencv_demo.py
│   └── test_combinations.py
└── sam2/
    └── sam2_demo.py
```

---

## Prerequisites

- Python **3.10+** required
- A desktop environment (for Tkinter/OpenCV windows)
- Optional webcam for live demos

### Python dependencies

Install core dependencies:

```bash
pip install numpy opencv-python onnxruntime pillow
```

For `sam2/sam2_demo.py`, install extra dependencies from Qualcomm AI Hub models:

```bash
pip install "qai-hub-models[sam2]" git+https://github.com/facebookresearch/sam2.git
```

---

## Model Files You Must Provide

Model binaries are not committed in this repo. Place ONNX files where each script expects them.

| Demo | Script | Expected model assets |
|---|---|---|
| EfficientNet-B0 classifier | `efficientnet_demo.py` | `./efficientnet_b0.onnx`, `./labels.txt` |
| BEiT classifier | `beit/beit_demo.py` | `./beit/beit.onnx`, `./beit/labels.txt` |
| Conditional-DETR detector | `conditional-detr/conditional_detr_demo.py` | `./conditional-detr/conditional-detr/conditional_detr_resnet50.onnx` (nested path is required because the script resolves `SCRIPT_DIR/conditional-detr/conditional_detr_resnet50.onnx`) |
| EdgeTAM Tkinter app | `edgetam/edgetam_demo.py` | `./edgetam/edgetam/encoder.onnx`, `./edgetam/edgetam/decoder.onnx` |
| SAM2 OpenCV app | `edgetam/sam2_opencv_demo.py` | CLI defaults: `encoder.onnx`, `decoder.onnx` in current working dir (or pass `--encoder` / `--decoder`) |
| Universal Explorer | `model_explorer.py` | One or more model folders containing `metadata.json` + listed model files |
| Qualcomm SAM2 demo | `sam2/sam2_demo.py` | Pulled by `qai_hub_models` at runtime |

---

## Quick Start

Run commands from repository root unless noted otherwise.

### 1) Universal model explorer

```bash
python model_explorer.py
```

What it does:
- Scans root and first-level subfolders for `metadata.json`
- Infers task type (classification, detection, segmentation)
- Loads model(s) with ONNX Runtime
- Supports image upload and webcam

### 2) EfficientNet-B0 classification demo

```bash
python efficientnet_demo.py
```

Features:
- Open image or use webcam
- Top-5 ImageNet predictions with animated confidence bars

### 3) BEiT classification demo

```bash
python beit/beit_demo.py
```

Features:
- Open image or use webcam
- Top-5 predictions similar to EfficientNet UI

### 4) Conditional-DETR detection demo

```bash
python conditional-detr/conditional_detr_demo.py
```

Features:
- Open image or use webcam
- Bounding boxes + class labels + confidence
- Adjustable detection threshold slider

### 5) EdgeTAM interactive segmentation (Tkinter)

```bash
python edgetam/edgetam_demo.py
```

Features:
- Open image or capture webcam frame
- Left-click = positive prompt, right-click = negative prompt
- Segmentation mask overlay + confidence score

### 6) SAM2 OpenCV interactive/non-interactive demo

Interactive:

```bash
python edgetam/sam2_opencv_demo.py --encoder <path/to/encoder.onnx> --decoder <path/to/decoder.onnx> --image <path/to/image>
```

Non-interactive (saves output directly):

```bash
python edgetam/sam2_opencv_demo.py --encoder <path/to/encoder.onnx> --decoder <path/to/decoder.onnx> --image <path/to/image> --output <path/to/output.png>
```

### 7) Qualcomm SAM2 demo

```bash
python sam2/sam2_demo.py
```

---

## Using `metadata.json` with `model_explorer.py`

`model_explorer.py` expects each model directory to include:
- `metadata.json`
- model file entries under `model_files`
- input/output descriptions used to choose and run the correct processor

Task detection logic in explorer:
- **Segmentation** if `encoder.onnx` is present in `model_files`
- **Detection** if any output has `"io_type": "bbox"`
- Otherwise **Classification**

Label behavior:
- Uses `labels_file` hints from metadata outputs when present
- Falls back to local `labels.txt`
- Falls back again to COCO class names for detection-like models

---

## Troubleshooting

### `Model load failed` / file not found
- Verify ONNX paths exactly match script expectations in the table above.
- Use absolute paths in CLI arguments where supported.

### Webcam not opening
- Some scripts try camera index `1` before `0`.
- Close other apps using the camera.
- Try rerunning after reconnecting webcam.

### Tkinter window does not open
- Ensure you are running in a desktop session (not headless shell-only environment).
- On Linux servers, use X forwarding or run locally.

### Slow inference
- CPU provider is used in several scripts by default.
- Reduce webcam usage or input resolution where possible.

---

## Notes and Limitations

- This repository is demo-oriented and does not have a configured automated test suite.
- Model binaries are intentionally not included.
- Some scripts are tailored to specific model I/O signatures (especially segmentation and detector paths).

---

## License

Check individual source files for licensing notices. For example, `sam2/sam2_demo.py` includes Qualcomm copyright and SPDX headers.
