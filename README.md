---
title: Pacific Parrotlet Behavior Detector
emoji: 🦜
colorFrom: green
colorTo: yellow
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
license: mit
---

# Parrot Recognition & Object Detection System (YOLOv8 + Gradio)

A modular, PEP 8-compliant Python application for parrot recognition and object detection using **YOLOv8** and **Gradio**.

---

## 📁 Directory Structure

```text
.
├── models/
│   └── parrot_yolov8.pt       # Trained YOLOv8 model weights
├── media/
│   ├── input/                  # Input image and video test files
│   │   ├── napping_image.jpg
│   │   └── eating_video.mp4
│   └── output/                 # Generated detection outputs
│       ├── result_napping_image.jpg
│       └── result_eating_video.mp4
├── src/
│   ├── __init__.py
│   ├── detector.py            # Core YOLOv8 inference class (ParrotDetector)
│   └── utils.py               # Helper utility functions (scaling, paths, formatting)
├── app.py                     # Gradio web interface entry point
├── detect.py                  # Command-line interface (CLI) entry point
├── requirements.txt           # Project dependencies
└── README.md                  # Project documentation
```

---

## 🚀 Installation & Setup

1. **Clone the repository** and navigate to the project directory:
   ```bash
   git clone <repository_url>
   cd parrot-detection
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

---

## 💻 CLI Execution (`detect.py`)

Run object detection locally via the command-line interface.

### Options:
- `--mode`: `image` or `video` (default: `image`)
- `--source`: Path to input image/video file (default: `media/input/napping_image.jpg`)
- `--weights`: Path to model weights (default: `models/parrot_yolov8.pt`)
- `--conf`: Confidence threshold (default: `0.5`)
- `--scale`: OpenCV preview window scale factor for video (default: `0.5`)
- `--show`: Optional flag to open local OpenCV preview window

### Examples:

- **Image Inference**:
  ```bash
  python detect.py --mode image --source media/input/napping_image.jpg --conf 0.5
  ```

- **Video Inference**:
  ```bash
  python detect.py --mode video --source media/input/eating_video.mp4 --conf 0.5
  ```

Outputs are automatically saved to `media/output/`.

---

## 🌐 Gradio Web Interface (`app.py`)

Launch the web application with interactive tabs for Image and Video analysis:

```bash
python app.py
```

Access the interface in your web browser at `http://localhost:7860`.

### Features:
- **Image Analysis Tab**: Displays annotated image alongside detected class names and confidence scores formatted as percentages (e.g. `Parrot: 97.89%`).
- **Video Analysis Tab**: Processes uploaded videos, saves output to `media/output/`, and provides preview and download capabilities.

---

## ⚙️ Module Design (`src/`)

- `src/detector.py`: Contains `ParrotDetector` class. Replaces print calls with standard Python `logging`. Supports customizable confidence thresholds (`conf=0.5`).
- `src/utils.py`: Provides helper utilities for path handling, image/frame scaling (`scale=0.5`), and detection dictionary/text formatting.
