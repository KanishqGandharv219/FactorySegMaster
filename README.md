# FactorySegMaster

**7-Day Bootcamp - Production Factory Segmentation System - FactoryTwin Foundation - $B China SME Opportunity**

[![Gradio](https://img.shields.io/badge/Gradio-Demo-brightgreen)](https://github.com/KanishqGandharv219/FactorySegMaster)
[![OpenCV](https://img.shields.io/badge/OpenCV-v4.9-blue)](https://opencv.org)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-v0.10-green)](https://mediapipe.dev)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-nano-orange)](https://ultralytics.com)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## The Vision
Build **world-class instance segmentation** for small factories (40M in China alone). From blurry CCTV - precise worker/machine/defect masks - AI production brain.

```
CCTV Frame -> [FactorySegMaster] -> Segmented Objects -> FactoryTwin Planner
   (CCTV)                 (Result)                    (Assets)              (AI) "Move Worker A to Line 2"
```

## 7-Day Mastery Path

| Day | Status | Topic | Tools |
|-----|--------|-------|-------|
| 1 | Done | **OpenCV Contours** (Machine Isolation) | Adaptive Thresh, Morphology |
| 2 | Done | **MediaPipe** (Worker Hands/Poses) | Safety Zone Monitoring |
| 3 | Done | **YOLOv8** (Factory Object Detection) | Custom Class Training |
| 4 | Pending | **SAM2** (Zero-shot Segmentation) | Tools, Belts, Specific Parts |
| 5 | Pending | **Tracking** (ByteTrack IDs) | Persistent ID Tracking |
| 6 | Pending | **Custom Dataset Training** (Finetuning) | Roboflow / CVAT |
| 7 | Pending | **Ensemble + FactoryTwin Demo** | Full Pipeline Integration |

## Day 1: OpenCV Segmentation
We use **Adaptive Gaussian Thresholding** and **Morphological Gradient** analysis to separate light-gray machines from light-gray factory floors. 

### Quick Start
```bash
# Clone and enter Day 1 directory
cd day1_opencv
pip install -r requirements.txt
python demo.py
```
Open the local URL - Upload your factory CCTV frame - Tune parameters in real-time.

### Key Features (Day 1)
- **Gradient Mode:** Boundary detection for light-on-light scenes.
- **Adaptive Mode:** Handles uneven factory lighting (best for high-res images).
- **Convexity Filter:** Isolates boxy machine shapes and filters organic noise.
- **Live Metrics:** Real-time object count and floor area coverage.

## Day 2: MediaPipe Worker Detection
Using **MediaPipe Tasks API PoseLandmarker** for multi-person pose estimation and **HandLandmarker** for hand tracking. Includes a safety zone framework for machine-area violation detection.

### Quick Start
```bash
cd day2_mediapipe
pip install -r requirements.txt
python demo.py
```
Models auto-download on first run (~13MB total).

### Key Features (Day 2)
- **Multi-person pose detection** via PoseLandmarker Tasks API (best-effort, num_poses=5).
- **Hand tracking** with HandLandmarker for assembly QC monitoring.
- **Slider-based safety zones** with real-time violation alerts.
- **Video processing** mode for frame-by-frame analysis of factory footage.
- **Per-worker colored skeletons** with joint visibility tracking.

### Limitations
- MediaPipe PoseLandmarker multi-person is officially "out of scope" for the model. Works well for 2-3 workers; Day 3 YOLOv8-Pose gives proper multi-person support.

## Day 3: YOLOv8 Factory Object Detection
Using **Ultralytics YOLOv8n** (nano) for robust multi-person and object detection. Solves the MediaPipe occlusion issues from Day 2. Detects workers, vehicles, and other COCO objects with proper bounding boxes.

### Quick Start
```bash
cd day3_yolov8
pip install -r requirements.txt
python demo.py
```
YOLOv8n (~6MB) auto-downloads on first run.

### Key Features (Day 3)
- **Proper multi-person detection** -- handles 20+ overlapping workers without landmark jumping.
- **Configurable class filtering** -- toggle Person vs Vehicle detection via checkboxes.
- **Confidence and NMS IoU sliders** for tuning precision vs recall.
- **Bounding-box safety zones** -- triggers alerts when a person's bottom-center enters a zone.
- **Full video processing** with per-frame annotation and summary stats.

## Endgame Stack
- **Real-time:** YOLOv8 + SAM2 ensemble
- **Tracking:** ByteTrack (persistent IDs)
- **Deploy:** FastAPI + WebRTC stream
- **Edge:** ONNX (Raspberry Pi factory cams)

## Why Segmentation-First?
1. **Factory AI bottleneck:** Nobody segments factories well at scale for SMEs.
2. **Viral demos:** Factory photo - precise masks = instant credibility.
3. **Foundation for FactoryTwin:** Precise spatial mapping is the prerequisite for AI motion planning.

---
**Day 4** next: SAM2 zero-shot segmentation for tools, belts, and specific parts.
