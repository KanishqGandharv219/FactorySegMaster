# FactorySegMaster

**7-Day Bootcamp - Production Factory Segmentation System - FactoryTwin Foundation - $B China SME Opportunity**

[![Gradio](https://img.shields.io/badge/Gradio-Demo-brightgreen)](https://github.com/KanishqGandharv219/FactorySegMaster)
[![OpenCV](https://img.shields.io/badge/OpenCV-v4.9-blue)](https://opencv.org)
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
| 2 | Pending | **MediaPipe** (Worker Hands/Poses) | Safety Zone Monitoring |
| 3 | Pending | **YOLOv8** (Factory Object Detection) | Custom Class Training |
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
**Day 2** Next is MediaPipe worker detection.
