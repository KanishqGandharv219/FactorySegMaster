"""
FactorySegMaster Day 3: YOLOv8 Object Detection

Core detector class wrapping ultralytics YOLO.
Handles multi-person tracking and object detection built for factory scenes.
"""

import cv2
import numpy as np
from ultralytics import YOLO

class YoloDetector:
    def __init__(self, model_size="yolov8n.pt", conf_thresh=0.25, iou_thresh=0.45):
        """
        Initialize the YOLOv8 detector.
        Downloads the model automatically on first run.
        """
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh
        # ultralytics caches the download in the current directory or ultralytics hub folder
        self.model = YOLO(model_size)
        
        # COCO class mapping for filtering
        # 0: person, 2: car (simulating a factory vehicle/AGV), 7: truck (simulating machinery)
        self.target_classes = [0, 2, 7] 
        self.class_names = self.model.names

    def set_thresholds(self, conf, iou):
        """Update detection thresholds interactively."""
        self.conf_thresh = conf
        self.iou_thresh = iou

    def detect(self, frame_rgb, target_classes=None):
        """
        Run inference on a single RGB frame.
        
        Args:
            frame_rgb: numpy array (H, W, 3) RGB image
            target_classes: list of int class IDs to filter. Defaults to [0, 2, 7]
            
        Returns:
            dict containing:
                "annotated": RGB image with bounding boxes drawn
                "objects": list of dicts {"bbox": (x1,y1,x2,y2), "class_id": int, "class_name": str, "conf": float}
        """
        if target_classes is None:
            target_classes = self.target_classes

        # Ultralytics accepts RGB/BGR arrays directly. 
        # Using half precision for speed if MPS/CUDA is available, but CPU runs float32
        results = self.model.predict(
            source=frame_rgb,
            conf=self.conf_thresh,
            iou=self.iou_thresh,
            classes=target_classes,
            verbose=False # Suppress per-frame terminal logs for video speed
        )
        
        # We only passed one frame, so results is a list of 1
        r = results[0]
        
        # The annotated image directly from ultralytics (adds labels + boxes)
        annotated = r.plot() 
        
        detected_objects = []
        if r.boxes:
            for box in r.boxes:
                # box.xyxy is a tensor of shape (1, 4), squeeze and convert to python floats
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                cls_name = self.class_names[cls_id]
                
                detected_objects.append({
                    "bbox": (int(x1), int(y1), int(x2), int(y2)),
                    "class_id": cls_id,
                    "class_name": cls_name,
                    "conf": conf
                })
                
        return {
            "annotated": annotated,
            "objects": detected_objects,
            "count": len(detected_objects)
        }
