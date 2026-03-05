"""
FactorySegMaster Day 5: ByteTrack Object Tracking

Core tracker class wrapping ultralytics YOLO + ByteTrack.
Provides persistent IDs to detected objects across video frames.
"""

import cv2
import numpy as np
from ultralytics import YOLO

class PersistentTracker:
    def __init__(self, model_size="yolov8n.pt", conf_thresh=0.25, iou_thresh=0.45):
        """
        Initialize the YOLO tracker.
        """
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh
        self.model = YOLO(model_size)
        
        # COCO class mapping for filtering
        # 0: person, 2: car (simulating a factory vehicle/AGV), 7: truck (simulating machinery)
        self.target_classes = [0, 2, 7] 
        self.class_names = self.model.names

    def set_thresholds(self, conf, iou):
        """Update detection thresholds interactively."""
        self.conf_thresh = conf
        self.iou_thresh = iou

    def track_frame(self, frame_rgb, target_classes=None):
        """
        Run inference and tracking on a single frame from a video stream.
        Maintains internal state between calls.
        
        Args:
            frame_rgb: numpy array (H, W, 3) RGB image
            target_classes: list of int class IDs to filter. Defaults to [0, 2, 7]
            
        Returns:
            dict containing:
                "annotated": RGB image with tracked bounding boxes drawn
                "objects": list of dicts {"id": int, "bbox": (x1,y1,x2,y2), "class_name": str, "conf": float}
        """
        if target_classes is None:
            target_classes = self.target_classes

        # Use .track() instead of .predict() to enable ByteTrack
        results = self.model.track(
            source=frame_rgb,
            conf=self.conf_thresh,
            iou=self.iou_thresh,
            classes=target_classes,
            tracker="bytetrack.yaml", # Built-in ultralytics tracker config
            persist=True,             # Keep tracking IDs alive between frames
            verbose=False             # Suppress per-frame terminal logs
        )
        
        r = results[0]
        annotated = r.plot() 
        
        tracked_objects = []
        if r.boxes is not None and r.boxes.id is not None:
            # We have detections AND they have been assigned IDs by ByteTrack
            for i, box in enumerate(r.boxes):
                # Sometimes a box is detected but not confident enough for the tracker to assign an ID yet
                if r.boxes.id is None or i >= len(r.boxes.id):
                    continue
                    
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                cls_name = self.class_names[cls_id]
                track_id = int(r.boxes.id[i].item())
                
                tracked_objects.append({
                    "id": track_id,
                    "bbox": (int(x1), int(y1), int(x2), int(y2)),
                    "class_id": cls_id,
                    "class_name": cls_name,
                    "conf": conf
                })
                
        return {
            "annotated": annotated,
            "objects": tracked_objects,
            "count": len(tracked_objects)
        }
        
    def reset(self):
        """
        Clear the internal tracker state. 
        MUST be called when processing a new completely different video, 
        otherwise IDs from the old video will corrupt the new one.
        """
        # Hacky way to reset ultralytics tracker state without reloading weights
        if hasattr(self.model, 'predictor') and self.model.predictor is not None:
            if hasattr(self.model.predictor, 'trackers'):
                self.model.predictor.trackers = None
