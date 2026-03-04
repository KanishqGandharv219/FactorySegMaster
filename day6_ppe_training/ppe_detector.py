"""
FactorySegMaster Day 6: Custom PPE Inference

Loads the fine-tuned custom weights (best.pt) to detect
Hardhats and Safety Vests on factory workers.
"""

import cv2
import os
import glob
from ultralytics import YOLO

def get_latest_weights(base_dir="runs/detect"):
    pattern = os.path.join(base_dir, "*", "weights", "best.pt")
    files = glob.glob(pattern)
    if not files:
        return "runs/detect/ppe_detector/weights/best.pt"
    return max(files, key=os.path.getmtime)

class PPEDetector:
    def __init__(self, model_path=None, conf_thresh=0.4):
        """
        Initialize the custom-trained PPE detector.
        """
        self.conf_thresh = conf_thresh
        
        if model_path is None:
            model_path = get_latest_weights()
            
        if not os.path.exists(model_path):
            print(f"WARNING: Custom model not found at {model_path}.")
            print("Falling back to standard yolov8n.pt for demonstration ONLY")
            print("Run train_ppe.py first to get custom PPE weights!")
            self.model = YOLO("yolov8n.pt")
        else:
            self.model = YOLO(model_path)
            
        self.class_names = self.model.names
        
    def set_threshold(self, conf):
        self.conf_thresh = conf
        
    def detect_ppe(self, frame_rgb):
        """
        Run inference using the custom weights.
        Returns the annotated image and a list of detections.
        """
        results = self.model.predict(
            source=frame_rgb,
            conf=self.conf_thresh,
            verbose=False
        )
        
        r = results[0]
        
        # We handle drawing the boxes manually to color-code them based on compliance
        annotated = frame_rgb.copy()
        
        detections = []
        if r.boxes is not None:
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                cls_name = self.class_names[cls_id]
                
                # Assume standard PPE dataset classes: 0: hardhat, 1: vest, etc.
                # If falling back to yolov8n, these will just be COCO classes
                is_ppe = "hard" in cls_name.lower() or "vest" in cls_name.lower() or "glove" in cls_name.lower()
                
                # Green for PPE, Red for non-PPE (e.g. standard person with no gear detected)
                # This logic depends heavily on exactly how the custom dataset is labeled
                color = (0, 255, 0) if is_ppe else (255, 0, 0) 
                
                cv2.rectangle(annotated, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                
                label = f"{cls_name} {conf:.2f}"
                cv2.putText(annotated, label, (int(x1), int(y1) - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                            
                detections.append({
                    "bbox": (int(x1), int(y1), int(x2), int(y2)),
                    "class_name": cls_name,
                    "conf": conf,
                    "is_ppe": is_ppe
                })
                
        return annotated, detections
