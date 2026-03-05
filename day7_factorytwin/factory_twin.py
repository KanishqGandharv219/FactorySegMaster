"""
FactorySegMaster Day 7: Master Orchestrator
Combines ByteTrack (Day 5), SAM2 (Day 4), and Custom PPE (Day 6) into a single pipeline.
"""

from tracker import PersistentTracker
from ppe_detector import PPEDetector
from sam2_segmenter import SAM2Segmenter
from analytics import FactoryAnalytics
import cv2
import numpy as np
import time

class FactoryTwin:
    def __init__(self):
        print("Initializing FactoryTwin Ensemble...")
        # 1. Base Object Detection & Tracking (YOLOv8 + ByteTrack)
        self.tracker = PersistentTracker(conf_thresh=0.35)
        
        # 2. Custom PPE Detection (Fine-tuned YOLOv8)
        # Note: If no weights are found, PPEDetector falls back to yolov8n internally
        self.ppe_detector = PPEDetector(conf_thresh=0.4)
        
        # 3. High-Precision Segmentation (SAM2)
        self.sam2 = SAM2Segmenter(model_size="sam2.1_t.pt")
        
        # 4. Analytics Dashboard
        self.analytics = FactoryAnalytics()
        
    def process_frame(self, frame_bgr, enable_ppe=True, enable_sam2=False):
        """
        Runs the full ensemble pipeline on a single frame.
        """
        start_time = time.time()
        
        # OpenCV uses BGR natively, Gradio passes RGB if image, BGR if video usually.
        # We assume frame_bgr is BGR from OpenCV VideoCapture or similar wrapper.
        
        display_frame = frame_bgr.copy()
        
        # --- 1. Base Tracking ---
        tracker_result = self.tracker.track_frame(display_frame)
        tracked_objects = tracker_result["objects"]
        
        # --- 2. PPE Detection ---
        ppe_detections = []
        if enable_ppe:
            # For simplicity in this demo, we run PPE detector on the whole frame
            # In a heavy production system, we'd crop each tracked 'Person' bbox and run PPE on the crop.
            _, ppe_detections = self.ppe_detector.detect_ppe(display_frame)
            
        # --- 3. Analytics Update ---
        self.analytics.update(tracked_objects, ppe_detections)
        
        # --- 4. Drawing & SAM2 Masks ---
        # Draw PPE boxes (they have color coding for compliance)
        if enable_ppe:
            for ppe in ppe_detections:
                x1, y1, x2, y2 = ppe["bbox"]
                color = (0, 255, 0) if ppe["is_ppe"] else (0, 0, 255)
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
                label = f"{ppe['class_name']}"
                cv2.putText(display_frame, label, (x1, max(10, y1 - 10)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                            
        # Draw Tracked Objects (Persons, Vehicles)
        for obj in tracked_objects:
            x1, y1, x2, y2 = obj["bbox"]
            track_id = obj["id"]
            cls_name = obj["class_name"]
            
            # Skip drawing standard BBox if PPE is enabled and it's a Person, 
            # to avoid cluttering the worker with too many boxes. We just want the ID.
            if enable_ppe and cls_name.lower() == "person":
                cv2.putText(display_frame, f"ID:{track_id}", (x1, y2 + 20), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            else:
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), (255, 165, 0), 2)
                cv2.putText(display_frame, f"ID:{track_id} {cls_name}", (x1, max(10, y1 - 10)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 165, 0), 2)
                            
            # SAM2 Segmentation (Only segmenting Persons for this demo feature)
            if enable_sam2 and cls_name.lower() == "person":
                # Give SAM2 the bounding box prompt
                sam_result = self.sam2.segment_from_box(display_frame, int(x1), int(y1), int(x2), int(y2))
                mask = sam_result.get("mask")
                if mask is not None:
                    # Apply semi-transparent blue mask overlay
                    colored_mask = np.zeros_like(display_frame, dtype=np.uint8)
                    colored_mask[mask > 0] = [255, 0, 0] # Blue in BGR
                    display_frame = cv2.addWeighted(display_frame, 1.0, colored_mask, 0.4, 0)

        # FPS calculation
        fps = 1.0 / (time.time() - start_time + 1e-5)
        
        # Draw Dashboard Overlay
        self._draw_dashboard(display_frame, fps)
        
        return display_frame
        
    def _draw_dashboard(self, frame, fps):
        """Draws the transparent analytics dashboard on the top-right."""
        workers, vehicles, violations, risk = self.analytics.get_dashboard_stats()
        
        # Dashboard parameters
        h, w = frame.shape[:2]
        dash_w, dash_h = 300, 180
        x1, y1 = w - dash_w - 20, 20
        x2, y2 = w - 20, 20 + dash_h
        
        # Transparent background
        overlay = frame.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        
        # Text settings
        font = cv2.FONT_HERSHEY_SIMPLEX
        text_x = x1 + 15
        
        cv2.putText(frame, "FACTORY TWIN DASHBOARD", (text_x, y1 + 30), font, 0.6, (255, 255, 255), 2)
        cv2.line(frame, (text_x, y1 + 40), (x2 - 15, y1 + 40), (255, 255, 255), 1)
        
        cv2.putText(frame, f"FPS: {fps:.1f}", (text_x, y1 + 65), font, 0.5, (0, 255, 0), 1)
        cv2.putText(frame, f"Active Workers: {workers}", (text_x, y1 + 90), font, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, f"Active Vehicles: {vehicles}", (text_x, y1 + 115), font, 0.5, (255, 255, 255), 1)
        
        risk_color = (0, 0, 255) if int(risk.replace("%", "")) > 50 else (0, 255, 255)
        cv2.putText(frame, f"PPE Violations: {violations}", (text_x, y1 + 140), font, 0.5, risk_color, 1)
        cv2.putText(frame, f"SYSTEM RISK: {risk}", (text_x, y1 + 165), font, 0.6, risk_color, 2)
