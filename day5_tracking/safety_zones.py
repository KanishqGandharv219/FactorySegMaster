"""
FactorySegMaster Day 5: Safety Zone Integration (Dwell Time)

Upgraded safety zone logic that keeps track of how long an object
(with a specific ID) has remained inside a hazardous area.
"""

import cv2
import time
import numpy as np

class SafetyZoneManager:
    def __init__(self, x1_norm, y1_norm, x2_norm, y2_norm, color=(255, 0, 0), max_dwell_seconds=3.0):
        """
        Hold state for a specific safety zone and its occupants.
        """
        self.x1_norm = x1_norm
        self.y1_norm = y1_norm
        self.x2_norm = x2_norm
        self.y2_norm = y2_norm
        self.color = color
        self.max_dwell_seconds = max_dwell_seconds
        
        # State tracking: {track_id: timestamp_entered}
        self.occupants = {}
        
    def set_zone(self, x1, y1, x2, y2):
        self.x1_norm = x1
        self.y1_norm = y1
        self.x2_norm = x2
        self.y2_norm = y2
        
    def set_dwell_time(self, seconds):
        self.max_dwell_seconds = seconds

    def process_frame(self, frame_rgb, tracked_objects, current_time=None):
        """
        Check all tracked objects against the zone.
        Updates internal dwell time state.
        
        Returns:
            list of strings: active violation alerts for this frame
        """
        if current_time is None:
            current_time = time.time()
            
        h, w = frame_rgb.shape[:2]
        zx1, zy1 = int(self.x1_norm * w), int(self.y1_norm * h)
        zx2, zy2 = int(self.x2_norm * w), int(self.y2_norm * h)
        
        # Draw the zone boundary
        overlay = frame_rgb.copy()
        cv2.rectangle(overlay, (zx1, zy1), (zx2, zy2), self.color, -1)
        # alpha blend
        frame_rgb = cv2.addWeighted(overlay, 0.2, frame_rgb, 0.8, 0)
        cv2.rectangle(frame_rgb, (zx1, zy1), (zx2, zy2), self.color, 2)
        
        current_frame_ids = []
        alerts = []
        
        for obj in tracked_objects:
            track_id = obj["id"]
            current_frame_ids.append(track_id)
            
            x1, y1, x2, y2 = obj["bbox"]
            # Bottom center point of bounding box represents feet/wheels
            bx, by = int((x1 + x2) / 2), int(y2)
            
            # Check if inside zone
            inside_x = zx1 <= bx <= zx2
            inside_y = zy1 <= by <= zy2
            
            if inside_x and inside_y:
                # Inside zone
                if track_id not in self.occupants:
                    # Just entered
                    self.occupants[track_id] = current_time
                    
                dwell_time = current_time - self.occupants[track_id]
                
                if dwell_time > self.max_dwell_seconds:
                    alerts.append(f"VIOLATION: {obj['class_name']} #{track_id} in zone for {dwell_time:.1f}s")
                    
                    # Optional: Draw a line from the zone center to the violating object
                    cv2.line(frame_rgb, (bx, by), (int((zx1+zx2)/2), int((zy1+zy2)/2)), (255, 0, 0), 2)
                    
            else:
                # Outside zone. If they were inside before, remove them.
                if track_id in self.occupants:
                    del self.occupants[track_id]
                    
        # Cleanup occupants that disappeared entirely from the tracker
        lost_ids = [tid for tid in self.occupants.keys() if tid not in current_frame_ids]
        for tid in lost_ids:
            del self.occupants[tid]
            
        return alerts, frame_rgb
