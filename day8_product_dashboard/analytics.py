"""
FactorySegMaster Day 7: Final Analytics Engine
Calculates risk scores and maintains a scrolling event log based on 
factory floor detections.
"""

from collections import deque

class FactoryAnalytics:
    def __init__(self, log_capacity=15):
        # Keeps the last N events for the UI log
        self.event_log = deque(maxlen=log_capacity)
        
        # State counters
        self.total_workers = 0
        self.total_vehicles = 0
        self.safety_violations = 0
        self.current_risk_score = 0
        
        # Track which IDs have already triggered a log so we don't spam
        self.logged_violations = set()

    def update(self, tracking_results, ppe_detections):
        """
        Updates the analytics state based on the current frame's detections.
        """
        self.total_workers = 0
        self.total_vehicles = 0
        self.safety_violations = 0
        
        # Count basics from tracking
        for track in tracking_results:
            cls_name = track["class_name"].lower()
            if cls_name == "person":
                self.total_workers += 1
            elif cls_name in ["car", "truck", "bus", "motorcycle", "vehicle"]:
                self.total_vehicles += 1
                
        # Calculate violations from PPE engine
        # Detections look like: {"bbox": (x1, y1, x2, y2), "class_name": cls, "is_ppe": True/False}
        # In a real system, we map PPE boxes to worker boxes via IoU. For this demo,
        # we'll look at the raw PPE detections to see if any missing PPE was explicitly flagged.
        for ppe in ppe_detections:
            cls_name = ppe["class_name"]
            # If the model explicitly detects NO-Hardhat or NO-Mask or NO-Safety Vest
            if "no-" in cls_name.lower():
                self.safety_violations += 1
                # Add to log if it's a new occurrence (just use the class name as a simple key for demo)
                if cls_name not in self.logged_violations:
                    self.log_event(f"VIOLATION DETECTED: {cls_name}")
                    self.logged_violations.add(cls_name)
                    
        # Calculate Risk Score (0-100)
        # Highly arbitrary formula for demo purposes
        base_risk = 10
        worker_density_risk = min(50, self.total_workers * 2) 
        violation_risk = min(40, self.safety_violations * 20)
        
        self.current_risk_score = base_risk + worker_density_risk + violation_risk
        
    def log_event(self, message):
        """Add a timestamped event to the scrolling log."""
        import datetime
        now = datetime.datetime.now().strftime("%H:%M:%S")
        self.event_log.append(f"[{now}] {message}")
        
    def get_log_text(self):
        """Return the event log as a single string."""
        return "\n".join(list(self.event_log))
        
    def get_last_stats(self):
        """Return a dictionary of current stats for the API."""
        return {
            "worker_count": self.total_workers,
            "vehicle_count": self.total_vehicles,
            "violation_count": self.safety_violations,
            "risk_score": self.current_risk_score
        }

    def get_dashboard_stats(self):
        """Return a tuple of current stats for the UI (back-compat for demo.py)."""
        return (
            self.total_workers,
            self.total_vehicles,
            self.safety_violations,
            f"{self.current_risk_score}%"
        )
        
    def reset(self):
        """Clear the analytics state (e.g. when a new video is loaded)."""
        self.event_log.clear()
        self.logged_violations.clear()
        self.log_event("FactoryTwin Analytics Initialized.")
