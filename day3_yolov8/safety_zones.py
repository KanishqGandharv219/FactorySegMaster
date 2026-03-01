"""
FactorySegMaster Day 3: Safety Zone Logic (YOLOv8 Bbox Integration)

Defines rectangular safety zones around machines and checks
whether a YOLO bounding box (specifically its bottom-center) 
violates a zone boundary.

Zones use normalized coords (0.0 to 1.0).
"""

import cv2
import numpy as np


class SafetyZone:
    """A rectangular safety zone defined in normalized coords."""

    def __init__(self, x1, y1, x2, y2, label="Zone"):
        # Normalized 0-1 coordinates
        self.x1 = min(x1, x2)
        self.y1 = min(y1, y2)
        self.x2 = max(x1, x2)
        self.y2 = max(y1, y2)
        self.label = label

    def contains(self, px_norm, py_norm):
        """Check if a normalized point is inside this zone."""
        return (self.x1 <= px_norm <= self.x2 and
                self.y1 <= py_norm <= self.y2)

    def to_pixels(self, w, h):
        """Convert to pixel coordinates for a given image size."""
        return (
            int(self.x1 * w), int(self.y1 * h),
            int(self.x2 * w), int(self.y2 * h),
        )


def draw_zones(frame, zones):
    """
    Draw safety zones on a frame.
    """
    h, w = frame.shape[:2]

    for zone in zones:
        x1, y1, x2, y2 = zone.to_pixels(w, h)

        # Semi-transparent red fill
        overlay = frame.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (255, 0, 0), cv2.FILLED)
        cv2.addWeighted(overlay, 0.15, frame, 0.85, 0, frame)

        # Red border
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

        # Label
        label = f"[ZONE] {zone.label}"
        (tw, th), _ = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
        )
        # Background for the label text so it pops
        cv2.rectangle(
            frame, (x1, y1 - th - 8), (x1 + tw + 8, y1), (255, 0, 0), cv2.FILLED
        )
        cv2.putText(
            frame, label, (x1 + 4, y1 - 4),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA
        )

    return frame


def check_violations_px(objects, zones, frame_w, frame_h):
    """
    Check if the bottom-center of any YOLO bounding box is inside a zone.
    
    Why bottom-center? Because we want to trigger zones when a worker 
    physically steps into it, not when their head/arm waves over it.
    
    Args:
        objects: list of dicts {"bbox": (x1,y1,x2,y2), "class_name": str, ...}
        zones: list of SafetyZone objects (normalized coords)
        frame_w, frame_h: frame dimensions for normalization

    Returns:
        list of violation strings
    """
    violations = []

    for obj in objects:
        x1, y1, x2, y2 = obj["bbox"]
        cls_name = obj.get("class_name", "Object")
        
        # Calculate the bottom-center coordinate of the bounding box
        center_x = (x1 + x2) // 2
        bottom_y = y2 
        
        # Normalize the coordinate
        nx = center_x / frame_w if frame_w > 0 else 0
        ny = bottom_y / frame_h if frame_h > 0 else 0

        for zone in zones:
            if zone.contains(nx, ny):
                violations.append(
                    f"ALERT: '{cls_name}' entered {zone.label}!"
                )

    return violations


def draw_violations(frame, violations):
    """
    Draw violation alerts on the top-left of the frame.
    """
    if not violations:
        # Safe indicator
        cv2.putText(
            frame, "STATUS: ALL CLEAR", (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 0), 2, cv2.LINE_AA
        )
        return frame

    # Warning header
    cv2.putText(
        frame, f"STATUS: {len(violations)} VIOLATION(S)", (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2, cv2.LINE_AA
    )

    # List violations
    for i, v in enumerate(violations[:5]):  # limit on-screen text to 5 to avoid clutter
        y = 60 + i * 25
        # Red background for the text
        cv2.rectangle(frame, (10, y-15), (400, y+5), (0,0,0), cv2.FILLED)
        cv2.putText(
            frame, v, (12, y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (50, 50, 255), 1, cv2.LINE_AA
        )

    return frame
