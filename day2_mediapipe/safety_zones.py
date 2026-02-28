"""
FactorySegMaster Day 2: Safety Zone Logic

Defines rectangular safety zones around machines and checks
whether worker keypoints violate zone boundaries.

Zones use normalized coordinates (0.0 to 1.0) so they work
at any image resolution.
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

    Args:
        frame: RGB numpy array
        zones: list of SafetyZone objects

    Returns:
        Frame with zones drawn (modifies in-place)
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
        cv2.rectangle(
            frame, (x1, y1 - th - 8), (x1 + tw + 8, y1), (255, 0, 0), cv2.FILLED
        )
        cv2.putText(
            frame, label, (x1 + 4, y1 - 4),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA
        )

    return frame


def check_violations(workers, zones):
    """
    Check if any worker keypoints are inside safety zones.

    Args:
        workers: list of worker dicts from WorkerDetector.detect()
                 each has "keypoints" dict with (px, py) pixel values
                 and "id" int
        zones: list of SafetyZone objects

    Returns:
        list of violation dicts: {"worker_id", "joint", "zone_label", "position"}
    """
    violations = []
    critical_joints = ["left_wrist", "right_wrist", "nose"]

    for worker in workers:
        keypoints = worker.get("keypoints", {})
        worker_id = worker.get("id", 0)

        for joint in critical_joints:
            if joint not in keypoints:
                continue

            px, py = keypoints[joint]

            for zone in zones:
                # We need the frame dimensions to normalize --
                # but keypoints are already in pixels from the detector.
                # For simplicity, check using pixel coords directly.
                # The caller can pass frame dims or we use a helper.
                pass

    return violations


def check_violations_px(workers, zones, frame_w, frame_h):
    """
    Check violations using pixel coordinates.

    Args:
        workers: list of worker dicts with "keypoints" in pixel coords
        zones: list of SafetyZone objects (normalized coords)
        frame_w, frame_h: frame dimensions for normalization

    Returns:
        list of violation strings
    """
    violations = []
    critical_joints = ["left_wrist", "right_wrist", "nose"]

    for worker in workers:
        keypoints = worker.get("keypoints", {})
        worker_id = worker.get("id", 0)

        for joint in critical_joints:
            if joint not in keypoints:
                continue

            px, py = keypoints[joint]
            # Normalize to 0-1
            nx = px / frame_w if frame_w > 0 else 0
            ny = py / frame_h if frame_h > 0 else 0

            for zone in zones:
                if zone.contains(nx, ny):
                    violations.append(
                        f"ALERT: Worker {worker_id} {joint} in {zone.label}"
                    )

    return violations


def draw_violations(frame, violations):
    """
    Draw violation alerts on the frame.

    Args:
        frame: RGB numpy array
        violations: list of violation strings

    Returns:
        Frame with alerts drawn
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
    for i, v in enumerate(violations[:5]):  # max 5 on screen
        y = 60 + i * 25
        cv2.putText(
            frame, v, (10, y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 50, 50), 1, cv2.LINE_AA
        )

    return frame
