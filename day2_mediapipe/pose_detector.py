"""
FactorySegMaster Day 2: Worker Pose and Hand Detection

Uses MediaPipe Tasks API (PoseLandmarker + HandLandmarker) for
multi-person best-effort detection on factory CCTV frames.

Note: PoseLandmarker multi-person (`num_poses > 1`) is officially
"out of scope" for the model. Works reasonably for 2-3 workers
but may have landmark-switching with more. Day 3 YOLOv8 fixes this.
"""

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision as mp_vision

from model_download import ensure_models

# Pose connections for drawing skeletons
POSE_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 7),    # face
    (0, 4), (4, 5), (5, 6), (6, 8),    # face
    (9, 10),                             # mouth
    (11, 12),                            # shoulders
    (11, 13), (13, 15),                  # left arm
    (12, 14), (14, 16),                  # right arm
    (11, 23), (12, 24),                  # torso
    (23, 24),                            # hips
    (23, 25), (25, 27),                  # left leg
    (24, 26), (26, 28),                  # right leg
    (27, 29), (29, 31),                  # left foot
    (28, 30), (30, 32),                  # right foot
]

# Per-worker colors (BGR for cv2)
WORKER_COLORS = [
    (0, 255, 0),    # green
    (255, 100, 0),  # blue
    (0, 200, 255),  # yellow
    (255, 0, 255),  # magenta
    (0, 255, 255),  # cyan
]

# Key joint indices
KEY_JOINTS = {
    "nose": 0,
    "left_shoulder": 11,
    "right_shoulder": 12,
    "left_wrist": 15,
    "right_wrist": 16,
    "left_hip": 23,
    "right_hip": 24,
}


class WorkerDetector:
    """Multi-person pose and hand detector using MediaPipe Tasks API."""

    def __init__(self, min_confidence=0.6, num_poses=5, enable_hands=True):
        self.min_confidence = min_confidence
        self.num_poses = num_poses
        self.enable_hands = enable_hands

        # Download models if needed
        model_paths = ensure_models()

        # Create PoseLandmarker
        pose_options = mp_vision.PoseLandmarkerOptions(
            base_options=mp_tasks.BaseOptions(
                model_asset_path=model_paths["pose_landmarker_lite.task"]
            ),
            running_mode=mp_vision.RunningMode.IMAGE,
            num_poses=num_poses,
            min_pose_detection_confidence=min_confidence,
            min_tracking_confidence=0.5,
        )
        self.pose_landmarker = mp_vision.PoseLandmarker.create_from_options(
            pose_options
        )

        # Create HandLandmarker (optional)
        self.hand_landmarker = None
        if enable_hands:
            hand_options = mp_vision.HandLandmarkerOptions(
                base_options=mp_tasks.BaseOptions(
                    model_asset_path=model_paths["hand_landmarker.task"]
                ),
                running_mode=mp_vision.RunningMode.IMAGE,
                num_hands=4,
                min_hand_detection_confidence=min_confidence,
            )
            self.hand_landmarker = mp_vision.HandLandmarker.create_from_options(
                hand_options
            )

    def detect(self, frame_rgb):
        """
        Run pose + hand detection on a single RGB frame.

        Args:
            frame_rgb: numpy array (H, W, 3), RGB uint8

        Returns:
            dict with keys:
                "annotated": annotated frame (RGB)
                "workers": list of worker dicts, each with "keypoints" dict
                "hands": list of (x, y) wrist positions
                "num_workers": int
                "num_hands": int
        """
        h, w = frame_rgb.shape[:2]

        # Convert to MediaPipe Image
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=frame_rgb,
        )

        # Pose detection
        pose_result = self.pose_landmarker.detect(mp_image)
        annotated = frame_rgb.copy()

        workers = []
        for i, landmarks in enumerate(pose_result.pose_landmarks):
            color = WORKER_COLORS[i % len(WORKER_COLORS)]

            # Extract key joints as pixel coords
            keypoints = {}
            for name, idx in KEY_JOINTS.items():
                if idx < len(landmarks):
                    lm = landmarks[idx]
                    if lm.visibility > 0.3:
                        keypoints[name] = (int(lm.x * w), int(lm.y * h))

            workers.append({"keypoints": keypoints, "color": color, "id": i + 1})

            # Draw skeleton
            points = {}
            for idx_val, lm in enumerate(landmarks):
                px, py = int(lm.x * w), int(lm.y * h)
                points[idx_val] = (px, py)
                if lm.visibility > 0.3:
                    cv2.circle(annotated, (px, py), 4, color, -1)

            for start, end in POSE_CONNECTIONS:
                if start in points and end in points:
                    s_lm = landmarks[start]
                    e_lm = landmarks[end]
                    if s_lm.visibility > 0.3 and e_lm.visibility > 0.3:
                        cv2.line(annotated, points[start], points[end], color, 2)

            # Worker label
            if "nose" in keypoints:
                nx, ny = keypoints["nose"]
                label = f"Worker {i + 1}"
                (tw, th), _ = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
                )
                cv2.rectangle(
                    annotated,
                    (nx - tw // 2 - 4, ny - 30 - th),
                    (nx + tw // 2 + 4, ny - 26),
                    (0, 0, 0),
                    cv2.FILLED,
                )
                cv2.putText(
                    annotated,
                    label,
                    (nx - tw // 2, ny - 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    2,
                    cv2.LINE_AA,
                )

        # Hand detection
        hands = []
        if self.hand_landmarker is not None:
            hand_result = self.hand_landmarker.detect(mp_image)
            for hand_landmarks in hand_result.hand_landmarks:
                # Draw hand skeleton
                hand_points = {}
                for idx_val, lm in enumerate(hand_landmarks):
                    px, py = int(lm.x * w), int(lm.y * h)
                    hand_points[idx_val] = (px, py)
                    cv2.circle(annotated, (px, py), 2, (255, 165, 0), -1)

                # Draw hand connections
                hand_conns = mp.solutions.hands.HAND_CONNECTIONS
                for conn in hand_conns:
                    s, e = conn
                    if s in hand_points and e in hand_points:
                        cv2.line(
                            annotated, hand_points[s], hand_points[e],
                            (255, 165, 0), 1
                        )

                # Wrist position
                wrist = hand_landmarks[0]
                hands.append((int(wrist.x * w), int(wrist.y * h)))

        return {
            "annotated": annotated,
            "workers": workers,
            "hands": hands,
            "num_workers": len(workers),
            "num_hands": len(hands),
        }

    def close(self):
        """Release MediaPipe resources."""
        self.pose_landmarker.close()
        if self.hand_landmarker is not None:
            self.hand_landmarker.close()
