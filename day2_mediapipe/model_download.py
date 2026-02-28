"""
Model download utility for MediaPipe Tasks API.

Downloads PoseLandmarker and HandLandmarker .task model files
from Google's CDN on first run. Cached in day2_mediapipe/models/.
"""

import os
import urllib.request

MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")

MODELS = {
    "pose_landmarker_lite.task": (
        "https://storage.googleapis.com/mediapipe-models/"
        "pose_landmarker/pose_landmarker_lite/float16/latest/"
        "pose_landmarker_lite.task"
    ),
    "hand_landmarker.task": (
        "https://storage.googleapis.com/mediapipe-models/"
        "hand_landmarker/hand_landmarker/float16/latest/"
        "hand_landmarker.task"
    ),
}


def ensure_models():
    """Download model files if they don't exist. Returns dict of model paths."""
    os.makedirs(MODELS_DIR, exist_ok=True)
    paths = {}

    for filename, url in MODELS.items():
        local_path = os.path.join(MODELS_DIR, filename)
        paths[filename] = local_path

        if os.path.exists(local_path):
            continue

        print(f"Downloading {filename}...")
        try:
            urllib.request.urlretrieve(url, local_path)
            size_mb = os.path.getsize(local_path) / (1024 * 1024)
            print(f"  Saved: {local_path} ({size_mb:.1f} MB)")
        except Exception as e:
            print(f"  ERROR downloading {filename}: {e}")
            if os.path.exists(local_path):
                os.remove(local_path)
            raise

    return paths


if __name__ == "__main__":
    paths = ensure_models()
    for name, path in paths.items():
        exists = "OK" if os.path.exists(path) else "MISSING"
        print(f"  [{exists}] {name}: {path}")
