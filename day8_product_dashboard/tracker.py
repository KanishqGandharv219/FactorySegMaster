"""
FactorySegMaster Day 5: ByteTrack Object Tracking

Core tracker class wrapping ultralytics YOLO + ByteTrack.
Provides persistent IDs to detected objects across video frames.
"""

import os
from ultralytics import YOLO


class PersistentTracker:
    def __init__(self, model_size="yolov8n.pt", conf_thresh=0.25, iou_thresh=0.45):
        """
        Initialize the YOLO tracker.
        """
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh

        abs_model_path = os.path.join(os.path.dirname(__file__), model_size)
        self.model = YOLO(abs_model_path if os.path.exists(abs_model_path) else model_size)

        # 0: person, 2: car, 7: truck
        self.target_classes = [0, 2, 7]
        self.class_names = self.model.names

    def set_thresholds(self, conf, iou):
        """Update detection thresholds interactively."""
        self.conf_thresh = conf
        self.iou_thresh = iou

    def _run_tracking(self, frame_rgb, target_classes):
        return self.model.track(
            source=frame_rgb,
            conf=self.conf_thresh,
            iou=self.iou_thresh,
            classes=target_classes,
            tracker="bytetrack.yaml",
            persist=True,
            verbose=False,
        )

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

        try:
            results = self._run_tracking(frame_rgb, target_classes)
        except TypeError as exc:
            # Ultralytics can keep a stale predictor tracker state between jobs.
            if "nonetype" in str(exc).lower() and "subscriptable" in str(exc).lower():
                self.reset()
                results = self._run_tracking(frame_rgb, target_classes)
            else:
                raise

        r = results[0]
        annotated = r.plot()

        tracked_objects = []
        if r.boxes is not None and r.boxes.id is not None:
            for i, box in enumerate(r.boxes):
                if r.boxes.id is None or i >= len(r.boxes.id):
                    continue

                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                cls_name = self.class_names[cls_id]
                track_id = int(r.boxes.id[i].item())

                tracked_objects.append(
                    {
                        "id": track_id,
                        "bbox": (int(x1), int(y1), int(x2), int(y2)),
                        "class_id": cls_id,
                        "class_name": cls_name,
                        "conf": conf,
                    }
                )

        return {
            "annotated": annotated,
            "objects": tracked_objects,
            "count": len(tracked_objects),
        }

    def reset(self):
        """
        Clear internal tracker state before a new image/video job.
        """
        # Recreate predictor on next .track() call to avoid stale tracker internals.
        if hasattr(self.model, "predictor"):
            self.model.predictor = None
