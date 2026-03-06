"""
FactorySegMaster Day 4: SAM2 Core Engine

Wraps Ultralytics SAM2 (Segment Anything Model 2) to provide
zero-shot pixel-perfect masking from point clicks or bounding boxes.
"""

import cv2
import numpy as np
import os
from ultralytics import SAM

class SAM2Segmenter:
    def __init__(self, model_size="sam2.1_t.pt"):
        """
        Initialize the SAM2 Segmenter.
        Downloads the nanoweight model automatically on first run.
        """
        # Use absolute path for model weights
        abs_model_path = os.path.join(os.path.dirname(__file__), model_size)
        self.model = SAM(abs_model_path if os.path.exists(abs_model_path) else model_size)

    def segment_from_point(self, frame_rgb, px, py):
        """
        Generate a mask containing the object at the clicked point (px, py).
        
        Args:
            frame_rgb: numpy array (H, W, 3) RGB image
            px, py: integer pixel coordinates of the prompt
            
        Returns:
            dict containing:
                "mask": 2D boolean numpy array (H, W) or None
                "annotated": RGB image with mask overlay
                "isolated": RGB image with background blacked out
        """
        # Format point for Ultralytics SAM prompt: [x, y]
        # SAM expects points as a list of lists, and labels (1 = positive click)
        results = self.model(
            source=frame_rgb,
            points=[[px, py]],
            labels=[1],
            verbose=False
        )
        
        r = results[0]
        return self._process_result(frame_rgb, r)

    def segment_from_box(self, frame_rgb, x1, y1, x2, y2):
        """
        Generate a mask containing the object within the bounding box.
        
        Args:
            frame_rgb: numpy array (H, W, 3) RGB image
            x1, y1, x2, y2: integer coordinates of bounding box
            
        Returns:
            dict containing:
                "mask": 2D boolean numpy array (H, W) or None
                "annotated": RGB image with mask overlay
                "isolated": RGB image with background blacked out
        """
        # Format box for Ultralytics SAM prompt
        results = self.model(
            source=frame_rgb,
            bboxes=[[x1, y1, x2, y2]],
            verbose=False
        )
        
        r = results[0]
        return self._process_result(frame_rgb, r)

    def _process_result(self, frame_rgb, result_obj):
        """Helper to extract mask and generate visual overlays."""
        if result_obj.masks is None or len(result_obj.masks) == 0:
            # Fallback if SAM fails to find anything
            return {
                "mask": None,
                "annotated": frame_rgb.copy(),
                "isolated": np.zeros_like(frame_rgb)
            }

        # Ultralytics masks are shape (N, H, W). We take the first one since we prompt 1 object.
        # Convert to boolean numpy array
        mask_data = result_obj.masks.data[0].cpu().numpy()
        
        # Sometimes dimensions get padded. Resize back to original image size if necessary
        h, w = frame_rgb.shape[:2]
        if mask_data.shape[0] != h or mask_data.shape[1] != w:
            mask_data = cv2.resize(mask_data, (w, h), interpolation=cv2.INTER_NEAREST)
            
        bool_mask = mask_data > 0.5

        # 1) Annotated: original image + semi-transparent green overlay
        annotated = frame_rgb.copy()
        color = np.array([0, 255, 0], dtype=np.uint8) # Green
        
        # Apply color where mask is true
        annotated[bool_mask] = annotated[bool_mask] * 0.5 + color * 0.5

        # Optional: draw outline contour
        contours, _ = cv2.findContours(bool_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(annotated, contours, -1, (0, 255, 0), 2)

        # 2) Isolated: Original object + pure black background
        isolated = np.zeros_like(frame_rgb)
        isolated[bool_mask] = frame_rgb[bool_mask]

        return {
            "mask": bool_mask,
            "annotated": annotated,
            "isolated": isolated
        }

