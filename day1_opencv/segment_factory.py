"""
FactorySegMaster Day 1: Factory Floor Object Segmentation

Key insight: Factory machines may have SIMILAR intensity to the floor.
Pure thresholding won't work. We use edges + texture + gradient approaches.

Strategies:
  1. Edge: Canny edges -> flood-fill closed regions -> filter
  2. Gradient: Morphological gradient (highlights object boundaries as thick edges)
  3. Adaptive: Local adaptive thresholding (good for texture differences)
  4. Otsu: Global auto-threshold (only works if machines differ in brightness)
  5. Combined: Union of edge + gradient masks (catches more objects)
"""

import cv2
import numpy as np
from dataclasses import dataclass


@dataclass
class SegmentationResult:
    """Container for segmentation outputs."""
    raw_mask: np.ndarray      # Raw mask BEFORE contour filtering (for debug)
    mask: np.ndarray          # Clean mask (filtered contours only)
    annotated: np.ndarray     # Annotated image (RGB)
    contours: list
    stats: dict


def segment_factory_objects(
    image,
    min_area: int = 500,
    max_area_pct: float = 30.0,
    min_convexity: float = 0.2,
    mode: str = "adaptive",
    blur_size: int = 5,
    canny_low: int = 20,
    canny_high: int = 80,
    adaptive_block: int = 51,
    adaptive_c: int = 8,
):
    """
    Segment factory objects from an image.

    Returns SegmentationResult with raw_mask (debug), clean mask, and annotated image.
    """
    if isinstance(image, str):
        img = cv2.imread(image)
        if img is None:
            raise FileNotFoundError(f"Could not load image: {image}")
    else:
        img = image.copy()

    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    img_area = h * w
    max_area = int(img_area * max_area_pct / 100.0)

    blur_k = blur_size if blur_size % 2 == 1 else blur_size + 1
    adapt_k = max(adaptive_block if adaptive_block % 2 == 1 else adaptive_block + 1, 3)
    blurred = cv2.GaussianBlur(gray, (blur_k, blur_k), 0)

    # --- Strategy 1: Edge-based with contour fill ---
    def _edge_mask():
        edges = cv2.Canny(blurred, canny_low, canny_high)
        # Very gentle close — just bridge tiny gaps
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        edges = cv2.dilate(edges, k, iterations=1)
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, k, iterations=1)
        # Find and fill contours from edges
        cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        filled = np.zeros_like(gray)
        for c in cnts:
            if cv2.contourArea(c) > 200:  # fill anything non-trivial
                cv2.drawContours(filled, [c], -1, 255, cv2.FILLED)
        return filled

    # --- Strategy 2: Morphological gradient (highlights all boundaries) ---
    def _gradient_mask():
        # Gradient = dilation - erosion = thick edges around objects
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        gradient = cv2.morphologyEx(blurred, cv2.MORPH_GRADIENT, k)
        # Threshold the gradient
        _, thresh = cv2.threshold(gradient, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # Close to form complete boundaries, then fill
        k2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, k2, iterations=2)
        # Find and fill contours
        cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        filled = np.zeros_like(gray)
        for c in cnts:
            if cv2.contourArea(c) > 200:
                cv2.drawContours(filled, [c], -1, 255, cv2.FILLED)
        return filled

    # --- Strategy 3: Adaptive threshold ---
    def _adaptive_mask():
        thresh = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, adapt_k, adaptive_c
        )
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, k, iterations=2)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, k, iterations=1)
        return thresh

    # --- Strategy 4: Otsu (both normal and inverted, pick better) ---
    def _otsu_mask():
        _, thresh_inv = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        _, thresh_norm = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        # Clean both
        t_inv = cv2.morphologyEx(thresh_inv, cv2.MORPH_OPEN, k, iterations=1)
        t_norm = cv2.morphologyEx(thresh_norm, cv2.MORPH_OPEN, k, iterations=1)
        # Pick the one with less coverage (more likely to be foreground objects)
        cov_inv = np.count_nonzero(t_inv) / img_area
        cov_norm = np.count_nonzero(t_norm) / img_area
        # We expect objects to cover <50% of image
        if abs(cov_inv - 0.3) < abs(cov_norm - 0.3):
            return t_inv
        else:
            return t_norm

    # Select strategy
    if mode == "edge":
        raw_mask = _edge_mask()
    elif mode == "gradient":
        raw_mask = _gradient_mask()
    elif mode == "adaptive":
        raw_mask = _adaptive_mask()
    elif mode == "otsu":
        raw_mask = _otsu_mask()
    else:  # combined = union of ALL strategies
        m1 = _edge_mask()
        m2 = _gradient_mask()
        m3 = _adaptive_mask()
        m4 = _otsu_mask()
        raw_mask = cv2.bitwise_or(m1, m2)
        raw_mask = cv2.bitwise_or(raw_mask, m3)
        raw_mask = cv2.bitwise_or(raw_mask, m4)

    # Re-find contours on the raw mask for filtering
    contours_raw, _ = cv2.findContours(raw_mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter
    filtered = []
    rejected = {"too_small": 0, "too_large": 0, "low_convexity": 0}

    for cnt in contours_raw:
        area = cv2.contourArea(cnt)
        if area < min_area:
            rejected["too_small"] += 1
            continue
        if area > max_area:
            rejected["too_large"] += 1
            continue

        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        if hull_area == 0:
            continue
        convexity = area / hull_area
        if convexity < min_convexity:
            rejected["low_convexity"] += 1
            continue

        filtered.append(cnt)

    # Clean mask
    clean_mask = np.zeros_like(gray)
    cv2.drawContours(clean_mask, filtered, -1, 255, cv2.FILLED)

    # Annotate
    annotated = img.copy()
    colors = [
        (0, 255, 0), (255, 100, 0), (0, 200, 255), (255, 0, 255),
        (100, 255, 100), (255, 255, 0), (0, 128, 255), (255, 128, 0),
        (128, 0, 255), (64, 255, 128), (255, 64, 128), (128, 64, 255),
    ]
    total_area = 0

    for i, cnt in enumerate(filtered):
        color = colors[i % len(colors)]
        area = cv2.contourArea(cnt)
        total_area += area

        overlay = annotated.copy()
        cv2.drawContours(overlay, [cnt], -1, color, cv2.FILLED)
        cv2.addWeighted(overlay, 0.3, annotated, 0.7, 0, annotated)
        cv2.drawContours(annotated, [cnt], -1, color, 2)

        x, y, bw, bh = cv2.boundingRect(cnt)
        hull = cv2.convexHull(cnt)
        ha = cv2.contourArea(hull)
        cvx = area / ha if ha > 0 else 0
        label = f"Obj {i+1} | {area:,.0f}px | cvx:{cvx:.2f}"

        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
        ly = max(y - 4, th + 6)
        cv2.rectangle(annotated, (x, ly - th - 4), (x + tw + 4, ly + 2), (0, 0, 0), cv2.FILLED)
        cv2.putText(annotated, label, (x + 2, ly - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)
        cv2.rectangle(annotated, (x, y), (x + bw, y + bh), color, 2)

    annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

    stats = {
        "objects_detected": len(filtered),
        "total_object_area_px": int(total_area),
        "coverage_pct": round(total_area / img_area * 100, 1) if img_area > 0 else 0,
        "image_size": f"{w}x{h}",
        "mode": mode,
        "min_area": min_area,
        "max_area_pct": max_area_pct,
        "min_convexity": min_convexity,
        "contours_before_filter": len(contours_raw),
        "rejected": rejected,
        "raw_mask_coverage_pct": round(np.count_nonzero(raw_mask) / img_area * 100, 1),
    }

    return SegmentationResult(
        raw_mask=raw_mask,
        mask=clean_mask,
        annotated=annotated_rgb,
        contours=filtered,
        stats=stats,
    )


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "factory_sample.jpg"
    try:
        result = segment_factory_objects(path)
        s = result.stats
        print(f"Detected {s['objects_detected']} objects (from {s['contours_before_filter']} raw)")
        print(f"Raw mask coverage: {s['raw_mask_coverage_pct']}%")
        print(f"Clean coverage: {s['coverage_pct']}%")
        print(f"Rejected: {s['rejected']}")
        cv2.imwrite("debug_raw_mask.jpg", result.raw_mask)
        cv2.imwrite("segmented_mask.jpg", result.mask)
        cv2.imwrite("segmented_annotated.jpg",
                    cv2.cvtColor(result.annotated, cv2.COLOR_RGB2BGR))
    except FileNotFoundError as e:
        print(f"Error: {e}")
