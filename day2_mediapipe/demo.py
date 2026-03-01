"""
FactorySegMaster Day 2: Gradio Demo

Worker pose detection + safety zone monitoring.
Supports both image upload and video (frame-by-frame) processing.
"""

# --- Monkey-patch for gradio_client JSON schema bug ---
import gradio_client.utils as _gc_utils

_original_get_type = _gc_utils.get_type

def _patched_get_type(schema):
    if isinstance(schema, bool):
        return "bool"
    return _original_get_type(schema)

_gc_utils.get_type = _patched_get_type

_original_json_schema = _gc_utils._json_schema_to_python_type

def _patched_json_schema(schema, defs=None):
    if isinstance(schema, bool):
        return "Any"
    return _original_json_schema(schema, defs)

_gc_utils._json_schema_to_python_type = _patched_json_schema
# --- End monkey-patch ---

import cv2
import gradio as gr
import numpy as np
import tempfile
import os
from PIL import Image

from pose_detector import WorkerDetector
from safety_zones import SafetyZone, draw_zones, check_violations_px, draw_violations


# Global detector (created once, reused)
_detector = None
_detector_confidence = None


def get_detector(confidence=0.6):
    """Get or create detector. Re-creates if confidence changes."""
    global _detector, _detector_confidence
    if _detector is None or _detector_confidence != confidence:
        if _detector is not None:
            _detector.close()
        _detector = WorkerDetector(
            min_confidence=confidence,
            num_poses=5,
            enable_hands=True,
        )
        _detector_confidence = confidence
    return _detector


def build_zones(z1_x1, z1_y1, z1_x2, z1_y2, z1_label,
                z2_x1, z2_y1, z2_x2, z2_y2, z2_label):
    """Build zone objects from slider values. Skips zones with zero area."""
    zones = []
    if abs(z1_x2 - z1_x1) > 0.01 and abs(z1_y2 - z1_y1) > 0.01:
        zones.append(SafetyZone(z1_x1, z1_y1, z1_x2, z1_y2, z1_label))
    if abs(z2_x2 - z2_x1) > 0.01 and abs(z2_y2 - z2_y1) > 0.01:
        zones.append(SafetyZone(z2_x1, z2_y1, z2_x2, z2_y2, z2_label))
    return zones


def format_stats(result, violations, zones):
    """Format detection stats as a text string."""
    lines = [
        f"Workers Detected: {result['num_workers']}",
        f"Hands Detected: {result['num_hands']}",
        f"Safety Zones: {len(zones)}",
        f"Violations: {len(violations)}",
    ]
    if violations:
        lines.append("")
        for v in violations:
            lines.append(f"  >> {v}")
    elif result["num_workers"] > 0:
        lines.append("")
        lines.append("  All workers in safe positions.")

    # Per-worker details
    if result["workers"]:
        lines.append("")
        lines.append("Worker Details:")
        for w in result["workers"]:
            kp = w["keypoints"]
            joints_visible = len(kp)
            lines.append(f"  Worker {w['id']}: {joints_visible} joints visible")

    return "\n".join(lines)


# --- Image Processing ---

def process_image(image, confidence, show_hands,
                  z1_x1, z1_y1, z1_x2, z1_y2, z1_label,
                  z2_x1, z2_y1, z2_x2, z2_y2, z2_label):
    """Process a single image."""
    if image is None:
        return None, None, "Upload a factory image to get started."

    detector = get_detector(confidence)
    frame_rgb = np.array(image)
    h, w = frame_rgb.shape[:2]

    # Detect
    result = detector.detect(frame_rgb)

    # Zones
    zones = build_zones(z1_x1, z1_y1, z1_x2, z1_y2, z1_label,
                        z2_x1, z2_y1, z2_x2, z2_y2, z2_label)

    # Check violations
    violations = check_violations_px(result["workers"], zones, w, h)

    # Annotated frame with zones + violations
    zone_frame = result["annotated"].copy()
    draw_zones(zone_frame, zones)
    draw_violations(zone_frame, violations)

    # Stats
    stats = format_stats(result, violations, zones)

    return (
        Image.fromarray(result["annotated"]),
        Image.fromarray(zone_frame),
        stats,
    )


# --- Video Processing ---

def process_video(video_path, confidence, show_hands,
                  z1_x1, z1_y1, z1_x2, z1_y2, z1_label,
                  z2_x1, z2_y1, z2_x2, z2_y2, z2_label):
    """Process a video file frame-by-frame."""
    if video_path is None:
        return None, "Upload a factory video."

    detector = get_detector(confidence)
    zones = build_zones(z1_x1, z1_y1, z1_x2, z1_y2, z1_label,
                        z2_x1, z2_y1, z2_x2, z2_y2, z2_label)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None, "Error: Could not open video file."

    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Output video
    out_path = tempfile.mktemp(suffix=".mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))

    frame_count = 0
    total_workers = 0
    total_violations = 0
    max_workers_in_frame = 0

    # Process every 2nd frame for speed, duplicate for smooth playback
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # Process every 2nd frame
        if frame_count % 2 == 0:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = detector.detect(frame_rgb)
            violations = check_violations_px(result["workers"], zones, w, h)

            out_frame = result["annotated"].copy()
            draw_zones(out_frame, zones)
            draw_violations(out_frame, violations)

            total_workers += result["num_workers"]
            total_violations += len(violations)
            max_workers_in_frame = max(max_workers_in_frame, result["num_workers"])

            # Store for duplication
            last_out = cv2.cvtColor(out_frame, cv2.COLOR_RGB2BGR)
            writer.write(last_out)
        else:
            if "last_out" in dir():
                writer.write(last_out)
            else:
                writer.write(frame)

    cap.release()
    writer.release()

    processed = frame_count // 2
    stats_lines = [
        f"Video Processing Complete",
        f"Frames: {total_frames} total, {processed} processed",
        f"FPS: {fps:.1f}",
        f"Resolution: {w}x{h}",
        f"Max Workers in Frame: {max_workers_in_frame}",
        f"Total Violation Events: {total_violations}",
        f"Avg Workers/Frame: {total_workers / max(processed, 1):.1f}",
    ]

    return out_path, "\n".join(stats_lines)


# --- Gradio UI ---

with gr.Blocks(
    title="FactorySegMaster Day 2",
    theme=gr.themes.Soft(),
) as demo:
    gr.Markdown(
        "# FactorySegMaster Day 2: MediaPipe Worker Detection\n"
        "Detect workers via pose estimation. Define safety zones and get violation alerts.\n\n"
        "**Note:** Multi-person detection uses best-effort mode (works for 2-3 workers). "
        "Day 3 YOLOv8 gives proper multi-person support."
    )

    with gr.Row():
        # Controls column
        with gr.Column(scale=1):
            confidence = gr.Slider(
                0.3, 0.9, value=0.6, step=0.05,
                label="Detection Confidence",
                info="Higher = fewer false positives, may miss some workers",
            )
            show_hands = gr.Checkbox(label="Enable Hand Tracking", value=True)

            gr.Markdown("### Safety Zone 1")
            with gr.Row():
                z1_x1 = gr.Slider(0.0, 1.0, value=0.05, step=0.01, label="X1")
                z1_y1 = gr.Slider(0.0, 1.0, value=0.3, step=0.01, label="Y1")
            with gr.Row():
                z1_x2 = gr.Slider(0.0, 1.0, value=0.35, step=0.01, label="X2")
                z1_y2 = gr.Slider(0.0, 1.0, value=0.8, step=0.01, label="Y2")
            z1_label = gr.Textbox(value="Machine Zone A", label="Label")

            gr.Markdown("### Safety Zone 2")
            with gr.Row():
                z2_x1 = gr.Slider(0.0, 1.0, value=0.6, step=0.01, label="X1")
                z2_y1 = gr.Slider(0.0, 1.0, value=0.3, step=0.01, label="Y1")
            with gr.Row():
                z2_x2 = gr.Slider(0.0, 1.0, value=0.95, step=0.01, label="X2")
                z2_y2 = gr.Slider(0.0, 1.0, value=0.8, step=0.01, label="Y2")
            z2_label = gr.Textbox(value="Machine Zone B", label="Label")

        # Output column
        with gr.Column(scale=2):
            with gr.Tabs():
                with gr.TabItem("Image Mode"):
                    image_input = gr.Image(
                        type="pil", label="Factory CCTV Frame",
                        sources=["upload", "clipboard"],
                    )
                    img_btn = gr.Button("Detect Workers", variant="primary")
                    with gr.Row():
                        pose_out = gr.Image(label="Pose Detection", type="pil")
                        zone_out = gr.Image(label="Safety Zone Monitor", type="pil")
                    img_stats = gr.Textbox(label="Stats", lines=10)

                with gr.TabItem("Video Mode"):
                    video_input = gr.Video(label="Factory Video")
                    vid_btn = gr.Button("Process Video", variant="primary")
                    video_out = gr.Video(label="Annotated Video")
                    vid_stats = gr.Textbox(label="Video Stats", lines=8)

    # Zone slider inputs (shared across image/video)
    zone_inputs = [
        z1_x1, z1_y1, z1_x2, z1_y2, z1_label,
        z2_x1, z2_y1, z2_x2, z2_y2, z2_label,
    ]

    # Image mode
    img_inputs = [image_input, confidence, show_hands] + zone_inputs
    img_outputs = [pose_out, zone_out, img_stats]
    img_btn.click(fn=process_image, inputs=img_inputs, outputs=img_outputs)

    # Video mode
    vid_inputs = [video_input, confidence, show_hands] + zone_inputs
    vid_outputs = [video_out, vid_stats]
    vid_btn.click(fn=process_video, inputs=vid_inputs, outputs=vid_outputs)


if __name__ == "__main__":
    print("Downloading models (first run only)...")
    get_detector(0.6)
    print("Models ready. Launching demo...")
    demo.launch(server_name="127.0.0.1", server_port=7861, share=True)
