"""
FactorySegMaster Day 3: YOLOv8 Gradio Demo

Full video/image processing demo using ultralytics YOLOv8.
Handles multi-person tracking and object detection built for factory scenes.
"""

# --- Monkey-patch for gradio_client JSON schema bug ---
# The bug: gradio_client/utils.py get_type() does "if 'const' in schema"
# but schema can be a boolean (from JSON schema additionalProperties: true),
# and you cannot use 'in' on a bool. We patch it before importing gradio.
import gradio_client.utils as _gc_utils

_original_get_type = _gc_utils.get_type

def _patched_get_type(schema):
    if isinstance(schema, bool):
        return "bool"
    return _original_get_type(schema)

_gc_utils.get_type = _patched_get_type

# Also patch _json_schema_to_python_type to handle bool schemas
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
from PIL import Image

from yolo_detector import YoloDetector
from safety_zones import SafetyZone, draw_zones, check_violations_px, draw_violations

# Global detector (reused across requests)
_detector = None

def get_detector(conf_thresh=0.25, iou_thresh=0.45):
    """Singleton pattern to avoid reloading YOLO weights for every frame."""
    global _detector
    if _detector is None:
        _detector = YoloDetector(conf_thresh=conf_thresh, iou_thresh=iou_thresh)
    else:
        _detector.set_thresholds(conf_thresh, iou_thresh)
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
    """Format YOLO detection stats as a clean text string."""
    lines = [
        f"Total Objects Detected: {result['count']}",
        f"Active Safety Zones: {len(zones)}",
        f"Total Zone Violations: {len(violations)}",
    ]
    if violations:
        lines.append("")
        for v in violations:
            lines.append(f"  >> {v}")
    elif result["count"] > 0:
        lines.append("")
        lines.append("  All detected objects are in safe positions.")

    if result["objects"]:
        lines.append("")
        lines.append("Object Details:")
        # Group by class to keep it clean
        counts = {}
        for obj in result["objects"]:
            cls = obj["class_name"]
            counts[cls] = counts.get(cls, 0) + 1
            
        for cls, count in counts.items():
            lines.append(f"  - {count}x {cls}")

    return "\n".join(lines)


# --- Image Processing ---

def process_image(image, conf_thresh, iou_thresh, target_person, target_vehicle, 
                  z1_x1, z1_y1, z1_x2, z1_y2, z1_label,
                  z2_x1, z2_y1, z2_x2, z2_y2, z2_label):
    
    if image is None:
        return None, "Upload a factory image to get started."

    detector = get_detector(conf_thresh, iou_thresh)
    
    # Determine which classes we want YOLO to look for
    target_classes = []
    if target_person: target_classes.append(0)    # COCO ID for person
    if target_vehicle: target_classes.extend([2, 5, 7]) # COCO IDs for car, bus, truck
    
    frame_rgb = np.array(image)
    h, w = frame_rgb.shape[:2]

    # Detect
    result = detector.detect(frame_rgb, target_classes=target_classes)

    # Zones
    zones = build_zones(z1_x1, z1_y1, z1_x2, z1_y2, z1_label,
                        z2_x1, z2_y1, z2_x2, z2_y2, z2_label)

    # Check violations
    violations = check_violations_px(result["objects"], zones, w, h)

    # Annotated frame with zones + violations
    zone_frame = result["annotated"].copy()
    draw_zones(zone_frame, zones)
    draw_violations(zone_frame, violations)

    # Stats
    stats = format_stats(result, violations, zones)

    return Image.fromarray(zone_frame), stats


# --- Video Processing ---

def process_video(video_path, conf_thresh, iou_thresh, target_person, target_vehicle,
                  z1_x1, z1_y1, z1_x2, z1_y2, z1_label,
                  z2_x1, z2_y1, z2_x2, z2_y2, z2_label):
                  
    if video_path is None:
        return None, "Upload a factory video."

    detector = get_detector(conf_thresh, iou_thresh)
    
    target_classes = []
    if target_person: target_classes.append(0)
    if target_vehicle: target_classes.extend([2, 5, 7])
    
    zones = build_zones(z1_x1, z1_y1, z1_x2, z1_y2, z1_label,
                        z2_x1, z2_y1, z2_x2, z2_y2, z2_label)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None, "Error: Could not open video file."

    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Output video mapping (Gradio expects generic valid mp4 for browser viewing)
    out_path = tempfile.mktemp(suffix=".mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))

    frame_count = 0
    total_violations = 0
    max_objects_in_frame = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        
        # We process every frame for YOLO since it is optimized
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = detector.detect(frame_rgb, target_classes=target_classes)
        violations = check_violations_px(result["objects"], zones, w, h)

        out_frame = result["annotated"].copy()
        draw_zones(out_frame, zones)
        draw_violations(out_frame, violations)

        max_objects_in_frame = max(max_objects_in_frame, result["count"])
        if len(violations) > 0:
            total_violations += 1

        writer.write(cv2.cvtColor(out_frame, cv2.COLOR_RGB2BGR))

    cap.release()
    writer.release()

    stats_lines = [
        f"Video Processing Complete",
        f"Frames Processed: {frame_count} / {total_frames}",
        f"Output Resolution: {w}x{h} @ {fps:.1f}fps",
        f"Max Objects in Single Frame: {max_objects_in_frame}",
        f"Total Violation Events (Frames with alerts): {total_violations}",
    ]

    return out_path, "\n".join(stats_lines)


# --- Gradio UI ---

with gr.Blocks(
    title="FactorySegMaster Day 3",
    theme=gr.themes.Soft(),
) as demo:
    gr.Markdown(
        "# FactorySegMaster Day 3: YOLOv8 Factory Object Detection\n"
        "Detect workers and machinery using semantic object tracking. Multi-person occlusion issues from Day 2 are solved.\n\n"
        "**Note:** First launch will auto-download the YOLOv8n checkpoint (~6MB)."
    )

    with gr.Row():
        # Controls column
        with gr.Column(scale=1):
            
            gr.Markdown("### YOLOv8 Engine Config")
            with gr.Row():
                target_person = gr.Checkbox(label="Detect Workers (Person)", value=True)
                target_vehicle = gr.Checkbox(label="Detect Machinery (Vehicles)", value=True)
                
            conf_thresh = gr.Slider(
                0.1, 0.9, value=0.25, step=0.05,
                label="Confidence Threshold",
                info="Higher = fewer false positives",
            )
            iou_thresh = gr.Slider(
                0.1, 0.9, value=0.45, step=0.05,
                label="NMS IoU Threshold",
                info="Controls overlapping bounding boxes",
            )

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
                    img_btn = gr.Button("Detect Objects", variant="primary")
                    zone_out = gr.Image(label="Safety Zone Monitor", type="pil")
                    img_stats = gr.Textbox(label="Detection Log", lines=8)

                with gr.TabItem("Video Mode"):
                    video_input = gr.Video(label="Factory Video (.mp4)")
                    vid_btn = gr.Button("Process Full Video", variant="primary")
                    video_out = gr.Video(label="Annotated Inference Stream")
                    vid_stats = gr.Textbox(label="Video Summary", lines=6)

    # Inputs
    zone_inputs = [
        z1_x1, z1_y1, z1_x2, z1_y2, z1_label,
        z2_x1, z2_y1, z2_x2, z2_y2, z2_label,
    ]
    core_inputs = [conf_thresh, iou_thresh, target_person, target_vehicle]

    # Links
    img_inputs = [image_input] + core_inputs + zone_inputs
    img_outputs = [zone_out, img_stats]
    img_btn.click(fn=process_image, inputs=img_inputs, outputs=img_outputs)

    vid_inputs = [video_input] + core_inputs + zone_inputs
    vid_outputs = [video_out, vid_stats]
    vid_btn.click(fn=process_video, inputs=vid_inputs, outputs=vid_outputs)


if __name__ == "__main__":
    print("Pre-loading YOLOv8 weights...")
    get_detector()
    print("Model ready. Launching full demo server...")
    demo.launch(server_name="127.0.0.1", server_port=7862, share=True)
