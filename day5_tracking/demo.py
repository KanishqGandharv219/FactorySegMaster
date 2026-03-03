"""
FactorySegMaster Day 5: ByteTrack Gradio Demo

Full video processing demo using YOLOv8 + ByteTrack.
Provides persistent object IDs across frames and monitors Dwell Time in safety zones.
"""

# --- Monkey-patch for gradio_client JSON schema bug (from Day 3) ---
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
import time
import tempfile
import gradio as gr
import numpy as np
from PIL import Image

from tracker import PersistentTracker
from safety_zones import SafetyZoneManager

# Global Tracker (reused but reset per video)
_tracker = None

def get_tracker():
    global _tracker
    if _tracker is None:
        _tracker = PersistentTracker("yolov8n.pt")
    return _tracker

def process_video(
    video_path,
    conf_thresh,
    iou_thresh,
    max_dwell,
    zx1, zy1, zx2, zy2,
    progress=gr.Progress()
):
    if video_path is None:
        return None, "Please upload a video first."
        
    tracker = get_tracker()
    tracker.set_thresholds(conf_thresh, iou_thresh)
    
    # CRITICAL: Reset tracker state so IDs start fresh for the new video
    tracker.reset()
    
    # Define safety zone
    zone = SafetyZoneManager(zx1, zy1, zx2, zy2, max_dwell_seconds=max_dwell)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None, "Error: Could not open video file."
        
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    out_path = tempfile.mktemp(suffix=".mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))
    
    # Custom loop timing base (so dwell time matches video speed)
    virtual_time_base = 0.0
    time_per_frame = 1.0 / fps
    
    all_alerts = set()
    max_unique_ids = 0
    
    for i in progress.tqdm(range(total_frames), desc="Tracking Objects"):
        ret, frame = cap.read()
        if not ret: break

        start_t = time.time()
        
        # 1. Run YOLO + ByteTrack
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = tracker.track_frame(rgb_frame)
        
        annotated_rgb = result["annotated"]
        tracked_objects = result["objects"]
        
        # Track total unique objects seen
        for obj in tracked_objects:
            if obj["id"] > max_unique_ids:
                max_unique_ids = obj["id"]
                
        # 2. Process Safety Zones with virtual video time
        virtual_time_base += time_per_frame
        alerts, final_rgb = zone.process_frame(annotated_rgb, tracked_objects, current_time=virtual_time_base)
        
        for a in alerts:
            all_alerts.add(a)
            
        # Draw dynamic alerts on frame
        y_offset = 30
        for idx, alert in enumerate(alerts):
            cv2.putText(final_rgb, alert, (10, y_offset + (idx*30)), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
            
        # Draw stats
        infer_time = (time.time() - start_t) * 1000
        run_fps = 1000.0 / (infer_time + 1)
        
        status_text = f"FPS: {run_fps:.1f} | Active Objects: {result['count']} | Max Dwell: {max_dwell}s"
        cv2.putText(final_rgb, status_text, (10, height - 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        out_bgr = cv2.cvtColor(final_rgb, cv2.COLOR_RGB2BGR)
        out.write(out_bgr)
        
    cap.release()
    out.release()
    
    stats = f"Tracking Complete.\nProcessed {total_frames} frames.\n"
    stats += f"Total Unique Object IDs Tracked: {max_unique_ids}\n\n"
    stats += "Dwell Violations:\n"
    if len(all_alerts) == 0:
        stats += "None. All objects left the zone in time."
    else:
        for a in all_alerts:
            stats += f"- {a}\n"
            
    return out_path, stats

# --- Gradio UI ---

with gr.Blocks(
    title="FactorySegMaster Day 5",
    theme=gr.themes.Soft(),
) as demo:
    gr.Markdown(
        "# FactorySegMaster Day 5: ByteTrack Persistent Tracking\n"
        "Provides persistent integer IDs to objects across video frames. Upgrades Safety Zones to monitor **Dwell Time** (how long an ID remains in the zone)."
    )
    
    with gr.Row():
        with gr.Column(scale=3):
            video_input = gr.Video(label="Upload Factory Video")
            video_output = gr.Video(label="Tracking Results (Persistent IDs)")
            
        with gr.Column(scale=1):
            gr.Markdown("### Tracking Rules")
            conf_slider = gr.Slider(0.1, 1.0, value=0.25, step=0.05, label="Confidence Tresh")
            iou_slider = gr.Slider(0.1, 1.0, value=0.45, step=0.05, label="NMS IoU")
            
            gr.Markdown("### Safety Zone & Dwell Time")
            dwell_slider = gr.Slider(0.5, 10.0, value=3.0, step=0.5, label="Max Dwell Time (Seconds)")
            
            zx1 = gr.Slider(0.0, 1.0, value=0.3, label="Zone X1")
            zy1 = gr.Slider(0.0, 1.0, value=0.5, label="Zone Y1")
            zx2 = gr.Slider(0.0, 1.0, value=0.8, label="Zone X2")
            zy2 = gr.Slider(0.0, 1.0, value=0.9, label="Zone Y2")
            
            run_btn = gr.Button("Track Video", variant="primary")
            stats_box = gr.Textbox(label="Tracking Summary", lines=6)

    run_btn.click(
        fn=process_video,
        inputs=[video_input, conf_slider, iou_slider, dwell_slider, zx1, zy1, zx2, zy2],
        outputs=[video_output, stats_box]
    )

if __name__ == "__main__":
    print("Pre-loading YOLOv8 weights...")
    get_tracker()
    print("Tracker ready. Launching full demo server...")
    demo.launch(server_name="127.0.0.1", server_port=7864, share=True)
