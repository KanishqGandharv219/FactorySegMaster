"""
FactorySegMaster Day 7: Ultimate Ensemble UI
Provides a Gradio dashboard to run the full FactoryTwin engine on images and video.
"""

import gradio as gr
import cv2
import tempfile
import os
from factory_twin import FactoryTwin

# Initialize the global heavy FactoryTwin engine once
print("Loading all AI Models into VRAM. This might take a moment...")
twin_engine = FactoryTwin()
print("FactoryTwin Ready!")

def process_image(img_rgb, enable_ppe, enable_sam2):
    # Gradio provides RGB, our engine expects BGR (like OpenCV VideoCapture)
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    
    # Process
    twin_engine.analytics.reset()
    out_bgr = twin_engine.process_frame(img_bgr, enable_ppe, enable_sam2)
    
    # Convert back to RGB for Gradio display
    out_rgb = cv2.cvtColor(out_bgr, cv2.COLOR_BGR2RGB)
    
    logs = twin_engine.analytics.get_log_text()
    if not logs:
        logs = "No safety events detected."
        
    return out_rgb, logs

def process_video(video_path, enable_ppe, enable_sam2):
    if not video_path:
        return None, "No video provided."
        
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None, "Error opening video file."
        
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # We will save the output to a temp file
    temp_dir = tempfile.mkdtemp()
    out_path = os.path.join(temp_dir, "factory_twin_output.mp4")
    
    # Use mp4v codec for gradio compatibility
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))
    
    twin_engine.analytics.reset()
    
    # Process the entire video
    frame_count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        processed_frame = twin_engine.process_frame(frame, enable_ppe, enable_sam2)
        out.write(processed_frame)
        frame_count += 1
        
    cap.release()
    out.release()
    
    logs = twin_engine.analytics.get_log_text()
    if not logs:
        logs = "No safety events detected."
        
    return out_path, logs

# --- Gradio UI Layout ---
with gr.Blocks(title="FactorySegMaster - Day 7 FactoryTwin", theme=gr.themes.Monochrome()) as app:
    gr.Markdown("# Day 7: FactoryTwin Master Ensemble")
    gr.Markdown("Combines YOLO Tracking (Day 5), Custom PPE Detection (Day 6), and SAM2 Zero-Shot Segmentation (Day 4) into a single visual analytics pipeline.")
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### Engine Controls")
            gr.Markdown("> **Warning:** Enabling all engines simultaneously requires heavy GPU compute. On laptops, expect low FPS.")
            
            enable_ppe = gr.Checkbox(label="Enable PPE Detection (YOLOv8 Custom)", value=True)
            enable_sam2 = gr.Checkbox(label="Enable Zero-Shot Worker Masks (SAM2)", value=False)
            
            log_output = gr.Textbox(label="Live Event Log", lines=10, max_lines=15, interactive=False)
            
        with gr.Column(scale=3):
            with gr.Tabs():
                with gr.Tab("Image Processing"):
                    img_input = gr.Image(label="Upload Factory Image")
                    img_btn = gr.Button("Process Image through FactoryTwin", variant="primary")
                    img_output = gr.Image(label="FactoryTwin Dashboard Output")
                    
                    img_btn.click(
                        fn=process_image,
                        inputs=[img_input, enable_ppe, enable_sam2],
                        outputs=[img_output, log_output]
                    )
                    
                with gr.Tab("Video Processing (150 Frames)"):
                    vid_input = gr.Video(label="Upload Factory Video")
                    vid_btn = gr.Button("Process Video through FactoryTwin", variant="primary")
                    vid_output = gr.Video(label="FactoryTwin Dashboard Output")
                    
                    vid_btn.click(
                        fn=process_video,
                        inputs=[vid_input, enable_ppe, enable_sam2],
                        outputs=[vid_output, log_output]
                    )

if __name__ == "__main__":
    app.launch()
