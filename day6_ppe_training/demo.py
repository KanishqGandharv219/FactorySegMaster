"""
FactorySegMaster Day 6: Custom PPE Training Demo

Gradio UI to test our fine-tuned YOLOv8 weights (best.pt)
on factory images to detect Hardhats and Safety Vests.
"""


import cv2
import gradio as gr
import numpy as np

from ppe_detector import PPEDetector

# Global Detector Load
_detector = None

def get_detector():
    global _detector
    if _detector is None:
        # Defaults to our custom trained model (or falls back to yolov8n if missing)
        _detector = PPEDetector()
    return _detector

def process_image(image, conf_thresh):
    if image is None:
        return None, "Upload an image."
        
    detector = get_detector()
    detector.set_threshold(conf_thresh)
    
    # Gradio provides RGB PIL images. Convert to BGR for OpenCV
    bgr_img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    
    # Run custom inference
    annotated_bgr, detections = detector.detect_ppe(bgr_img)
    
    # Convert back to RGB for Gradio
    annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
    
    # Build a simple compliance report
    report = f"Found {len(detections)} objects total.\n\n"
    ppe_count = sum(1 for d in detections if d["is_ppe"])
    
    report += f"PPE Detected (Hardhats/Vests): {ppe_count}\n"
    
    return annotated_rgb, report

# --- Gradio UI ---

with gr.Blocks(
    title="FactorySegMaster Day 6",
    theme=gr.themes.Soft()
) as demo:
    gr.Markdown(
        "# FactorySegMaster Day 6: Custom PPE Model\n"
        "Testing our fine-tuned YOLOv8 model (`best.pt`) on custom Hardhat and Safety Vest datasets."
    )
    
    with gr.Row():
        with gr.Column(scale=3):
            img_input = gr.Image(type="pil", label="Factory Image")
            img_output = gr.Image(type="numpy", label="PPE Detections (Green=PPE, Red=Non-PPE)")
            
        with gr.Column(scale=1):
            conf_slider = gr.Slider(0.1, 1.0, value=0.4, step=0.05, label="Confidence Tresh")
            run_btn = gr.Button("Detect Safety Gear", variant="primary")
            stats_box = gr.Textbox(label="Compliance Report", lines=4)
            
            gr.Markdown(
                "### Instructions\n"
                "1. If you haven't trained a model, run `python train_ppe.py` first.\n"
                "2. The app looks for `runs/detect/ppe_detector/weights/best.pt`.\n"
                "3. If missing, it will fall back to standard `yolov8n.pt` and just detect generic 'Persons'."
            )

    run_btn.click(
        fn=process_image,
        inputs=[img_input, conf_slider],
        outputs=[img_output, stats_box]
    )

if __name__ == "__main__":
    print("Pre-loading Custom YOLO PPE Model...")
    get_detector()
    print("Detector ready. Launching Custom Model Demo Server...")
    demo.launch(server_name="127.0.0.1", server_port=7865, share=True)
