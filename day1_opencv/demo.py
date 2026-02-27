"""
FactorySegMaster Day 1: Interactive Gradio Demo
"""

import cv2
import gradio as gr
import numpy as np
from PIL import Image

from segment_factory import segment_factory_objects


def run_segmentation(image, min_area, max_area_pct, min_convexity, mode,
                     blur_size, canny_low, canny_high,
                     adaptive_block, adaptive_c):
    if image is None:
        return None, None, None, "Upload a factory image to get started!"

    img_array = np.array(image)
    if len(img_array.shape) == 2:
        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_GRAY2BGR)
    else:
        img_bgr = img_array[:, :, ::-1].copy()

    try:
        result = segment_factory_objects(
            image=img_bgr,
            min_area=int(min_area),
            max_area_pct=max_area_pct,
            min_convexity=min_convexity,
            mode=mode,
            blur_size=int(blur_size),
            canny_low=int(canny_low),
            canny_high=int(canny_high),
            adaptive_block=int(adaptive_block),
            adaptive_c=int(adaptive_c),
        )
    except Exception as e:
        import traceback
        return None, None, None, f"Error: {str(e)}\n{traceback.format_exc()}"

    s = result.stats
    r = s["rejected"]
    stats_text = (
        f"Objects Detected: {s['objects_detected']}\n"
        f"Raw Mask Coverage: {s['raw_mask_coverage_pct']}%\n"
        f"Clean Coverage: {s['coverage_pct']}%\n"
        f"Image Size: {s['image_size']}\n"
        f"Mode: {s['mode']}\n"
        f"Raw Contours: {s['contours_before_filter']}\n"
        f"Rejected: {r['too_small']} small, {r['too_large']} large, {r['low_convexity']} non-convex\n"
        f"Filters: area [{s['min_area']}-{s['max_area_pct']}%] | convexity >= {s['min_convexity']}"
    )

    raw_pil = Image.fromarray(result.raw_mask)
    mask_pil = Image.fromarray(result.mask)
    annotated_pil = Image.fromarray(result.annotated)

    return raw_pil, mask_pil, annotated_pil, stats_text


with gr.Blocks(
    title="FactorySegMaster Day 1",
    theme=gr.themes.Soft(),
) as demo:
    gr.Markdown(
        "# FactorySegMaster Day 1: OpenCV Factory Segmentation\n"
        "**Best mode:** `adaptive` (most detections). "
        "`combined` = union of all. `gradient`/`edge`/`otsu` for specific scenes."
    )

    with gr.Row():
        with gr.Column(scale=1):
            image_input = gr.Image(type="pil", label="Factory CCTV Frame",
                                   sources=["upload", "clipboard"])
            mode = gr.Radio(
                choices=["adaptive", "combined", "gradient", "edge", "otsu"],
                value="adaptive",
                label="Segmentation Mode",
            )

            gr.Markdown("### Filters")
            min_area = gr.Slider(100, 20000, value=500, step=100, label="Min Area (px)")
            max_area_pct = gr.Slider(5.0, 80.0, value=30.0, step=5.0, label="Max Area (% of image)")
            min_convexity = gr.Slider(0.0, 1.0, value=0.2, step=0.05, label="Min Convexity")

            with gr.Accordion("Advanced", open=False):
                blur_size = gr.Slider(3, 15, value=5, step=2, label="Blur Kernel")
                canny_low = gr.Slider(5, 150, value=20, step=5, label="Canny Low")
                canny_high = gr.Slider(20, 250, value=80, step=5, label="Canny High")
                adaptive_block = gr.Slider(11, 101, value=51, step=2, label="Adaptive Block")
                adaptive_c = gr.Slider(2, 30, value=8, step=1, label="Adaptive C")

            run_btn = gr.Button("Segment!", variant="primary", size="lg")

        with gr.Column(scale=2):
            with gr.Row():
                raw_output = gr.Image(label="Raw Mask (before filtering)", type="pil")
                mask_output = gr.Image(label="Clean Mask (after filtering)", type="pil")
            annotated_output = gr.Image(label="Segmented Objects", type="pil")
            stats_output = gr.Textbox(label="Stats", lines=8)

    inputs = [image_input, min_area, max_area_pct, min_convexity, mode,
              blur_size, canny_low, canny_high, adaptive_block, adaptive_c]
    outputs = [raw_output, mask_output, annotated_output, stats_output]

    run_btn.click(fn=run_segmentation, inputs=inputs, outputs=outputs)
    image_input.change(fn=run_segmentation, inputs=inputs, outputs=outputs)


if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860, share=True)
