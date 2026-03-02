"""
FactorySegMaster Day 4: SAM2 Interactive Demo

Interactive zero-shot segmentation using Meta's SAM2 model.
Users can click anywhere on a factory image to isolate tools/parts.
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
import gradio as gr
import numpy as np
from PIL import Image

from sam2_segmenter import SAM2Segmenter

# Global segmenter (reused across requests)
_segmenter = None

def get_segmenter():
    """Singleton pattern to load SAM2 weights once."""
    global _segmenter
    if _segmenter is None:
        _segmenter = SAM2Segmenter("sam2.1_t.pt")
    return _segmenter

# Need to store the original image to run subsequent clicks against it
_last_image = None
_last_result_type = "annotated" # "annotated" or "isolated"

def handle_upload(image):
    """When user uploads a new image, store it and return it clean."""
    if image is None: return None
    
    global _last_image, _last_result_type
    # Convert PIL directly to numpy RGB array
    _last_image = np.array(image)
    _last_result_type = "annotated"
    
    return image, "Image loaded. Click anywhere to segment an object!"

def toggle_view_mode(mode_str):
    """Switch between showing neon outlines or full background subtraction."""
    global _last_result_type
    
    if "Isolate" in mode_str:
        _last_result_type = "isolated"
    else:
        _last_result_type = "annotated"
        
    return f"View mode changed to: {_last_result_type}"

def handle_click(image, evt: gr.SelectData):
    """
    Triggered when a user clicks on the image component.
    evt.index contains the (x, y) pixel coordinates of the click.
    """
    global _last_image, _last_result_type
    
    print(f"DEBUG: Click received at coordinates {evt.index}")
    
    if _last_image is None or image is None:
        print("DEBUG: Image is None or _last_image is None")
        return image, "Please upload an image first."

    # X, Y coordinates of the click
    px, py = evt.index 
    
    segmenter = get_segmenter()
    
    try:
        print(f"DEBUG: Running SAM2 prediction for point ({px}, {py})...")
        # Ask SAM2 to segment whatever is at this pixel
        result = segmenter.segment_from_point(_last_image, px, py)
        print("DEBUG: SAM2 prediction complete.")
        
        # Determine which image to return based on view toggle
        if _last_result_type == "isolated":
            out_img = result["isolated"]
        else:
            out_img = result["annotated"]
            
            # Draw a distinct red circle at the exact click point so user knows what prompted it
            cv2.circle(out_img, (px, py), 5, (255, 0, 0), -1)
            cv2.circle(out_img, (px, py), 7, (255, 255, 255), 2)
            
        stats = f"Segmented object from click at ({px}, {py}).\n"
        if result["mask"] is not None:
            area = np.sum(result["mask"])
            stats += f"Object covers {area:,} pixels."
            print(f"DEBUG: Success. Mask area: {area}")
        else:
            stats += "SAM2 failed to find an object at this point."
            print("DEBUG: Warning - SAM2 returned empty mask.")
            
        return out_img, stats
        
    except Exception as e:
        import traceback
        err_msg = f"SAM2 Error: {str(e)}\n\n{traceback.format_exc()}"
        print(f"DEBUG: Exception caught:\n{err_msg}")
        return image, err_msg


# --- Gradio UI ---

with gr.Blocks(
    title="FactorySegMaster Day 4",
    theme=gr.themes.Soft(),
) as demo:
    gr.Markdown(
        "# FactorySegMaster Day 4: SAM2 Zero-Shot Segmentation\n"
        "Click on ANY object in the factory image (tool, machine part, defect) to generate a pixel-perfect mask instantly using Meta's SAM2 model.\n\n"
        "**Note:** First launch will auto-download the `sam2.1_t.pt` checkpoint (~8MB)."
    )
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### Step 1: Upload")
            input_image = gr.Image(
                type="pil", 
                label="Upload Factory Image",
                sources=["upload", "clipboard"],
            )

            gr.Markdown("### Control Panel")
            view_mode = gr.Radio(
                choices=["Show Outlines (Annotated)", "Isolate Object (Black Background)"],
                value="Show Outlines (Annotated)",
                label="View Mode"
            )
            
            clear_btn = gr.Button("Reset View", variant="secondary")
            
            status_box = gr.Textbox(
                label="Status & Stats", 
                lines=4, 
                value="Waiting for image upload..."
            )

        with gr.Column(scale=3):
            gr.Markdown("### Step 2: Interactive Canvas (Click to Segment)")
            # This is the display image where clicks are registered
            output_image = gr.Image(
                type="pil", 
                label="Factory Output View",
                interactive=False # Must be False to cleanly register coordinate clicks
            )

    # Behavior Links
    
    # 1. Uploading an image updates the display image
    input_image.upload(
        fn=handle_upload, 
        inputs=[input_image], 
        outputs=[output_image, status_box]
    )
    
    # 2. Clicking on the output image triggers the SAM2 segmentation
    output_image.select(
        fn=handle_click,
        inputs=[output_image],
        outputs=[output_image, status_box]
    )
    
    # 3. Toggling the radio button changes the global mode
    view_mode.change(
        fn=toggle_view_mode,
        inputs=[view_mode],
        outputs=[status_box]
    )
    
    # 4. Clear button puts the original unannotated image back
    def reset_view():
        global _last_image
        if _last_image is not None:
            return Image.fromarray(_last_image), "View reset to original."
        return None, "No image to reset."
        
    clear_btn.click(
        fn=reset_view,
        inputs=[],
        outputs=[output_image, status_box]
    )

if __name__ == "__main__":
    print("Pre-loading SAM2 weights...")
    get_segmenter()
    print("Model ready. Launching full demo server...")
    demo.launch(server_name="127.0.0.1", server_port=7863, share=True)
