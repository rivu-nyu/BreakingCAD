import gradio as gr
import numpy as np
from src.vision import detect_primitives
from src.geometry import find_constraints
from src.exporter import generate_dxf
import os

# Create output directory if it doesn't exist
os.makedirs("outputs", exist_ok=True)
OUTPUT_DXF_PATH = "outputs/output_sketch.dxf"

def process_sketch(input_image: np.ndarray) -> (np.ndarray, str):
    """
    The main processing pipeline for the Gradio app.
    Takes an image, runs it through the pipeline, and returns
    a visualization and the path to the DXF file.
    """
    if input_image is None:
        raise gr.Error("Please upload an image first!")

    # 1. Vision Module: Detect primitives from the image
    primitives = detect_primitives(input_image)
    
    # 2. Geometry Module: Find constraints between primitives
    constrained_primitives = find_constraints(primitives)
    
    # 3. Exporter Module: Generate the DXF file
    generate_dxf(constrained_primitives, OUTPUT_DXF_PATH)
    
    # MLE 3 will later replace this with a proper visualization plot.
    visualization_output = input_image
    
    return visualization_output, OUTPUT_DXF_PATH


# --- Build the Gradio UI ---
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# ✏️ PaperCAD Edge")
    gr.Markdown("Upload a hand-drawn technical sketch to convert it into a parametric CAD file.")
    
    with gr.Row():
        with gr.Column(scale=1):
            image_input = gr.Image(type="numpy", label="Upload Sketch")
            submit_btn = gr.Button("Convert to CAD", variant="primary")
        with gr.Column(scale=1):
            image_output = gr.Image(label="Detected Geometry")
            file_output = gr.File(label="Download DXF")
            
    submit_btn.click(
        fn=process_sketch,
        inputs=image_input,
        outputs=[image_output, file_output]
    )

if __name__ == "__main__":
    demo.launch()
