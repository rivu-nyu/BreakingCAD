#!/usr/bin/env python3
"""
Local PNG to DXF Converter - Web Interface
Gradio app for the local floorplan to 3D DXF converter with hand-drawn sketch support
"""

import gradio as gr
import numpy as np
from pathlib import Path
import os
import tempfile
import time
from local_png_to_dxf import process_png_to_dxf, Config

# Create output directory if it doesn't exist
os.makedirs("outputs", exist_ok=True)

def process_handdrawn_sketch(input_image: np.ndarray, pixels_per_meter: float, wall_thickness: float, wall_height: float, show_preview: bool = False) -> (np.ndarray, str):
    """
    Process hand-drawn sketch using standalone.py and convert to DXF
    
    Args:
        input_image: Input hand-drawn image as numpy array
        pixels_per_meter: Scale setting
        wall_thickness: Wall thickness in meters
        wall_height: Wall height in meters
        show_preview: Whether to show 3D preview
    
    Returns:
        Tuple of (processed_image, dxf_file_path)
    """
    if input_image is None:
        raise gr.Error("Please upload an image first!")
    
    try:
        # Save input image temporarily
        temp_input = "outputs/temp_handdrawn.png"
        from PIL import Image
        Image.fromarray(input_image).save(temp_input)
        
        # Process hand-drawn sketch using standalone.py
        from standalone import ColoredFloorPlanProcessor
        processor = ColoredFloorPlanProcessor()
        
        # Generate intermediate PNG output
        timestamp = int(time.time())
        intermediate_png = f"outputs/intermediate_{timestamp}.png"
        os.makedirs(os.path.dirname(intermediate_png), exist_ok=True)
        
        # Process hand-drawn sketch to PNG
        processor.process_floor_plan(temp_input, intermediate_png)
        
        # Verify intermediate PNG was created
        if not os.path.exists(intermediate_png):
            raise Exception("Failed to process hand-drawn sketch")
        
        # Update config with user settings
        Config.PIXELS_PER_METER = pixels_per_meter
        Config.WALL_THICK_M = wall_thickness
        Config.WALL_HEIGHT_M = wall_height
        
        # Generate final DXF output
        output_path = f"outputs/floorplan_handdrawn_{timestamp}.dxf"
        
        # Process the intermediate PNG to DXF
        result_path = process_png_to_dxf(
            intermediate_png,
            output_path=output_path,
            show_preview=show_preview
        )
        
        # Verify the file was created
        if not os.path.exists(result_path):
            raise Exception(f"DXF file was not created at {result_path}")
        
        # Load the intermediate PNG for visualization
        processed_image = np.array(Image.open(intermediate_png))
        
        # Clean up temp files
        if os.path.exists(temp_input):
            os.remove(temp_input)
        if os.path.exists(intermediate_png):
            os.remove(intermediate_png)
        
        return processed_image, result_path
        
    except Exception as e:
        # Clean up temp files on error
        if os.path.exists(temp_input):
            os.remove(temp_input)
        if os.path.exists(intermediate_png):
            os.remove(intermediate_png)
        raise gr.Error(f"Hand-drawn sketch processing failed: {str(e)}")

def process_png_direct(input_image: np.ndarray, pixels_per_meter: float, wall_thickness: float, wall_height: float, show_preview: bool = False) -> (np.ndarray, str):
    """
    Process PNG image directly and convert to DXF using the local converter
    
    Args:
        input_image: Input PNG image as numpy array
        pixels_per_meter: Scale setting
        wall_thickness: Wall thickness in meters
        wall_height: Wall height in meters
        show_preview: Whether to show 3D preview
    
    Returns:
        Tuple of (processed_image, dxf_file_path)
    """
    if input_image is None:
        raise gr.Error("Please upload an image first!")
    
    try:
        # Save input image temporarily
        temp_input = "outputs/temp_input.png"
        from PIL import Image
        Image.fromarray(input_image).save(temp_input)
        
        # Update config with user settings
        Config.PIXELS_PER_METER = pixels_per_meter
        Config.WALL_THICK_M = wall_thickness
        Config.WALL_HEIGHT_M = wall_height
        
        # Generate unique output filename
        timestamp = int(time.time())
        output_path = f"outputs/floorplan_png_{timestamp}.dxf"
        
        # Process the image using local converter
        result_path = process_png_to_dxf(
            temp_input,
            output_path=output_path,
            show_preview=show_preview
        )
        
        # Verify the file was created
        if not os.path.exists(result_path):
            raise Exception(f"DXF file was not created at {result_path}")
        
        # Return the input image as visualization
        processed_image = input_image.copy()
        
        # Clean up temp file
        if os.path.exists(temp_input):
            os.remove(temp_input)
        
        return processed_image, result_path
        
    except Exception as e:
        # Clean up temp file on error
        if os.path.exists(temp_input):
            os.remove(temp_input)
        raise gr.Error(f"PNG processing failed: {str(e)}")

def create_sample_floorplan():
    """Create a sample floorplan for testing"""
    try:
        from PIL import Image, ImageDraw
        import numpy as np
        
        # Create a simple floorplan
        width, height = 400, 300
        img = Image.new('RGB', (width, height), 'white')
        draw = ImageDraw.Draw(img)
        
        # Draw some walls
        walls = [
            # Outer walls
            (50, 50, 350, 50),   # Top wall
            (50, 50, 50, 250),   # Left wall
            (350, 50, 350, 250), # Right wall
            (50, 250, 350, 250), # Bottom wall
            
            # Inner walls
            (200, 50, 200, 150), # Vertical divider
            (50, 150, 200, 150), # Horizontal divider
        ]
        
        for wall in walls:
            draw.line(wall, fill='black', width=3)
        
        # Add some text (to test text removal)
        draw.text((100, 100), "LIVING ROOM", fill='red')
        draw.text((250, 200), "BEDROOM", fill='blue')
        
        # Convert to numpy array
        img_array = np.array(img)
        return img_array
        
    except Exception as e:
        print(f"Error creating sample floorplan: {e}")
        return None

# Build the Gradio UI
with gr.Blocks(theme=gr.themes.Soft(), title="Floorplan to DXF Converter") as demo:
    gr.Markdown("# 🏠 Floorplan to 3D DXF Converter")
    gr.Markdown("Convert hand-drawn sketches or PNG floorplan images into professional 3D DXF files with NPU acceleration support.")
    
    with gr.Row():
        with gr.Column(scale=1):
            # Processing mode selection
            processing_mode = gr.Dropdown(
                choices=["Hand-drawn Sketch", "PNG Image"],
                value="PNG Image",
                label="Processing Mode",
                info="Choose how to process your input image"
            )
            
            # Image input
            image_input = gr.Image(
                type="numpy", 
                label="Upload Image",
                height=300
            )
            
            # Sample image button
            sample_btn = gr.Button("📋 Load Sample Floorplan", variant="secondary")
            
            # Advanced settings
            with gr.Accordion("⚙️ Advanced Settings", open=False):
                pixels_per_meter = gr.Slider(
                    minimum=50, maximum=500, value=100, step=10,
                    label="Pixels per Meter", 
                    info="Scale of the floorplan (100 = 1 meter = 100 pixels)"
                )
                wall_thickness = gr.Slider(
                    minimum=0.01, maximum=0.5, value=0.05, step=0.01,
                    label="Wall Thickness (m)", 
                    info="Thickness of walls in meters"
                )
                wall_height = gr.Slider(
                    minimum=1.0, maximum=10.0, value=3.0, step=0.1,
                    label="Wall Height (m)", 
                    info="Height of walls in meters"
                )
                show_preview = gr.Checkbox(
                    label="Show 3D Preview", 
                    value=False,
                    info="Show 3D preview window (may slow down processing)"
                )
            
            # Process button
            submit_btn = gr.Button("🔄 Convert to 3D DXF", variant="primary", size="lg")
            
        with gr.Column(scale=1):
            # Output image
            image_output = gr.Image(
                label="Processed Image", 
                height=300
            )
            
            # File output
            file_output = gr.File(
                label="Download 3D DXF File",
                file_count="single",
                file_types=[".dxf"]
            )
            
            # Status and info
            status_text = gr.Textbox(
                label="Status",
                value="Ready to process floorplan...",
                interactive=False
            )
            
            # Instructions
            gr.Markdown("""
            ### 📋 Instructions:
            1. **Select** processing mode (Hand-drawn Sketch or PNG Image)
            2. **Upload** your floorplan image
            3. **Adjust** the scale and wall parameters if needed
            4. **Click** "Convert to 3D DXF"
            5. **Download** the generated DXF file
            
            ### ✨ Features:
            - 🎨 **Hand-drawn sketch processing** using AI-powered edge detection
            - 🧹 **Automatic text removal** using MSER
            - 📏 **Line detection** with LSD/Hough transforms
            - 🎨 **Color-based layer assignment**
            - ⚡ **NPU acceleration** (when available)
            - 🔧 **Configurable parameters**
            
            ### 💡 Tips:
            - **Hand-drawn sketches**: Use clear, high-contrast drawings
            - **PNG images**: Use professional floorplan images
            - Adjust "Pixels per Meter" to match your image scale
            - Enable 3D preview to see the result before downloading
            """)
    
    # Event handlers
    def process_with_status(image, mode, pixels, thickness, height, preview):
        """Process image with status updates based on selected mode"""
        if image is None:
            return None, None, "Please upload an image first!"
        
        try:
            if mode == "Hand-drawn Sketch":
                # Process hand-drawn sketch
                processed_img, dxf_path = process_handdrawn_sketch(
                    image, pixels, thickness, height, preview
                )
                status_prefix = "Hand-drawn sketch"
            else:
                # Process PNG directly
                processed_img, dxf_path = process_png_direct(
                    image, pixels, thickness, height, preview
                )
                status_prefix = "PNG image"
            
            # Verify the file exists and get its size
            if os.path.exists(dxf_path):
                file_size = os.path.getsize(dxf_path)
                status_text = f"✅ Success! {status_prefix} processed. DXF file created: {Path(dxf_path).name} ({file_size:,} bytes)"
            else:
                status_text = f"❌ Error: DXF file not found at {dxf_path}"
                return None, None, status_text
            
            return processed_img, dxf_path, status_text
            
        except Exception as e:
            error_msg = f"❌ Error: {str(e)}"
            return None, None, error_msg
    
    def load_sample():
        """Load sample floorplan"""
        sample_img = create_sample_floorplan()
        if sample_img is not None:
            return sample_img, "Sample floorplan loaded. Click 'Convert to 3D DXF' to process."
        else:
            return None, "Failed to create sample floorplan."
    
    def update_ui_for_mode(mode):
        """Update UI elements based on selected processing mode"""
        if mode == "Hand-drawn Sketch":
            return "Upload Hand-drawn Sketch", "Ready to process hand-drawn sketch..."
        else:
            return "Upload PNG Floorplan Image", "Ready to process PNG floorplan..."
    
    # Connect the event handlers
    submit_btn.click(
        fn=process_with_status,
        inputs=[image_input, processing_mode, pixels_per_meter, wall_thickness, wall_height, show_preview],
        outputs=[image_output, file_output, status_text]
    )
    
    sample_btn.click(
        fn=load_sample,
        outputs=[image_input, status_text]
    )
    
    # Update UI when processing mode changes
    processing_mode.change(
        fn=update_ui_for_mode,
        inputs=[processing_mode],
        outputs=[image_input, status_text]
    )
    
    # Update status when image is uploaded
    def on_image_upload(image, mode):
        if image is not None:
            if mode == "Hand-drawn Sketch":
                return "Hand-drawn sketch uploaded. Ready to process!"
            else:
                return "PNG image uploaded. Ready to process!"
        return "Ready to process floorplan..."
    
    image_input.change(
        fn=on_image_upload,
        inputs=[image_input, processing_mode],
        outputs=[status_text]
    )

# Launch the app
if __name__ == "__main__":
    print(" Starting Floorplan to DXF Converter...")
    print("=" * 50)
    
    # Check NPU support
    try:
        import onnxruntime as ort
        providers = ort.get_available_providers()
        print(f"🔍 Available ONNX providers: {providers}")
        
        if 'QNNExecutionProvider' in providers:
            print("✅ NPU support available (QNN)")
        else:
            print("⚠️  NPU support not available, using CPU fallback")
    except ImportError:
        print("⚠️  ONNX Runtime not installed, NPU support disabled")
    
    print(" Launching web interface...")
    print("📱 Open your browser and go to: http://localhost:7860")
    print(" Press Ctrl+C to stop the server")
    
    try:
        demo.launch(
            server_name="0.0.0.0", 
            server_port=7860,
            share=False,
            show_error=True
        )
    except KeyboardInterrupt:
        print("\n👋 Server stopped by user")
    except Exception as e:
        print(f"❌ Error launching server: {e}")
