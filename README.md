# PaperCAD Edge: Intelligent Sketch-to-CAD Converter

PaperCAD Edge is a powerful, local-first application that transforms both **hand-drawn sketches** and **professional floor plans** into clean, editable, and intelligent CAD models. By leveraging NPU acceleration, PaperCAD Edge delivers real-time performance, making it ideal for on-the-go design and digitization.

-----

## 🚀 Key Use Cases

  * **Real-Time Conversion**: Use your device's camera to capture a live image of a sketch and watch it instantly become a clean CAD model on your screen.
  * **Hand-Drawn Sketch Digitization**: Transform rough, back-of-the-napkin sketches into geometrically perfect and editable DXF files. Our "**Geometric Intelligence**" engine cleans up imperfections and understands your design intent.
  * **Professional Floor Plan Processing**: Upload existing high-quality floor plans or scanned drawings to quickly convert them into layered, parametric CAD models.

-----

## ✨ Features

  * **Multi-Input Support**: Processes PNGs, JPEGs, and live camera feeds.
  * **Geometric Intelligence**: Goes beyond simple tracing to infer constraints like perpendicularity and parallelism, cleaning up rough drawings.
  * **AI-Powered Line & Symbol Detection**: Uses a pipeline of advanced models to accurately identify walls, doors, windows, and other symbols.
  * **Text & Dimension Recognition**: Automatically reads text labels from drawings using **OCR** (Optical Character Recognition), preparing the geometry for clean, scaled export.
  * **3D Generation**: Instantly extrudes 2D floor plans into 3D wall prisms with proper thickness and height.
  * **NPU Acceleration**: Seamlessly utilizes the NPU for core AI tasks to ensure real-time performance and energy efficiency.

-----

## 🧠 Our AI Pipeline

Our system uses a combination of specialized, lightweight models for maximum accuracy and speed.

  * **Structure Detection**: A lightweight CNN + Transformer architecture (like M-LSD) turns the image into clean line segments.
  * **Symbol Detection**: A quantized YOLOv8-Nano model identifies specific symbols like doors and windows.
  * **Text Recognition**: We use PaddleOCR for its high accuracy and efficiency in reading dimensions and labels.
  * **Geometric Refinement**: A Graph Neural Network (GNN) analyzes the relationships between detected lines to refine alignment and snap junctions, ensuring a geometrically perfect output.

-----

## 🛠️ Installation

1.  Install Python 3.8 or higher.
2.  Install dependencies:
    ```bash
    pip install -r requirements_local.txt
    ```

-----

## 💻 Usage

### Command Line

```bash
# Convert a hand-drawn sketch
python local_png_to_dxf.py my_sketch.png -o sketch.dxf

# Convert a professional floorplan with custom scale
python local_png_to_dxf.py professional_plan.png --pixels-per-meter 150 --wall-thickness 0.1 --wall-height 2.5
```

### Python API

```python
from local_png_to_dxf import process_png_to_dxf

# Convert PNG to DXF
output_path = process_png_to_dxf(
    "input_floorplan.png",
    output_path="output.dxf",
    show_preview=True
)
```

-----

## ⚙️ Configuration

You can modify the `Config` class in `local_png_to_dxf.py` to adjust:

  * **Scale**: `PIXELS_PER_METER`
  * **Wall Properties**: `WALL_THICK_M`, `WALL_HEIGHT_M`
  * **Processing**: `SNAP_TOL_DEG`, `MERGE_GAP_PX`
  * **NPU**: `USE_NPU`, `NPU_PROVIDER`

-----

## 📂 File Structure

```
├── local_png_to_dxf.py      # Main converter logic
├── app_local.py             # Gradio web interface for real-time demo
├── test_local_converter.py  # Test script
├── requirements_local.txt   # Dependencies
├── README.md                # This file
└── outputs/                 # Default output directory
    └── *.dxf                # Generated DXF files
```
