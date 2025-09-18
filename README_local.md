# Local PNG to DXF Converter

A streamlined version of the floorplan to 3D DXF converter that runs locally and can utilize NPU acceleration when available.

## Features

- **PNG to DXF Conversion**: Converts floorplan images to 3D DXF files
- **Text Removal**: Automatically removes text labels from floorplans using MSER
- **Line Detection**: Detects wall lines using LSD (Line Segment Detector) or Hough transforms
- **3D Generation**: Creates 3D wall prisms with proper thickness and height
- **Color Analysis**: Assigns different layers based on detected colors
- **NPU Support**: Optional NPU acceleration for faster processing

## Installation

1. Install Python 3.8 or higher
2. Install dependencies:
   ```bash
   pip install -r requirements_local.txt
   ```

## Usage

### Command Line

```bash
# Basic usage
python local_png_to_dxf.py input_floorplan.png

# With custom output path
python local_png_to_dxf.py input_floorplan.png -o output.dxf

# With custom scale and dimensions
python local_png_to_dxf.py input_floorplan.png --pixels-per-meter 150 --wall-thickness 0.1 --wall-height 2.5

# Disable 3D preview
python local_png_to_dxf.py input_floorplan.png --no-preview
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

### Test the Converter

```bash
# Place a test PNG file in the current directory
python test_local_converter.py
```

## Configuration

You can modify the `Config` class in `local_png_to_dxf.py` to adjust:

- **Scale**: `PIXELS_PER_METER` - pixels per meter ratio
- **Wall Properties**: `WALL_THICK_M`, `WALL_HEIGHT_M` - wall dimensions
- **Processing**: `SNAP_TOL_DEG`, `MERGE_GAP_PX` - line processing parameters
- **NPU**: `USE_NPU`, `NPU_PROVIDER` - NPU acceleration settings

## Output

The converter generates:
- **3D DXF file** with colored wall layers
- **3D preview** (optional) showing the generated geometry
- **Console output** with processing statistics

## NPU Support

If ONNX Runtime is available and NPU providers are detected, the converter will automatically use NPU acceleration for:
- Text removal
- Line detection
- Other computer vision tasks

## Troubleshooting

1. **No line segments detected**: Try adjusting `MIN_SEG_LEN_PX` or `SNAP_TOL_DEG`
2. **Poor text removal**: The MSER parameters can be tuned in `boxes_from_mser_filtered()`
3. **NPU not working**: Check that ONNX Runtime is installed and NPU providers are available

## File Structure

```
├── local_png_to_dxf.py      # Main converter (standalone)
├── app_local.py             # Gradio web interface
├── test_local_converter.py  # Test script
├── batch_convert.py         # Batch processing
├── setup_local.py           # Setup script
├── example_usage.py         # Usage examples
├── requirements_local.txt   # Dependencies
├── README_local.md         # This file
└── outputs/                # Output directory
    └── *.dxf              # Generated DXF files
```

