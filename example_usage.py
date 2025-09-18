#!/usr/bin/env python3
"""
Example usage of the local PNG to DXF converter
"""

import os
import sys
from pathlib import Path
from local_png_to_dxf import process_png_to_dxf, Config

def example_basic_usage():
    """Basic usage example"""
    print("=== Basic Usage Example ===")
    
    # Check if we have a test image
    test_image = None
    for img in ["test_floorplan.png", "sample.png", "floorplan.png"]:
        if Path(img).exists():
            test_image = img
            break
    
    if not test_image:
        print("❌ No test image found. Please place a PNG file in the current directory.")
        return
    
    print(f"��️  Processing: {test_image}")
    
    # Basic conversion
    output_path = process_png_to_dxf(
        test_image,
        output_path="outputs/example_basic.dxf",
        show_preview=True
    )
    
    print(f"✅ Basic conversion complete: {output_path}")

def example_custom_settings():
    """Custom settings example"""
    print("\n=== Custom Settings Example ===")
    
    # Update configuration
    Config.PIXELS_PER_METER = 150.0  # Different scale
    Config.WALL_THICK_M = 0.1        # Thicker walls
    Config.WALL_HEIGHT_M = 2.5       # Lower walls
    Config.SNAP_TOL_DEG = 2.0        # More precise snapping
    
    test_image = None
    for img in ["test_floorplan.png", "sample.png", "floorplan.png"]:
        if Path(img).exists():
            test_image = img
            break
    
    if not test_image:
        print("❌ No test image found for custom settings example.")
        return
    
    print(f"🖼️  Processing with custom settings: {test_image}")
    print(f"�� Scale: {Config.PIXELS_PER_METER} pixels/meter")
    print(f"�� Wall thickness: {Config.WALL_THICK_M}m")
    print(f"📐 Wall height: {Config.WALL_HEIGHT_M}m")
    
    # Custom conversion
    output_path = process_png_to_dxf(
        test_image,
        output_path="outputs/example_custom.dxf",
        show_preview=True
    )
    
    print(f"✅ Custom conversion complete: {output_path}")

def main():
    """Main example function"""
    print("🏠 Local PNG to DXF Converter - Usage Examples")
    print("=" * 50)
    
    # Create output directory
    os.makedirs("outputs", exist_ok=True)
    
    # Run examples
    example_basic_usage()
    example_custom_settings()
    
    print("\n🎉 All examples completed!")
    print("\n📁 Generated files:")
    for dxf_file in Path("outputs").glob("*.dxf"):
        print(f"  - {dxf_file}")

if __name__ == "__main__":
    main()
