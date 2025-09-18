#!/usr/bin/env python3
"""
Batch convert multiple PNG files to DXF using the local converter
"""

import os
import sys
from pathlib import Path
from local_png_to_dxf import process_png_to_dxf, Config

def batch_convert(input_dir: str, output_dir: str = None, **kwargs):
    """
    Convert all PNG files in a directory to DXF
    
    Args:
        input_dir: Directory containing PNG files
        output_dir: Output directory for DXF files (optional)
        **kwargs: Additional arguments for process_png_to_dxf
    """
    input_path = Path(input_dir)
    if not input_path.exists():
        print(f"❌ Input directory not found: {input_dir}")
        return
    
    if output_dir is None:
        output_dir = input_path / "dxf_output"
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all PNG files
    png_files = list(input_path.glob("*.png")) + list(input_path.glob("*.PNG"))
    
    if not png_files:
        print(f"❌ No PNG files found in {input_dir}")
        return
    
    print(f"�� Found {len(png_files)} PNG files to convert")
    print(f"�� Output directory: {output_dir}")
    
    success_count = 0
    for i, png_file in enumerate(png_files, 1):
        try:
            print(f"\n🔄 Processing {i}/{len(png_files)}: {png_file.name}")
            
            # Generate output filename
            output_path = output_dir / f"{png_file.stem}_3d.dxf"
            
            # Process the file
            result_path = process_png_to_dxf(
                str(png_file),
                str(output_path),
                show_preview=False,  # Disable preview for batch processing
                **kwargs
            )
            
            if Path(result_path).exists():
                success_count += 1
                print(f"✅ Success: {Path(result_path).name}")
            else:
                print(f"❌ Failed: Output file not created")
            
        except Exception as e:
            print(f"❌ Failed: {png_file.name} - {e}")
    
    print(f"\n🎉 Batch conversion complete!")
    print(f"✅ Successfully converted: {success_count}/{len(png_files)} files")
    print(f"📁 Output directory: {output_dir}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Batch convert PNG files to DXF")
    parser.add_argument("input_dir", help="Input directory containing PNG files")
    parser.add_argument("-o", "--output", help="Output directory for DXF files")
    parser.add_argument("--pixels-per-meter", type=float, default=100.0, help="Pixels per meter scale")
    parser.add_argument("--wall-thickness", type=float, default=0.05, help="Wall thickness in meters")
    parser.add_argument("--wall-height", type=float, default=3.0, help="Wall height in meters")
    
    args = parser.parse_args()
    
    # Update config
    Config.PIXELS_PER_METER = args.pixels_per_meter
    Config.WALL_THICK_M = args.wall_thickness
    Config.WALL_HEIGHT_M = args.wall_height
    
    print("🏠 Local PNG to DXF Batch Converter")
    print("=" * 40)
    print(f"📁 Input directory: {args.input_dir}")
    print(f"📤 Output directory: {args.output or 'dxf_output'}")
    print(f"�� Scale: {args.pixels_per_meter} pixels/meter")
    print(f"�� Wall thickness: {args.wall_thickness}m")
    print(f"📐 Wall height: {args.wall_height}m")
    
    # Run batch conversion
    batch_convert(
        args.input_dir,
        args.output,
        pixels_per_meter=args.pixels_per_meter,
        wall_thickness=args.wall_thickness,
        wall_height=args.wall_height
    )
