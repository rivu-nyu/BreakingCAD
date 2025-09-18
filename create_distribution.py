#!/usr/bin/env python3
"""
Create a complete distribution package
"""

import os
import shutil
import zipfile
from pathlib import Path

def create_distribution():
    """Create a complete distribution package"""
    
    print("📦 Creating distribution package...")
    
    # Create distribution directory
    dist_dir = "FloorPlanConverter_v1.0"
    if os.path.exists(dist_dir):
        shutil.rmtree(dist_dir)
    os.makedirs(dist_dir)
    
    # Copy executable and related files
    files_to_copy = [
        ("dist/FloorPlanConverter.exe", "FloorPlanConverter.exe"),
        ("dist/README.txt", "README.txt"),
        ("launch_converter.bat", "launch_converter.bat"),
    ]
    
    for src, dst in files_to_copy:
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(dist_dir, dst))
            print(f"  ✅ Copied {dst}")
        else:
            print(f"  ⚠️  Warning: {src} not found")
    
    # Create outputs directory
    os.makedirs(os.path.join(dist_dir, "outputs"))
    
    # Create sample inputs directory
    sample_inputs_dir = os.path.join(dist_dir, "sample_inputs")
    os.makedirs(sample_inputs_dir)
    
    # Copy sample images if they exist
    if os.path.exists("inputs"):
        for file in os.listdir("inputs"):
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                shutil.copy2(
                    os.path.join("inputs", file),
                    os.path.join(sample_inputs_dir, file)
                )
                print(f"  ✅ Added sample input: {file}")
    
    # Create zip file
    zip_name = f"{dist_dir}.zip"
    with zipfile.ZipFile(zip_name, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(dist_dir):
            for file in files:
                file_path = os.path.join(root, file)
                arc_path = os.path.relpath(file_path, dist_dir)
                zipf.write(file_path, arc_path)
    
    print(f"\n🎉 Distribution created: {zip_name}")
    print(f"📁 Distribution folder: {dist_dir}")
    print("\nTo distribute:")
    print(f"  1. Share the {zip_name} file")
    print(f"  2. Or share the entire {dist_dir} folder")
    print("\nUser instructions:")
    print("  1. Extract the zip file")
    print("  2. Run FloorPlanConverter.exe")
    print("  3. Open browser to http://localhost:7860")

if __name__ == "__main__":
    create_distribution()
