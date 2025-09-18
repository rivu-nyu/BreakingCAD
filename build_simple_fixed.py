#!/usr/bin/env python3
"""
Simplified build script for creating standalone Windows executable
Only includes packages that are actually installed
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path

def clean_build_dirs():
    """Clean previous build artifacts"""
    dirs_to_clean = ['build', 'dist', '__pycache__']
    for dir_name in dirs_to_clean:
        if os.path.exists(dir_name):
            print(f"Cleaning {dir_name}...")
            shutil.rmtree(dir_name)

def ensure_directories():
    """Ensure required directories exist"""
    os.makedirs("outputs", exist_ok=True)
    os.makedirs("models", exist_ok=True)

def build_executable():
    """Build the executable using PyInstaller with only available packages"""
    print("Building standalone executable (SIMPLIFIED)...")
    
    # PyInstaller command with only essential options
    cmd = [
        "pyinstaller",
        "--onefile",  # Create a single executable file
        "--console",  # Show console for debugging
        "--name=FloorPlanConverter",  # Name of the executable
        "--add-data=src;src",  # Include src directory
        "--add-data=models;models",  # Include models directory if it exists
        "--add-data=build;build",  # Include build directory if it exists
        # Only include packages that are actually installed
        "--hidden-import=gradio",
        "--hidden-import=gradio.components",
        "--hidden-import=gradio.blocks", 
        "--hidden-import=gradio.themes",
        "--hidden-import=matplotlib",
        "--hidden-import=matplotlib.pyplot",
        "--hidden-import=matplotlib.patches",
        "--hidden-import=matplotlib.colors",
        "--hidden-import=onnxruntime",
        "--hidden-import=torch",
        "--hidden-import=cv2",
        "--hidden-import=PIL",
        "--hidden-import=PIL.Image",
        "--hidden-import=ezdxf",
        "--hidden-import=numpy",
        "--hidden-import=psutil",
        "--hidden-import=local_png_to_dxf",
        "--hidden-import=standalone",
        "--hidden-import=src",
        "--hidden-import=src.exporter",
        "--hidden-import=src.geometry",
        "--hidden-import=src.vision",
        "app_local.py"
    ]
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("✅ Build successful!")
        print(f"Executable created in: {os.path.abspath('dist/FloorPlanConverter.exe')}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Build failed: {e}")
        print(f"Error output: {e.stderr}")
        print(f"Standard output: {e.stdout}")
        return False

def main():
    """Main build process"""
    print("🔨  Building FloorPlan Converter Executable (SIMPLIFIED)")
    print("=" * 60)
    
    # Clean previous builds
    clean_build_dirs()
    
    # Ensure directories exist
    ensure_directories()
    
    # Build the executable
    if build_executable():
        print("\n🎉 Build completed successfully!")
        print("Files created:")
        print(f"  📁 dist/FloorPlanConverter.exe")
        print("\nTo test: Run FloorPlanConverter.exe from the dist folder")
        print("Note: This version shows console output for debugging")
    else:
        print("\n❌ Build failed. Check the error messages above.")

if __name__ == "__main__":
    main()
