#!/usr/bin/env python3
"""
Setup script for the local PNG to DXF converter
"""

import subprocess
import sys
import os
from pathlib import Path

def install_requirements():
    """Install required packages"""
    print("📦 Installing requirements...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements_local.txt"])
        print("✅ Requirements installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install requirements: {e}")
        return False

def create_directories():
    """Create necessary directories"""
    print("📁 Creating directories...")
    directories = ["outputs", "test_images"]
    
    for dir_name in directories:
        Path(dir_name).mkdir(exist_ok=True)
        print(f"✅ Created directory: {dir_name}")

def check_npu_support():
    """Check if NPU support is available"""
    print("🔍 Checking NPU support...")
    try:
        import onnxruntime as ort
        providers = ort.get_available_providers()
        print(f"Available providers: {providers}")
        
        if 'QNNExecutionProvider' in providers:
            print("✅ NPU support available (QNN)")
        else:
            print("⚠️  NPU support not available, using CPU fallback")
        
        return True
    except ImportError:
        print("⚠️  ONNX Runtime not installed, NPU support disabled")
        return False

def test_installation():
    """Test the installation"""
    print("�� Testing installation...")
    try:
        from local_png_to_dxf import process_png_to_dxf, Config
        print("✅ Local converter imported successfully")
        return True
    except Exception as e:
        print(f"❌ Installation test failed: {e}")
        return False

def main():
    """Main setup function"""
    print("�� Setting up Local PNG to DXF Converter")
    print("=" * 50)
    
    # Install requirements
    if not install_requirements():
        print("Setup failed at requirements installation")
        return False
    
    # Create directories
    create_directories()
    
    # Check NPU support
    check_npu_support()
    
    # Test installation
    if not test_installation():
        print("Setup failed at installation test")
        return False
    
    print("\n🎉 Setup completed successfully!")
    print("\n📋 Next steps:")
    print("1. Run tests: python run_local_pipeline.py test")
    print("2. Convert image: python local_png_to_dxf.py your_image.png")
    print("3. Launch web UI: python app_local.py")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
