#!/usr/bin/env python3
"""
Complete guide to test and run the local PNG to DXF pipeline
"""

import os
import sys
import subprocess
from pathlib import Path

def check_dependencies():
    """Check if all required dependencies are installed"""
    print("🔍 Checking dependencies...")
    
    required_packages = [
        'opencv-python-headless',
        'numpy', 
        'Pillow',
        'ezdxf',
        'matplotlib',
        'onnxruntime'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            if package == 'opencv-python-headless':
                import cv2
            elif package == 'Pillow':
                import PIL
            elif package == 'onnxruntime':
                import onnxruntime
            else:
                __import__(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} - MISSING")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n⚠️  Missing packages: {missing_packages}")
        print("Run: pip install -r requirements_local.txt")
        return False
    
    print("✅ All dependencies are installed!")
    return True

def install_dependencies():
    """Install missing dependencies"""
    print("📦 Installing dependencies...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements_local.txt"])
        print("✅ Dependencies installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        return False

def create_test_image():
    """Create a simple test floorplan image"""
    print("🎨 Creating test floorplan image...")
    
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
        
        # Save test image
        test_image_path = "test_floorplan.png"
        img.save(test_image_path)
        print(f"✅ Test image created: {test_image_path}")
        return test_image_path
        
    except Exception as e:
        print(f"❌ Failed to create test image: {e}")
        return None

def test_basic_conversion():
    """Test basic PNG to DXF conversion"""
    print("\n🧪 Testing basic conversion...")
    
    try:
        from local_png_to_dxf import process_png_to_dxf, Config
        
        # Set up test configuration
        Config.PIXELS_PER_METER = 100.0
        Config.WALL_THICK_M = 0.05
        Config.WALL_HEIGHT_M = 3.0
        
        # Test with created image
        test_image = "test_floorplan.png"
        if not Path(test_image).exists():
            print("❌ Test image not found")
            return False
        
        print(f"Processing: {test_image}")
        
        # Run conversion
        output_path = process_png_to_dxf(
            test_image,
            output_path="outputs/test_basic.dxf",
            show_preview=False  # Disable preview for testing
        )
        
        if Path(output_path).exists():
            print(f"✅ Basic conversion successful: {output_path}")
            return True
        else:
            print("❌ Output file not created")
            return False
            
    except Exception as e:
        print(f"❌ Basic conversion failed: {e}")
        return False

def test_web_interface():
    """Test the web interface"""
    print("\n🌐 Testing web interface...")
    
    try:
        # Check if app_local.py exists
        if not Path("app_local.py").exists():
            print("❌ app_local.py not found")
            return False
        
        print("✅ Web interface file found")
        print("To run the web interface:")
        print("  python app_local.py")
        print("  Then open http://localhost:7860 in your browser")
        return True
        
    except Exception as e:
        print(f"❌ Web interface test failed: {e}")
        return False

def test_batch_processing():
    """Test batch processing"""
    print("\n📁 Testing batch processing...")
    
    try:
        # Check if batch_convert.py exists
        if not Path("batch_convert.py").exists():
            print("❌ batch_convert.py not found")
            return False
        
        print("✅ Batch processing file found")
        print("To run batch processing:")
        print("  python batch_convert.py .")
        return True
        
    except Exception as e:
        print(f"❌ Batch processing test failed: {e}")
        return False

def run_comprehensive_test():
    """Run comprehensive test suite"""
    print("�� Running Comprehensive Test Suite")
    print("=" * 50)
    
    # Step 1: Check dependencies
    if not check_dependencies():
        print("\n📦 Installing missing dependencies...")
        if not install_dependencies():
            print("❌ Failed to install dependencies. Please install manually.")
            return False
    
    # Step 2: Create test image
    test_image = create_test_image()
    if not test_image:
        print("❌ Failed to create test image")
        return False
    
    # Step 3: Create output directory
    os.makedirs("outputs", exist_ok=True)
    print("✅ Output directory created")
    
    # Step 4: Test basic conversion
    if not test_basic_conversion():
        print("❌ Basic conversion test failed")
        return False
    
    # Step 5: Test web interface
    test_web_interface()
    
    # Step 6: Test batch processing
    test_batch_processing()
    
    print("\n🎉 All tests completed successfully!")
    print("\n📋 Next Steps:")
    print("1. Test with your own floorplan: python local_png_to_dxf.py your_image.png")
    print("2. Launch web interface: python app_local.py")
    print("3. Run batch processing: python batch_convert.py input_directory/")
    
    return True

def main():
    """Main function"""
    print("�� Local PNG to DXF Pipeline - Test & Run Guide")
    print("=" * 60)
    
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        
        if command == "test":
            run_comprehensive_test()
        elif command == "install":
            install_dependencies()
        elif command == "convert":
            if len(sys.argv) > 2:
                image_path = sys.argv[2]
                from local_png_to_dxf import process_png_to_dxf
                process_png_to_dxf(image_path)
            else:
                print("Usage: python run_local_pipeline.py convert <image_path>")
        elif command == "web":
            test_web_interface()
        else:
            print("Unknown command. Use: test, install, convert, or web")
    else:
        # Interactive mode
        print("Choose an option:")
        print("1. Run comprehensive test")
        print("2. Install dependencies")
        print("3. Convert specific image")
        print("4. Launch web interface")
        
        choice = input("\nEnter your choice (1-4): ").strip()
        
        if choice == "1":
            run_comprehensive_test()
        elif choice == "2":
            install_dependencies()
        elif choice == "3":
            image_path = input("Enter image path: ").strip()
            if Path(image_path).exists():
                from local_png_to_dxf import process_png_to_dxf
                process_png_to_dxf(image_path)
            else:
                print("Image file not found!")
        elif choice == "4":
            test_web_interface()
        else:
            print("Invalid choice!")

if __name__ == "__main__":
    main()
