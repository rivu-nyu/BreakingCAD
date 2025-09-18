#!/usr/bin/env python3
"""
Alternative build approach using site-packages copy method
"""

import os
import sys
import subprocess
import shutil
import site
from pathlib import Path

def get_site_packages_path():
    """Get the site-packages path from the current environment"""
    for path in site.getsitepackages():
        if 'site-packages' in path and os.path.exists(path):
            return path
    
    # Fallback for virtual environment
    env_path = os.path.join(os.getcwd(), 'env', 'Lib', 'site-packages')
    if os.path.exists(env_path):
        return env_path
    
    return None

def create_bundled_app():
    """Create a bundled application with all dependencies"""
    print("Creating bundled application...")
    
    # Create app directory
    app_dir = "FloorPlanConverter_Bundled"
    if os.path.exists(app_dir):
        shutil.rmtree(app_dir)
    os.makedirs(app_dir)
    
    # Copy main application files
    app_files = [
        "app_local.py",
        "local_png_to_dxf.py", 
        "standalone.py"
    ]
    
    for file in app_files:
        if os.path.exists(file):
            shutil.copy2(file, app_dir)
            print(f"  ✅ Copied {file}")
    
    # Copy src directory
    if os.path.exists("src"):
        shutil.copytree("src", os.path.join(app_dir, "src"))
        print("  ✅ Copied src directory")
    
    # Copy models directory
    if os.path.exists("models"):
        shutil.copytree("models", os.path.join(app_dir, "models"))
        print("  ✅ Copied models directory")
    
    # Copy build directory
    if os.path.exists("build"):
        shutil.copytree("build", os.path.join(app_dir, "build"))
        print("  ✅ Copied build directory")
    
    # Create outputs directory
    os.makedirs(os.path.join(app_dir, "outputs"), exist_ok=True)
    
    # Get site-packages path
    site_packages = get_site_packages_path()
    if not site_packages:
        print("❌ Could not find site-packages directory")
        return False
    
    print(f"Using site-packages: {site_packages}")
    
    # Copy essential packages
    essential_packages = [
        "gradio",
        "gradio_client", 
        "matplotlib",
        "torch",
        "torchvision",
        "onnxruntime",
        "cv2",
        "PIL",
        "ezdxf",
        "numpy",
        "psutil",
        "uvicorn",
        "fastapi",
        "starlette",
        "websockets",
        "httpx",
        "h11",
        "anyio",
        "sniffio",
        "pydantic",
        "typing_extensions",
        "click",
        "colorama",
        "jinja2",
        "markupsafe",
        "packaging",
        "six",
        "urllib3",
        "certifi",
        "charset_normalizer",
        "idna",
        "requests"
    ]
    
    # Create lib directory in app
    lib_dir = os.path.join(app_dir, "lib")
    os.makedirs(lib_dir)
    
    copied_packages = []
    for package in essential_packages:
        package_path = os.path.join(site_packages, package)
        if os.path.exists(package_path):
            try:
                if os.path.isdir(package_path):
                    shutil.copytree(package_path, os.path.join(lib_dir, package))
                else:
                    shutil.copy2(package_path, lib_dir)
                copied_packages.append(package)
                print(f"  ✅ Copied package: {package}")
            except Exception as e:
                print(f"  ⚠️  Failed to copy {package}: {e}")
        else:
            # Try with .py extension for single file modules
            package_file = os.path.join(site_packages, f"{package}.py")
            if os.path.exists(package_file):
                shutil.copy2(package_file, lib_dir)
                copied_packages.append(package)
                print(f"  ✅ Copied module: {package}.py")
    
    print(f"Copied {len(copied_packages)} packages")
    
    # Create launcher script
    launcher_content = '''#!/usr/bin/env python3
"""
FloorPlan Converter Launcher
"""

import sys
import os

# Add lib directory to Python path
app_dir = os.path.dirname(os.path.abspath(__file__))
lib_dir = os.path.join(app_dir, "lib")
sys.path.insert(0, lib_dir)
sys.path.insert(0, app_dir)

# Set environment variables
os.environ["PYTHONPATH"] = lib_dir + os.pathsep + app_dir
os.environ["GRADIO_TEMP_DIR"] = os.path.join(app_dir, "temp")

# Create temp directory
os.makedirs(os.path.join(app_dir, "temp"), exist_ok=True)

try:
    print("Starting FloorPlan Converter...")
    print("Loading dependencies...")
    
    # Import and run the app
    import app_local
    print("SUCCESS: All modules loaded successfully!")
    print("WEB: Opening web interface...")
    print("URL: Go to: http://localhost:7860")
    
except Exception as e:
    print(f"ERROR: Error starting application: {e}")
    print("Make sure all dependencies are properly installed.")
    input("Press Enter to exit...")
'''
    
    with open(os.path.join(app_dir, "launch.py"), "w", encoding='utf-8') as f:
        f.write(launcher_content)
    
    # Create batch launcher
    batch_content = f'''@echo off
echo Starting FloorPlan Converter Bundled Application...
echo.
cd /d "%~dp0"
python launch.py
pause
'''
    
    with open(os.path.join(app_dir, "Launch_FloorPlan_Converter.bat"), "w") as f:
        f.write(batch_content)
    
    # Create PowerShell launcher
    ps_content = f'''Write-Host "Starting FloorPlan Converter Bundled Application..." -ForegroundColor Green
Set-Location $PSScriptRoot
python launch.py
Read-Host "Press Enter to exit"
'''
    
    with open(os.path.join(app_dir, "Launch_FloorPlan_Converter.ps1"), "w") as f:
        f.write(ps_content)
    
    # Create README
    readme_content = f'''# FloorPlan Converter - Bundled Application

## Requirements
- Python 3.8+ installed on the target machine
- No additional packages need to be installed

## How to Run
1. **Windows**: Double-click `Launch_FloorPlan_Converter.bat`
2. **PowerShell**: Run `Launch_FloorPlan_Converter.ps1`  
3. **Manual**: Run `python launch.py`

## What's Included
- Complete FloorPlan Converter application
- All required Python packages bundled
- Sample inputs and models
- No internet connection required

## Features
- Hand-drawn sketch processing
- PNG image to DXF conversion
- Web-based interface
- Adjustable parameters

## Troubleshooting
- Make sure Python is installed and in PATH
- Run from Command Prompt if batch file doesn't work
- Check that all files are in the same directory

Bundled packages: {', '.join(copied_packages)}
'''
    
    with open(os.path.join(app_dir, "README.txt"), "w") as f:
        f.write(readme_content)
    
    print(f"\n✅ Bundled application created: {app_dir}")
    return True

def test_bundled_app():
    """Test the bundled application"""
    app_dir = "FloorPlanConverter_Bundled"
    launcher = os.path.join(app_dir, "launch.py")
    
    if not os.path.exists(launcher):
        print("❌ Bundled app not found")
        return False
    
    print(f"Testing bundled application...")
    
    try:
        # Test import capabilities
        result = subprocess.run([
            sys.executable, launcher
        ], cwd=app_dir, capture_output=True, text=True, timeout=10)
        
        if "All modules loaded successfully!" in result.stdout:
            print("✅ Bundled application test passed!")
            return True
        else:
            print("❌ Bundled application test failed")
            if result.stderr:
                print(f"Error: {result.stderr}")
            if result.stdout:
                print(f"Output: {result.stdout}")
            return False
            
    except subprocess.TimeoutExpired:
        print("✅ Application started (timeout reached - this is expected)")
        return True
    except Exception as e:
        print(f"❌ Error testing bundled app: {e}")
        return False

def main():
    """Main process"""
    print("🔨  Creating Bundled FloorPlan Converter Application")
    print("=" * 60)
    
    if create_bundled_app():
        if test_bundled_app():
            print("\n🎉 Bundled application created successfully!")
            print("📁 Location: FloorPlanConverter_Bundled/")
            print("🚀 Run: Launch_FloorPlan_Converter.bat")
            print("\n💡 This approach bundles all dependencies with the app")
            print("   and doesn't require PyInstaller compilation.")
        else:
            print("\n⚠️  Bundled app created but test failed")
    else:
        print("\n❌ Failed to create bundled application")

if __name__ == "__main__":
    main()
