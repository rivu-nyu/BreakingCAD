#!/usr/bin/env python3
"""
Build script using cx_Freeze for app_local.py
"""

import os
import sys
import subprocess
import shutil
import time

def clean_build_dirs():
    """Clean previous build artifacts"""
    dirs_to_clean = ['build', 'dist']
    for dir_name in dirs_to_clean:
        if os.path.exists(dir_name):
            print(f"Cleaning {dir_name}...")
            shutil.rmtree(dir_name)

def check_cx_freeze():
    """Check if cx_Freeze is installed"""
    try:
        import cx_Freeze
        print(f"cx_Freeze version: {cx_Freeze.version}")
        return True
    except ImportError:
        print("cx_Freeze not found! Please install it with: pip install cx_Freeze")
        return False

def build_with_cx_freeze():
    """Build using cx_Freeze"""
    print("\nBuilding with cx_Freeze...")
    
    cmd = [sys.executable, "setup_cx_freeze.py", "build"]
    
    try:
        print("Running cx_Freeze build...")
        print("This may take several minutes...")
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("Build successful!")
        
        # Find the created executable
        build_dir = None
        for item in os.listdir("build"):
            if item.startswith("exe."):
                build_dir = os.path.join("build", item)
                break
        
        if build_dir and os.path.exists(build_dir):
            exe_path = os.path.join(build_dir, "AppLocalCxFreeze.exe")
            if os.path.exists(exe_path):
                print(f"Executable created: {os.path.abspath(exe_path)}")
                return build_dir, exe_path
        
        print("Warning: Could not locate the executable")
        return None, None
        
    except subprocess.CalledProcessError as e:
        print(f"Build failed with return code: {e.returncode}")
        if e.stderr:
            print("ERROR OUTPUT:")
            print(e.stderr)
        if e.stdout:
            print("STANDARD OUTPUT:")
            print(e.stdout)
        return None, None

def test_executable(exe_path):
    """Test the built executable"""
    if not exe_path or not os.path.exists(exe_path):
        print("Executable not found!")
        return False
    
    print(f"\nTesting executable: {exe_path}")
    file_size_mb = os.path.getsize(exe_path) / (1024*1024)
    print(f"File size: {file_size_mb:.1f} MB")
    
    try:
        print("Testing executable startup...")
        process = subprocess.Popen([exe_path], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        
        # Wait for startup
        time.sleep(8)
        
        if process.poll() is None:
            print("Executable started successfully!")
            print("Application should be running at http://localhost:7860")
            process.terminate()
            time.sleep(2)
            if process.poll() is None:
                process.kill()
            return True
        else:
            stdout, stderr = process.communicate()
            print(f"Executable failed. Error: {stderr}")
            print(f"Output: {stdout}")
            return False
            
    except Exception as e:
        print(f"Error testing executable: {e}")
        return False

def create_launcher(build_dir):
    """Create launcher script"""
    if not build_dir:
        return
        
    launcher_content = f'''@echo off
echo Starting App Local Converter (cx_Freeze Version)...
echo.
echo This version built with cx_Freeze should handle dependencies better.
echo The application will open in your web browser.
echo If it doesn't open automatically, go to: http://localhost:7860
echo.
echo Press Ctrl+C to stop the application.
echo.
cd /d "{os.path.abspath(build_dir)}"
AppLocalCxFreeze.exe
pause
'''
    
    launcher_path = os.path.join(build_dir, "launch_cx_freeze.bat")
    with open(launcher_path, "w") as f:
        f.write(launcher_content)
    
    print(f"Created launcher: {launcher_path}")

def main():
    """Main build process"""
    print("Building App Local Converter with cx_Freeze")
    print("=" * 60)
    
    # Check cx_Freeze installation
    if not check_cx_freeze():
        return
    
    # Clean previous builds
    clean_build_dirs()
    
    # Build with cx_Freeze
    build_dir, exe_path = build_with_cx_freeze()
    
    if build_dir and exe_path:
        # Test the executable
        if test_executable(exe_path):
            # Create launcher
            create_launcher(build_dir)
            
            print("\ncx_Freeze BUILD COMPLETED SUCCESSFULLY!")
            print("Files created:")
            print(f"  {exe_path}")
            print(f"  {os.path.join(build_dir, 'launch_cx_freeze.bat')}")
            print("\nTo use:")
            print(f"  1. Go to: {build_dir}")
            print("  2. Run launch_cx_freeze.bat")
            print("  3. Wait for browser to open at http://localhost:7860")
            print("\ncx_Freeze often handles complex dependencies better than PyInstaller!")
        else:
            print("\nBuild completed but test failed.")
    else:
        print("\nBuild failed. Check error messages above.")

if __name__ == "__main__":
    main()
