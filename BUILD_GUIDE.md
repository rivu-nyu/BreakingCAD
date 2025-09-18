# FloorPlan Converter - Build Guide

This guide will help you create a standalone Windows executable for the FloorPlan Converter application.

## Prerequisites

1. **Python Environment**: Make sure you have Python 3.8+ installed
2. **Dependencies**: Install all required packages
3. **PyInstaller**: Install PyInstaller for building executables

## Quick Start

### Option 1: Automated Build (Recommended)

1. **Run the automated build script**:
   ```bash
   python build_executable.py
   ```

2. **Create distribution package**:
   ```bash
   python create_distribution.py
   ```

### Option 2: Manual Build

1. **Run the simple batch file**:
   ```bash
   build_simple.bat
   ```

## Detailed Steps

### Step 1: Install Dependencies

```bash
# Install all required packages
pip install -r requirements.txt

# Ensure PyInstaller is installed
pip install pyinstaller
```

### Step 2: Prepare Your Environment

Make sure these directories exist:
- `src/` - Contains your source code
- `models/` - Contains ML models (if any)
- `build/` - Contains compiled models (if any)
- `outputs/` - Will be created automatically

### Step 3: Build the Executable

#### Method A: Using the automated script
```bash
python build_executable.py
```

#### Method B: Using the batch file
```bash
build_simple.bat
```

#### Method C: Manual PyInstaller command
```bash
pyinstaller --onefile --windowed --name=FloorPlanConverter --add-data="src;src" --add-data="models;models" --hidden-import=gradio --hidden-import=matplotlib --hidden-import=onnxruntime --hidden-import=torch --hidden-import=cv2 --hidden-import=PIL --hidden-import=ezdxf app_local.py
```

### Step 4: Test the Executable

1. Navigate to the `dist` folder
2. Run `FloorPlanConverter.exe`
3. Open your browser to `http://localhost:7860`
4. Test both processing modes:
   - Hand-drawn sketch processing
   - PNG image processing

### Step 5: Create Distribution Package

```bash
python create_distribution.py
```

This will create:
- `FloorPlanConverter_v1.0/` folder with all necessary files
- `FloorPlanConverter_v1.0.zip` for easy distribution

## File Structure After Build

```
dist/
├── FloorPlanConverter.exe          # Main executable
├── README.txt                      # User instructions
└── FloorPlanConverter_v1.0/        # Distribution folder
    ├── FloorPlanConverter.exe
    ├── launch_converter.bat
    ├── README.txt
    ├── outputs/                    # Output directory
    └── sample_inputs/              # Sample images
```

## Troubleshooting

### Common Issues

1. **"Failed to execute script"**
   - Remove `--windowed` flag to see console output
   - Check for missing dependencies

2. **Missing DLL files**
   - Use `--collect-all` flag
   - Manually add missing DLLs to the build

3. **Large file size**
   - Use `--exclude-module` for unused libraries
   - Consider using `--onedir` instead of `--onefile`

4. **Slow startup**
   - This is normal for the first launch
   - Subsequent launches should be faster

### Debug Mode

To see console output and debug issues:

1. Edit `build_executable.py`
2. Remove `"--windowed"` from the cmd list
3. Rebuild the executable

### File Size Optimization

For smaller executables, modify the build command:

```bash
pyinstaller --onefile --strip --exclude-module=pytest --exclude-module=jupyter --name=FloorPlanConverter app_local.py
```

## Distribution

### For End Users

1. **Share the zip file**: `FloorPlanConverter_v1.0.zip`
2. **Instructions for users**:
   - Extract the zip file
   - Run `FloorPlanConverter.exe`
   - Open browser to `http://localhost:7860`

### For Development

1. **Test on clean Windows machine**
2. **Test without internet connection**
3. **Test both processing modes**
4. **Verify all features work correctly**

## Advanced Configuration

### Custom PyInstaller Options

You can modify `build_executable.py` to add custom options:

```python
# Add custom hidden imports
"--hidden-import=custom_module",

# Exclude unused modules
"--exclude-module=unused_module",

# Add custom data files
"--add-data=custom_data;custom_data",
```

### Environment Variables

Set these environment variables for better control:

```bash
# Disable console window
PYINSTALLER_WINDOWED=1

# Enable debug mode
PYINSTALLER_DEBUG=1
```

## Support

If you encounter issues:

1. Check the console output (remove `--windowed` flag)
2. Verify all dependencies are installed
3. Test on a clean Windows machine
4. Check PyInstaller documentation for specific error messages

## Version History

- v1.0: Initial release with hand-drawn sketch and PNG processing support
