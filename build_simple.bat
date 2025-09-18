@echo off
echo Building FloorPlan Converter...

pyinstaller --onefile --windowed ^
--name=FloorPlanConverter ^
--add-data="src;src" ^
--add-data="models;models" ^
--hidden-import=gradio ^
--hidden-import=matplotlib ^
--hidden-import=onnxruntime ^
--hidden-import=torch ^
--hidden-import=cv2 ^
--hidden-import=PIL ^
--hidden-import=ezdxf ^
app_local.py

echo Build complete! Check the dist folder.
pause
