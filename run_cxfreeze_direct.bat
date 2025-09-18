@echo off
echo Running direct cx_Freeze command...
cxfreeze app_local.py --target-dir=build_cx_simple --target-name=AppLocalDirect.exe --include-path=src --include-path=models --packages=gradio --packages=gradio_client --packages=numpy --packages=PIL --packages=cv2 --packages=ezdxf --packages=matplotlib --packages=torch --packages=onnxruntime --packages=local_png_to_dxf --packages=standalone --packages=src
pause
