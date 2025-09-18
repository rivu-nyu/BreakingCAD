# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['app_local.py'],
    pathex=[],
    binaries=[],
    datas=[('src', 'src'), ('models', 'models'), ('build', 'build')],
    hiddenimports=['gradio', 'gradio.components', 'gradio.blocks', 'gradio.themes', 'gradio.interface', 'matplotlib', 'matplotlib.pyplot', 'onnxruntime', 'torch', 'cv2', 'PIL', 'PIL.Image', 'ezdxf', 'numpy', 'psutil', 'uvicorn', 'fastapi', 'starlette', 'local_png_to_dxf', 'standalone', 'src', 'src.exporter', 'src.geometry', 'src.vision'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='FloorPlanConverter_Fallback',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='FloorPlanConverter_Fallback',
)
