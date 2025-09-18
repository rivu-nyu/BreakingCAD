# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['app_local.py'],
    pathex=[],
    binaries=[],
    datas=[('src', 'src'), ('models', 'models'), ('build', 'build')],
    hiddenimports=['gradio', 'gradio.components', 'gradio.blocks', 'gradio.themes', 'matplotlib', 'matplotlib.pyplot', 'matplotlib.patches', 'matplotlib.colors', 'onnxruntime', 'torch', 'cv2', 'PIL', 'PIL.Image', 'ezdxf', 'numpy', 'psutil', 'local_png_to_dxf', 'standalone', 'src', 'src.exporter', 'src.geometry', 'src.vision'],
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
    a.binaries,
    a.datas,
    [],
    name='FloorPlanConverter',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
