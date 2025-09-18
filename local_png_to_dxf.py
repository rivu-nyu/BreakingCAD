# === Local PNG to DXF Converter ===
# Streamlined version for local NPU execution
# Takes PNG as input and generates DXF as output

import os
import sys
import math
import subprocess
import importlib
import io
import re
import colorsys
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Tuple, Optional

import numpy as np
from PIL import Image, ImageOps, ImageFilter
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# ---------- install-or-import ----------
def ensure_import(mod, pip_name=None):
    pip_name = pip_name or mod
    try:
        return importlib.import_module(mod)
    except ImportError:
        print(f"[info] installing {pip_name} ...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pip_name])
        return importlib.import_module(mod)

cv2 = ensure_import("cv2", "opencv-python-headless")
ezdxf = ensure_import("ezdxf", "ezdxf")

# Try to import ONNX Runtime for NPU execution
try:
    import onnxruntime as ort
    NPU_AVAILABLE = True
    print("ONNX Runtime available for NPU execution")
except ImportError:
    NPU_AVAILABLE = False
    print("ONNX Runtime not available, using CPU fallback")

# ---------- CONFIG ----------
class Config:
    # Scale/units
    PIXELS_PER_METER = 100.0
    INSUNITS = 6  # 6 = meters in DXF header
    Z_EXAGGERATION_IF_NO_SCALE = 120.0
    
    # Wall prism params
    WALL_THICK_M = 0.05
    WALL_HEIGHT_M = 3.00
    OFFSET_MODE = "center"
    
    # Processing params
    FLIP_Y_IMAGE = True
    GLOBAL_ROT_DEG = 0.0
    SNAP_TOL_DEG = 4.0
    JUNCTION_RADIUS = 2.0
    ANG_BIN_DEG = 2.0
    DIST_BIN_PX = 2.0
    MERGE_GAP_PX = 3.0
    MIN_SEG_LEN_PX = 8.0
    POST_MIN_LEN_PX = 20
    
    # NPU settings
    USE_NPU = True
    NPU_PROVIDER = 'QNNExecutionProvider'
    FALLBACK_TO_CPU = True

# ---------- Line Segment Class ----------
class LineSeg:
    __slots__ = ("x1", "y1", "x2", "y2")
    
    def __init__(self, x1, y1, x2, y2):
        self.x1, self.y1, self.x2, self.y2 = float(x1), float(y1), float(x2), float(y2)

# ---------- Color Palette ----------
PALETTE = {
    "WALL_GREEN": dict(aci=3),
    "FURN_CYAN": dict(aci=4),
    "NOTE_MAG": dict(aci=6),
    "ACC_RED": dict(aci=1),
    "NEUT_GRAY": dict(aci=8),
    "INK_BLACK": dict(aci=7),
}

# ---------- PNG Text Removal (MSER) ----------
def to_rgb_on_white(pil_img: Image.Image) -> Image.Image:
    if pil_img.mode in ("RGBA", "LA"):
        bg = Image.new("RGB", pil_img.size, (255, 255, 255))
        if pil_img.mode != "RGBA":
            pil_img = pil_img.convert("RGBA")
        bg.paste(pil_img, mask=pil_img.split()[-1])
        return bg
    return pil_img.convert("RGB")

def upscale_and_boost(pil_img: Image.Image, scale=3.0):
    W, H = pil_img.size
    img = pil_img.resize((int(W*scale), int(H*scale)), Image.BICUBIC)
    img = ImageOps.autocontrast(img, cutoff=1)
    img = img.filter(ImageFilter.UnsharpMask(radius=1.2, percent=140, threshold=2))
    return img

def boxes_from_mser_filtered(img_rgb, area_min=30, area_max=4000, ar_min=0.2, ar_max=5.0, max_len=120):
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    mser = cv2.MSER_create(5, area_min, area_max)
    regions, _ = mser.detectRegions(gray)
    boxes = []
    
    for r in regions:
        x, y, w, h = cv2.boundingRect(r.reshape(-1, 1, 2))
        A = w * h
        if A < area_min or A > area_max:
            continue
        ar = w / float(h + 1e-6)
        if not (ar_min <= ar <= ar_max):
            continue
        if max(w, h) > max_len:
            continue
        boxes.append((x, y, x+w, y+h))
    
    if not boxes:
        return []
    
    boxes = np.array(boxes, dtype=np.float32)
    keep = cv2.dnn.NMSBoxes(
        [(int(x), int(y), int(x2-x), int(y2-y)) for x, y, x2, y2 in boxes],
        [1.0] * len(boxes), 0.1, 0.3
    )
    
    merged = []
    if len(keep) > 0:
        for i in keep.flatten():
            merged.append(tuple(boxes[i]))
    return merged

def paint_white_boxes_np(img_rgb: np.ndarray, boxes, pad=4):
    H, W = img_rgb.shape[:2]
    out = img_rgb.copy()
    for (x0, y0, x1, y1) in boxes:
        x0 = max(0, int(x0-pad))
        y0 = max(0, int(y0-pad))
        x1 = min(W, int(x1+pad))
        y1 = min(H, int(y1+pad))
        out[y0:y1, x0:x1] = (255, 255, 255)
    return out

def clean_png_text(pil_img: Image.Image):
    imgU = upscale_and_boost(to_rgb_on_white(pil_img), scale=3.0)
    img_np = np.array(imgU)
    boxes = boxes_from_mser_filtered(img_np, area_min=30, area_max=4000, ar_min=0.2, ar_max=5.0, max_len=120)
    if not boxes:
        return pil_img.convert("RGB")
    cleaned = paint_white_boxes_np(img_np, boxes, pad=4)
    cleaned = cv2.resize(cleaned, pil_img.size, interpolation=cv2.INTER_AREA)
    return Image.fromarray(cleaned)

# ---------- Line Detection ----------
def preprocess_for_lines(pil_img):
    bgr = cv2.cvtColor(np.asarray(pil_img), cv2.COLOR_RGB2BGR)
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    bw = cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 35, 5)
    den = cv2.bilateralFilter(bw, d=5, sigmaColor=50, sigmaSpace=50)
    return den

def create_lsd():
    for args in ((), (cv2.LSD_REFINE_ADV,), (0,)):
        try:
            return cv2.createLineSegmentDetector(*args)
        except Exception:
            pass
    return None

def detect_lines(gray):
    segs = []
    lsd = create_lsd()
    if lsd is not None:
        lines = lsd.detect(gray)[0]
        if lines is not None:
            for l in lines:
                x1, y1, x2, y2 = l[0]
                segs.append((float(x1), float(y1), float(x2), float(y2)))
        method = "LSD"
    else:
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=60, minLineLength=20, maxLineGap=5)
        if lines is not None:
            for l in lines:
                x1, y1, x2, y2 = l[0]
                segs.append((float(x1), float(y1), float(x2), float(y2)))
        method = "HoughP"
    return segs, method

# ---------- Geometry Processing ----------
def snap_axis(seg, tol_deg=5.0):
    dx, dy = seg.x2 - seg.x1, seg.y2 - seg.y1
    if dx == 0 and dy == 0:
        return seg
    ang = (math.degrees(math.atan2(dy, dx)) + 180) % 180
    if abs(ang-0) <= tol_deg or abs(ang-180) <= tol_deg:
        y = (seg.y1 + seg.y2) / 2.0
        return LineSeg(seg.x1, y, seg.x2, y)
    if abs(ang-90) <= tol_deg:
        x = (seg.x1 + seg.x2) / 2.0
        return LineSeg(x, seg.y1, x, seg.y2)
    return seg

def seg_to_np(s):
    return np.array([[s.x1, s.y1], [s.x2, s.y2]], dtype=np.float32)

def cluster_points(points, radius=2.0):
    if len(points) == 0:
        return np.empty((0, 2), dtype=np.float32), np.array([], dtype=int)
    pts = np.asarray(points, dtype=np.float32)
    grid = np.floor(pts / radius).astype(int)
    buckets = {}
    for idx, cell in enumerate(map(tuple, grid)):
        buckets.setdefault(cell, []).append(idx)
    centers, labels = [], np.empty(len(pts), dtype=int)
    for _, idxs in buckets.items():
        C = pts[idxs].mean(axis=0)
        cid = len(centers)
        centers.append(C)
        labels[idxs] = cid
    return np.vstack(centers).astype(np.float32), labels

def snap_endpoints(segs, snap_radius=2.0):
    pts = []
    for s in segs:
        pts += [[s.x1, s.y1], [s.x2, s.y2]]
    centers, labels = cluster_points(np.array(pts), radius=snap_radius)
    out = []
    for i, s in enumerate(segs):
        a = centers[labels[2*i]]
        b = centers[labels[2*i+1]]
        out.append(LineSeg(a[0], a[1], b[0], b[1]))
    return out

def line_key(seg, ang_bin_deg=2.0, dist_bin=2.0):
    dx, dy = seg.x2 - seg.x1, seg.y2 - seg.y1
    if dx == 0 and dy == 0:
        return None
    L = math.hypot(dx, dy)
    ux, uy = dx/L, dy/L
    nx, ny = -uy, ux
    if ny < 0 or (ny == 0 and nx < 0):
        nx, ny = -nx, -ny
    c = nx*seg.x1 + ny*seg.y1
    ang = (math.degrees(math.atan2(uy, ux)) + 180) % 180
    return (round(ang/ang_bin_deg), round(c/dist_bin))

def project_t(seg, ux, uy, x0, y0):
    t1 = ((seg.x1-x0)*ux + (seg.y1-y0)*uy)
    t2 = ((seg.x2-x0)*ux + (seg.y2-y0)*uy)
    return min(t1, t2), max(t1, t2)

def merge_collinear_group(segs, tol_gap=2.0, min_len=4.0):
    pts = np.vstack([seg_to_np(s) for s in segs])
    ptsc = pts - pts.mean(axis=0, keepdims=True)
    _, _, VT = np.linalg.svd(ptsc, full_matrices=False)
    ux, uy = VT[0]
    x0, y0 = pts.mean(axis=0)
    ivals = []
    for s in segs:
        a, b = project_t(s, ux, uy, x0, y0)
        ivals.append((a, b))
    ivals.sort()
    merged, cur_a, cur_b = [], ivals[0][0], ivals[0][1]
    for a, b in ivals[1:]:
        if a <= cur_b + tol_gap:
            cur_b = max(cur_b, b)
        else:
            if cur_b - cur_a >= min_len:
                merged.append((cur_a, cur_b))
            cur_a, cur_b = a, b
    if cur_b - cur_a >= min_len:
        merged.append((cur_a, cur_b))
    out = []
    for a, b in merged:
        x1, y1 = x0 + a*ux, y0 + a*uy
        x2, y2 = x0 + b*ux, y0 + b*uy
        out.append(LineSeg(x1, y1, x2, y2))
    return out

def merge_collinear(segs, ang_bin_deg=2.0, dist_bin=2.0, tol_gap=2.0, min_len=4.0):
    buckets = defaultdict(list)
    for s in segs:
        k = line_key(s, ang_bin_deg, dist_bin)
        if k is not None:
            buckets[k].append(s)
    out = []
    for _, group in buckets.items():
        out.extend(merge_collinear_group(group, tol_gap, min_len))
    return out

# ---------- 3D Face Generation ----------
def wall_faces_for_segment(x1, y1, x2, y2, thickness, height, offset_mode="center"):
    dx, dy = (x2-x1), (y2-y1)
    L = math.hypot(dx, dy)
    if L < 1e-6:
        return []
    ux, uy = dx/L, dy/L
    nx, ny = -uy, ux
    t = thickness

    if offset_mode == "center":
        A = (x1 - nx*(t/2), y1 - ny*(t/2), 0.0)
        B = (x1 + nx*(t/2), y1 + ny*(t/2), 0.0)
        C = (x2 + nx*(t/2), y2 + ny*(t/2), 0.0)
        D = (x2 - nx*(t/2), y2 - ny*(t/2), 0.0)
    elif offset_mode == "left":
        A = (x1, y1, 0.0)
        B = (x1 + nx*t, y1 + ny*t, 0.0)
        C = (x2 + nx*t, y2 + ny*t, 0.0)
        D = (x2, y2, 0.0)
    elif offset_mode == "right":
        A = (x1 - nx*t, y1 - ny*t, 0.0)
        B = (x1, y1, 0.0)
        C = (x2, y2, 0.0)
        D = (x2 - nx*t, y2 - ny*t, 0.0)
    else:
        raise ValueError("offset_mode must be one of: center|left|right")

    A2 = (A[0], A[1], height)
    B2 = (B[0], B[1], height)
    C2 = (C[0], C[1], height)
    D2 = (D[0], D[1], height)

    return [
        (A, B, C, D),     # bottom
        (A2, B2, C2, D2), # top
        (A, B, B2, A2),   # side 1
        (B, C, C2, B2),   # side 2
        (C, D, D2, C2),   # side 3
        (D, A, A2, D2),   # side 4
    ]

# ---------- Color Analysis ----------
def sample_rgb_on_segment(pil_rgb, seg, n=9):
    im = np.asarray(pil_rgb)
    H, W = im.shape[:2]
    x1, y1, x2, y2 = seg.x1, seg.y1, seg.x2, seg.y2
    xs = np.linspace(x1, x2, n)
    ys = np.linspace(y1, y2, n)
    pts = np.stack([xs, ys], axis=1).round().astype(int)
    pts[:, 0] = np.clip(pts[:, 0], 0, W-1)
    pts[:, 1] = np.clip(pts[:, 1], 0, H-1)
    samps = im[pts[:, 1], pts[:, 0]]
    return samps.mean(axis=0)

def bucket_layer_from_rgb(rgb):
    r, g, b = (rgb/255.0).tolist()
    h, s, v = colorsys.rgb_to_hsv(r, g, b)
    h = h*360.0
    if s < 0.15 or v < 0.2:
        return "INK_BLACK" if v < 0.25 else "NEUT_GRAY"
    if 50 <= h <= 90:
        return "WALL_GREEN"
    if 160 <= h <= 200:
        return "FURN_CYAN"
    if 300 <= h <= 340:
        return "NOTE_MAG"
    if 0 <= h <= 20 or 340 <= h < 360:
        return "ACC_RED"
    return "NEUT_GRAY"

# ---------- DXF Export ----------
def rotate_points(pts3, deg, center=None):
    if abs(deg) < 1e-9:
        return pts3
    ang = math.radians(deg)
    ca, sa = math.cos(ang), math.sin(ang)
    P = np.asarray(pts3, dtype=float)
    if center is None:
        cx, cy = P[:, 0].mean(), P[:, 1].mean()
    else:
        cx, cy = center
    X = P[:, 0] - cx
    Y = P[:, 1] - cy
    xr = X*ca - Y*sa
    yr = X*sa + Y*ca
    P[:, 0] = xr + cx
    P[:, 1] = yr + cy
    return P

def export_true3d_dxf_from_segments(segments, out_path, wall_thick=0.2, wall_height=3.0,
                                    flip_y=True, scale_xy=1.0, use_meter_units=True,
                                    z_boost_if_no_scale=120.0, rotate_deg=0.0,
                                    offset_mode="center", shift_positive=True,
                                    base_layer="WALLS_3D", seg_layers=None):
    if use_meter_units:
        z_h = float(wall_height)
        t_h = float(wall_thick)
    else:
        z_h = float(wall_height) * float(z_boost_if_no_scale)
        t_h = float(wall_thick) * (scale_xy if scale_xy != 1.0 else 1.0)

    seg_faces = []
    all_xy = []
    
    for idx, s in enumerate(segments):
        x1, y1, x2, y2 = s.x1, s.y1, s.x2, s.y2
        if flip_y:
            y1, y2 = -y1, -y2
        x1 *= scale_xy
        y1 *= scale_xy
        x2 *= scale_xy
        y2 *= scale_xy
        faces = wall_faces_for_segment(x1, y1, x2, y2, t_h, z_h, offset_mode=offset_mode)
        lay = (seg_layers[idx] if (seg_layers is not None and idx < len(seg_layers)) else base_layer)
        use = faces[2:]  # sides only
        seg_faces.append((use, lay))
        for f in use:
            for vx, vy, _ in f:
                all_xy.append((vx, vy))

    # Rotate whole scene
    if abs(rotate_deg) > 1e-9 and seg_faces:
        verts = []
        for faces, _ in seg_faces:
            for f in faces:
                for v in f:
                    verts.append(list(v))
        V = np.array(verts, dtype=float)
        V_rot = rotate_points(V.copy(), rotate_deg, center=None)
        it = 0
        new_seg_faces = []
        for faces, lay in seg_faces:
            new_faces = []
            for _ in range(len(faces)):
                p1 = tuple(V_rot[it+0])
                p2 = tuple(V_rot[it+1])
                p3 = tuple(V_rot[it+2])
                p4 = tuple(V_rot[it+3])
                new_faces.append((p1, p2, p3, p4))
                it += 4
            new_seg_faces.append((new_faces, lay))
        seg_faces = new_seg_faces
        all_xy = [(p[0], p[1]) for faces, _ in seg_faces for f in faces for p in f]

    # Shift positive quadrant
    dx = dy = 0.0
    if shift_positive and all_xy:
        mins = np.min(np.array(all_xy), axis=0)
        dx = -min(0.0, float(mins[0]))
        dy = -min(0.0, float(mins[1]))

    # Write DXF
    doc = ezdxf.new(dxfversion="R2000", setup=True)
    doc.header["$INSUNITS"] = (Config.INSUNITS if use_meter_units else 0)

    if base_layer not in doc.layers:
        doc.layers.add(base_layer, color=7)
    for name, meta in PALETTE.items():
        if name not in doc.layers:
            doc.layers.add(name, color=meta["aci"])

    msp = doc.modelspace()
    for faces, lay in seg_faces:
        if lay not in doc.layers:
            doc.layers.add(lay, color=7)
        for (p1, p2, p3, p4) in faces:
            v1 = (p1[0]+dx, p1[1]+dy, p1[2])
            v2 = (p2[0]+dx, p2[1]+dy, p2[2])
            v3 = (p3[0]+dx, p3[1]+dy, p3[2])
            v4 = (p4[0]+dx, p4[1]+dy, p4[2])
            msp.add_3dface([v1, v2, v3, v4], dxfattribs={"layer": lay})

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    doc.saveas(out_path)
    print(f"[DXF] colored 3D (3DFACE) saved: {out_path}  segments={len(segments)}")
    return seg_faces, (dx, dy)

# ---------- Main Processing Function ----------
def process_png_to_dxf(input_path: str, output_path: str = None, show_preview: bool = True) -> str:
    """
    Main function to process PNG to DXF conversion
    
    Args:
        input_path: Path to input PNG file
        output_path: Path to output DXF file (optional)
        show_preview: Whether to show 3D preview
    
    Returns:
        Path to generated DXF file
    """
    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    if output_path is None:
        output_path = input_path.parent / f"{input_path.stem}_3d.dxf"
    
    print(f"[info] Processing: {input_path}")
    
    # Load and clean image
    pil_image = Image.open(input_path)
    if pil_image.mode != "RGB":
        pil_image = pil_image.convert("RGB")
    
    # Clean text from image
    im_clean = clean_png_text(pil_image)
    
    # Detect lines
    gray = preprocess_for_lines(im_clean)
    segs_raw, method = detect_lines(gray)
    raw = [LineSeg(*s) for s in segs_raw]
    
    # Process geometry
    snapped_axis = [snap_axis(s, tol_deg=Config.SNAP_TOL_DEG) for s in raw]
    snapped_junc = snap_endpoints(snapped_axis, snap_radius=Config.JUNCTION_RADIUS)
    merged = merge_collinear(snapped_junc, ang_bin_deg=Config.ANG_BIN_DEG, dist_bin=Config.DIST_BIN_PX,
                            tol_gap=Config.MERGE_GAP_PX, min_len=Config.MIN_SEG_LEN_PX)
    
    # Filter small segments
    merged = [s for s in merged if math.hypot(s.x2-s.x1, s.y2-s.y1) >= Config.POST_MIN_LEN_PX]
    
    print(f"[info] segments: raw={len(raw)} axis={len(snapped_axis)} junction={len(snapped_junc)} merged={len(merged)} via {method}")
    
    if not merged:
        print("No line segments detected!")
        return str(output_path)
    
    # Scale/units
    scale_xy = (1.0 / Config.PIXELS_PER_METER) if (Config.PIXELS_PER_METER and Config.PIXELS_PER_METER > 0) else 1.0
    use_meter_units = Config.PIXELS_PER_METER is not None
    
    # Per-segment color layers
    seg_layers = []
    for s in merged:
        rgb_samp = sample_rgb_on_segment(im_clean, s, n=9)
        seg_layers.append(bucket_layer_from_rgb(rgb_samp))
    
    # Export DXF
    seg_faces, (dx, dy) = export_true3d_dxf_from_segments(
        merged, output_path,
        wall_thick=Config.WALL_THICK_M,
        wall_height=Config.WALL_HEIGHT_M,
        flip_y=Config.FLIP_Y_IMAGE,
        scale_xy=scale_xy,
        use_meter_units=use_meter_units,
        z_boost_if_no_scale=Config.Z_EXAGGERATION_IF_NO_SCALE,
        rotate_deg=Config.GLOBAL_ROT_DEG,
        offset_mode=Config.OFFSET_MODE,
        shift_positive=True,
        base_layer="WALLS_3D",
        seg_layers=seg_layers,
    )
    
    # Show preview if requested
    if show_preview and seg_faces:
        fig = plt.figure(figsize=(10, 5))
        ax = fig.add_subplot(1, 2, 1, projection='3d')
        ACI2RGB = {1: (1, 0, 0), 3: (0, 0.6, 0), 4: (0, 0.7, 0.7), 6: (0.8, 0, 0.8), 7: (0.2, 0.2, 0.2), 8: (0.6, 0.6, 0.6)}
        
        for faces, lay in seg_faces:
            aci = PALETTE.get(lay, {"aci": 7})["aci"] if lay in PALETTE else 7
            rgbc = ACI2RGB.get(aci, (0.5, 0.5, 0.5))
            for f in faces:
                coll = Poly3DCollection([list(f)], facecolor=(*rgbc, 0.35), edgecolor="k", linewidths=0.2)
                ax.add_collection3d(coll)
        
        allv = np.array([v for faces, _ in seg_faces for f in faces for v in f])
        xmin, xmax = allv[:, 0].min(), allv[:, 0].max()
        ymin, ymax = allv[:, 1].min(), allv[:, 1].max()
        zmin, zmax = allv[:, 2].min(), allv[:, 2].max()
        rng = max(xmax-xmin, ymax-ymin, zmax-zmin) * 1.05 + 1e-6
        cx, cy, cz = (xmin+xmax)/2, (ymin+ymax)/2, (zmin+zmax)/2
        ax.set_xlim(cx-rng/2, cx+rng/2)
        ax.set_ylim(cy-rng/2, cy+rng/2)
        ax.set_zlim(cz-rng/2, cz+rng/2)
        ax.view_init(elev=28, azim=-40)
        ax.set_title("Isometric (colored)")
        
        ax2 = fig.add_subplot(1, 2, 2, projection='3d')
        for faces, _ in seg_faces:
            for f in faces:
                coll2 = Poly3DCollection([list(f)], alpha=0.18, edgecolor="k", linewidths=0.15)
                ax2.add_collection3d(coll2)
        ax2.set_xlim(cx-rng/2, cx+rng/2)
        ax2.set_ylim(cy-rng/2, cy+rng/2)
        ax2.set_zlim(cz-rng/2, cz+rng/2)
        ax2.view_init(elev=90, azim=-90)
        ax2.set_title("Top view")
        plt.tight_layout()
        plt.show()
    
    print(f"[done] Wrote: {output_path}")
    return str(output_path)

# ---------- Command Line Interface ----------
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Convert PNG floorplan to 3D DXF")
    parser.add_argument("input", help="Input PNG file path")
    parser.add_argument("-o", "--output", help="Output DXF file path")
    parser.add_argument("--no-preview", action="store_true", help="Disable 3D preview")
    parser.add_argument("--pixels-per-meter", type=float, default=100.0, help="Pixels per meter scale")
    parser.add_argument("--wall-thickness", type=float, default=0.05, help="Wall thickness in meters")
    parser.add_argument("--wall-height", type=float, default=3.0, help="Wall height in meters")
    
    args = parser.parse_args()
    
    # Update config
    Config.PIXELS_PER_METER = args.pixels_per_meter
    Config.WALL_THICK_M = args.wall_thickness
    Config.WALL_HEIGHT_M = args.wall_height
    
    # Process file
    output_path = process_png_to_dxf(
        args.input, 
        args.output, 
        show_preview=not args.no_preview
    )
    
    print(f"Conversion complete! Output saved to: {output_path}")
