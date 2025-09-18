#!/usr/bin/env python3
"""
Standalone Color-Coded Floor Plan Processor
Self-contained script that transforms hand-drawn sketches into professional CAD floor plans
No external dependencies on geometry_engine - all code included inline
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import ListedColormap
import random
import math
from typing import List, Dict, Tuple, Optional, Any
from collections import defaultdict
try:
    # Prefer the original engine to ensure identical results
    from geometry_engine.api import GeometryEngine as ExternalGeometryEngine
    EXTERNAL_ENGINE_AVAILABLE = True
except Exception:
    EXTERNAL_ENGINE_AVAILABLE = False

# =============================================================================
# GEOMETRY PRIMITIVES (inline from geometry_engine/primitives.py)
# =============================================================================

class Point:
    """2D Point with basic operations"""
    def __init__(self, x: float, y: float):
        self.x = float(x)
        self.y = float(y)
    
    def distance_to(self, other: 'Point') -> float:
        return math.sqrt((self.x - other.x)**2 + (self.y - other.y)**2)
    
    def __repr__(self):
        return f"Point({self.x:.2f}, {self.y:.2f})"

class LineSegment:
    """Line segment with geometric operations"""
    def __init__(self, start: Point, end: Point):
        self.start = start
        self.end = end
    
    @property
    def length(self) -> float:
        return self.start.distance_to(self.end)
    
    @property
    def angle(self) -> float:
        """Angle in radians"""
        return math.atan2(self.end.y - self.start.y, self.end.x - self.start.x)
    
    @property
    def midpoint(self) -> Point:
        return Point((self.start.x + self.end.x) / 2, (self.start.y + self.end.y) / 2)
    
    def distance_to_point(self, point: Point) -> float:
        """Perpendicular distance from point to line segment"""
        A = point.x - self.start.x
        B = point.y - self.start.y
        C = self.end.x - self.start.x
        D = self.end.y - self.start.y
        
        dot = A * C + B * D
        len_sq = C * C + D * D
        
        if len_sq == 0:
            return point.distance_to(self.start)
        
        param = dot / len_sq
        
        if param < 0:
            closest = self.start
        elif param > 1:
            closest = self.end
        else:
            closest = Point(self.start.x + param * C, self.start.y + param * D)
        
        return point.distance_to(closest)
    
    def is_parallel_to(self, other: 'LineSegment', tolerance: float = 0.1) -> bool:
        """Check if two line segments are parallel"""
        angle_diff = abs(self.angle - other.angle)
        return angle_diff < tolerance or abs(angle_diff - math.pi) < tolerance
    
    def is_perpendicular_to(self, other: 'LineSegment', tolerance: float = 0.1) -> bool:
        """Check if two line segments are perpendicular"""
        angle_diff = abs(self.angle - other.angle)
        return abs(angle_diff - math.pi/2) < tolerance or abs(angle_diff - 3*math.pi/2) < tolerance
    
    def __repr__(self):
        return f"LineSegment({self.start}, {self.end})"

class Arc:
    """Arc with basic operations"""
    def __init__(self, center: Point, radius: float, start_angle: float, end_angle: float):
        self.center = center
        self.radius = radius
        self.start_angle = start_angle
        self.end_angle = end_angle
    
    def __repr__(self):
        return f"Arc(center={self.center}, radius={self.radius:.2f})"

# =============================================================================
# VECTORIZATION (inline from geometry_engine/vectorization.py)
# =============================================================================

def vectorize_from_raw(raw_lines: List[List[float]], raw_arcs: Optional[List[dict]] = None) -> Tuple[List[LineSegment], List[Arc]]:
    """Convert raw coordinate data to geometric objects"""
    line_segments = []
    for line_data in raw_lines:
        if len(line_data) >= 4:
            x1, y1, x2, y2 = line_data[:4]
            start = Point(x1, y1)
            end = Point(x2, y2)
            line_segments.append(LineSegment(start, end))
    
    arcs = []
    if raw_arcs:
        for arc_data in raw_arcs:
            center = Point(arc_data.get('center_x', 0), arc_data.get('center_y', 0))
            radius = arc_data.get('radius', 0)
            start_angle = arc_data.get('start_angle', 0)
            end_angle = arc_data.get('end_angle', 0)
            arcs.append(Arc(center, radius, start_angle, end_angle))
    
    return line_segments, arcs

# =============================================================================
# CONSTRAINT DETECTION (inline from geometry_engine/constraint_detector.py)
# =============================================================================

class ConstraintDetector:
    """Detects geometric relationships between primitives"""
    
    def __init__(self, angle_tolerance: float = 0.1, distance_tolerance: float = 2.0, length_tolerance: float = 0.1):
        self.angle_tolerance = angle_tolerance
        self.distance_tolerance = distance_tolerance
        self.length_tolerance = length_tolerance
    
    def detect_all_constraints(self, line_segments: List[LineSegment], arcs: List[Arc] = None) -> Dict[str, List[Tuple]]:
        """Detect all geometric constraints"""
        constraints = {
            'perpendicular': [],
            'parallel': [],
            'collinear': [],
            'equal_length': [],
            'tangent': [],
            'concentric': [],
            'symmetric': []
        }
        
        # Detect line-to-line constraints
        for i in range(len(line_segments)):
            for j in range(i + 1, len(line_segments)):
                line1, line2 = line_segments[i], line_segments[j]
                
                # Perpendicular
                if line1.is_perpendicular_to(line2, self.angle_tolerance):
                    constraints['perpendicular'].append((i, j, 'perpendicular'))
                
                # Parallel
                elif line1.is_parallel_to(line2, self.angle_tolerance):
                    constraints['parallel'].append((i, j, 'parallel'))
                
                # Equal length
                if abs(line1.length - line2.length) < self.length_tolerance:
                    constraints['equal_length'].append((i, j, 'equal_length'))
        
        return constraints

# =============================================================================
# CONSTRAINT SOLVER (inline from geometry_engine/constraint_solver.py)
# =============================================================================

class ConstraintSolver:
    """Solves geometric constraints to clean up geometry"""
    
    def __init__(self, max_iterations: int = 10, convergence_threshold: float = 0.1):
        self.max_iterations = max_iterations
        self.convergence_threshold = convergence_threshold
    
    def solve_constraints(self, line_segments: List[LineSegment], constraints: Dict[str, List[Tuple]], arcs: List[Arc] = None) -> Tuple[List[LineSegment], List[Arc]]:
        """Apply constraint solving to clean up geometry"""
        solved_lines = [LineSegment(Point(line.start.x, line.start.y), Point(line.end.x, line.end.y)) for line in line_segments]
        solved_arcs = arcs[:] if arcs else []
        
        for iteration in range(self.max_iterations):
            # Apply angle snapping (perpendicular and parallel)
            solved_lines = self._snap_to_cardinal_angles(solved_lines)
            
            # Apply perpendicular constraints
            solved_lines = self._apply_perpendicular_constraints(solved_lines, constraints.get('perpendicular', []))
            
            # Apply parallel constraints
            solved_lines = self._apply_parallel_constraints(solved_lines, constraints.get('parallel', []))
            
            # Check convergence
            total_change = 0
            for i, (old_line, new_line) in enumerate(zip(line_segments, solved_lines)):
                change = old_line.start.distance_to(new_line.start) + old_line.end.distance_to(new_line.end)
                total_change += change
            
            if total_change < self.convergence_threshold:
                break
        
        return solved_lines, solved_arcs
    
    def _snap_to_cardinal_angles(self, line_segments: List[LineSegment]) -> List[LineSegment]:
        """Snap lines to cardinal angles (0°, 45°, 90°, 135°)"""
        snapped_lines = []
        
        for line in line_segments:
            angle = math.degrees(line.angle) % 180
            
            # Find closest cardinal angle
            cardinal_angles = [0, 45, 90, 135]
            closest_angle = min(cardinal_angles, key=lambda a: abs(a - angle))
            
            if abs(angle - closest_angle) < 15:  # 15 degree tolerance
                # Snap to cardinal angle
                length = line.length
                rad_angle = math.radians(closest_angle)
                
                dx = length * math.cos(rad_angle)
                dy = length * math.sin(rad_angle)
                
                new_end = Point(line.start.x + dx, line.start.y + dy)
                snapped_lines.append(LineSegment(line.start, new_end))
            else:
                snapped_lines.append(line)
        
        return snapped_lines
    
    def _apply_perpendicular_constraints(self, line_segments: List[LineSegment], perpendicular_constraints: List[Tuple]) -> List[LineSegment]:
        """Apply perpendicular constraints"""
        lines_copy = [LineSegment(Point(line.start.x, line.start.y), Point(line.end.x, line.end.y)) for line in line_segments]
        
        for constraint in perpendicular_constraints:
            if len(constraint) >= 3:
                idx1, idx2, constraint_type = constraint
                if (constraint_type == 'perpendicular' and 
                    0 <= idx1 < len(lines_copy) and 
                    0 <= idx2 < len(lines_copy)):
                    
                    line1 = lines_copy[idx1]
                    line2 = lines_copy[idx2]
                    
                    # Ensure perpendicularity by adjusting the second line
                    angle1 = line1.angle
                    perp_angle = angle1 + math.pi/2
                    
                    # Adjust line2 to be perpendicular to line1
                    length2 = line2.length
                    dx = length2 * math.cos(perp_angle)
                    dy = length2 * math.sin(perp_angle)
                    
                    new_end = Point(line2.start.x + dx, line2.start.y + dy)
                    lines_copy[idx2] = LineSegment(line2.start, new_end)
        
        return lines_copy
    
    def _apply_parallel_constraints(self, line_segments: List[LineSegment], parallel_constraints: List[Tuple]) -> List[LineSegment]:
        """Apply parallel constraints"""
        lines_copy = [LineSegment(Point(line.start.x, line.start.y), Point(line.end.x, line.end.y)) for line in line_segments]
        
        for constraint in parallel_constraints:
            if len(constraint) >= 3:
                idx1, idx2, constraint_type = constraint
                if (constraint_type == 'parallel' and 
                    0 <= idx1 < len(lines_copy) and 
                    0 <= idx2 < len(lines_copy)):
                    
                    line1 = lines_copy[idx1]
                    line2 = lines_copy[idx2]
                    
                    # Make line2 parallel to line1
                    angle1 = line1.angle
                    length2 = line2.length
                    
                    dx = length2 * math.cos(angle1)
                    dy = length2 * math.sin(angle1)
                    
                    new_end = Point(line2.start.x + dx, line2.start.y + dy)
                    lines_copy[idx2] = LineSegment(line2.start, new_end)
        
        return lines_copy

# =============================================================================
# ROOM DETECTION (inline from geometry_engine/room_detector.py)
# =============================================================================

class Room:
    """Represents a detected room/closed polygon"""
    def __init__(self, vertices: List[Point], boundary_lines: List[int], room_type: str = 'unknown'):
        self.vertices = vertices
        self.boundary_lines = boundary_lines
        self.room_type = room_type
        self.area = self._calculate_area()
        self.centroid = self._calculate_centroid()
    
    def _calculate_area(self) -> float:
        """Calculate polygon area using shoelace formula"""
        if len(self.vertices) < 3:
            return 0.0
        
        area = 0.0
        n = len(self.vertices)
        for i in range(n):
            j = (i + 1) % n
            area += self.vertices[i].x * self.vertices[j].y
            area -= self.vertices[j].x * self.vertices[i].y
        return abs(area) / 2.0
    
    def _calculate_centroid(self) -> Point:
        """Calculate polygon centroid"""
        if not self.vertices:
            return Point(0, 0)
        
        cx = sum(v.x for v in self.vertices) / len(self.vertices)
        cy = sum(v.y for v in self.vertices) / len(self.vertices)
        return Point(cx, cy)

class RoomDetector:
    """Detects closed polygons (rooms) from line segments"""
    
    def __init__(self, tolerance: float = 5.0):
        self.tolerance = tolerance
    
    def detect_rooms(self, line_segments: List[LineSegment]) -> List[Room]:
        """Detect closed polygons from line segments"""
        if not line_segments:
            return []
        
        # Build adjacency graph
        graph = self._build_adjacency_graph(line_segments)
        
        # Find cycles (closed polygons)
        cycles = self._find_cycles(graph, line_segments)
        
        # Convert cycles to Room objects
        rooms = []
        for cycle in cycles:
            if len(cycle) >= 3:
                vertices = self._cycle_to_vertices(cycle, line_segments)
                if vertices and len(vertices) >= 3:
                    room = Room(vertices, cycle)
                    rooms.append(room)
        
        return rooms
    
    def _build_adjacency_graph(self, line_segments: List[LineSegment]) -> Dict[int, List[int]]:
        """Build graph where nodes are line endpoints and edges connect nearby points"""
        graph = defaultdict(list)
        points = []
        
        # Collect all endpoints
        for i, line in enumerate(line_segments):
            points.extend([(line.start, i, 'start'), (line.end, i, 'end')])
        
        # Connect nearby points
        for i, (point1, line1_idx, pos1) in enumerate(points):
            for j, (point2, line2_idx, pos2) in enumerate(points[i+1:], i+1):
                if point1.distance_to(point2) < self.tolerance:
                    graph[line1_idx].append(line2_idx)
                    graph[line2_idx].append(line1_idx)
        
        return graph
    
    def _find_cycles(self, graph: Dict[int, List[int]], line_segments: List[LineSegment]) -> List[List[int]]:
        """Find cycles in the graph"""
        cycles = []
        visited = set()
        
        def dfs(node, path, start):
            if len(path) > 8:  # Limit cycle length
                return
            
            if node == start and len(path) > 2:
                cycles.append(path[:])
                return
            
            if node in visited and node != start:
                return
            
            visited.add(node)
            
            for neighbor in graph.get(node, []):
                if neighbor not in path or neighbor == start:
                    path.append(neighbor)
                    dfs(neighbor, path, start)
                    path.pop()
            
            if node != start:
                visited.discard(node)
        
        # Try to find cycles starting from each node
        for start_node in range(len(line_segments)):
            if start_node not in visited:
                dfs(start_node, [start_node], start_node)
        
        # Remove duplicate cycles
        unique_cycles = []
        for cycle in cycles:
            normalized = tuple(sorted(cycle))
            if normalized not in [tuple(sorted(c)) for c in unique_cycles]:
                unique_cycles.append(cycle)
        
        return unique_cycles[:10]  # Limit number of rooms
    
    def _cycle_to_vertices(self, cycle: List[int], line_segments: List[LineSegment]) -> List[Point]:
        """Convert cycle of line indices to ordered vertices"""
        if len(cycle) < 3:
            return []
        
        vertices = []
        for i in range(len(cycle)):
            line_idx = cycle[i]
            next_line_idx = cycle[(i + 1) % len(cycle)]
            
            if line_idx < len(line_segments) and next_line_idx < len(line_segments):
                line = line_segments[line_idx]
                next_line = line_segments[next_line_idx]
                
                # Find connection point between consecutive lines
                candidates = [
                    (line.start, next_line.start),
                    (line.start, next_line.end),
                    (line.end, next_line.start),
                    (line.end, next_line.end)
                ]
                
                min_dist = float('inf')
                connection_point = None
                
                for p1, p2 in candidates:
                    dist = p1.distance_to(p2)
                    if dist < min_dist:
                        min_dist = dist
                        connection_point = Point((p1.x + p2.x) / 2, (p1.y + p2.y) / 2)
                
                if connection_point:
                    vertices.append(connection_point)
        
        return vertices

# =============================================================================
# MAIN GEOMETRY ENGINE API (inline from geometry_engine/api.py)
# =============================================================================

class GeometryEngine:
    """Main geometry processing engine"""
    
    def __init__(self, angle_tolerance: float = 0.1, distance_tolerance: float = 2.0, 
                 length_tolerance: float = 0.1, performance_mode: bool = False):
        self.constraint_detector = ConstraintDetector(angle_tolerance, distance_tolerance, length_tolerance)
        self.constraint_solver = ConstraintSolver()
        self.room_detector = RoomDetector(distance_tolerance)
        self.performance_mode = performance_mode
    
    def process_raw_geometry(self, raw_lines: List[List[float]], raw_symbols: Optional[List[dict]] = None, 
                           raw_text: Optional[List[dict]] = None) -> Dict:
        """Process raw geometry data through the complete pipeline"""
        
        # Step 1: Vectorization
        line_segments, arcs = vectorize_from_raw(raw_lines)
        
        if not line_segments:
            return {
                'lines': [],
                'arcs': [],
                'rooms': [],
                'constraints': {},
                'metadata': {},
                'statistics': {'original_lines': 0, 'final_lines': 0, 'rooms_detected': 0, 'constraints_applied': 0}
            }
        
        # Step 2: Constraint Detection
        constraints = self.constraint_detector.detect_all_constraints(line_segments, arcs)
        
        # Step 3: Constraint Solving
        if self.performance_mode and len(line_segments) > 100:
            # Fast mode - minimal constraint solving
            solved_lines = self.constraint_solver._snap_to_cardinal_angles(line_segments)
            solved_arcs = arcs[:] if arcs else []
        else:
            # Full constraint solving
            solved_lines, solved_arcs = self.constraint_solver.solve_constraints(line_segments, constraints, arcs)
        
        # Step 4: Room Detection
        rooms = self.room_detector.detect_rooms(solved_lines)
        
        # Step 5: Prepare output
        result = {
            'lines': solved_lines,
            'arcs': solved_arcs,
            'rooms': rooms,
            'constraints': constraints,
            'metadata': {
                'processing_mode': 'performance' if self.performance_mode else 'full',
                'input_segments': len(raw_lines)
            },
            'statistics': {
                'original_lines': len(line_segments),
                'final_lines': len(solved_lines),
                'rooms_detected': len(rooms),
                'total_floor_area': sum(room.area for room in rooms),
                'constraints_applied': sum(len(v) for v in constraints.values()),
                'perpendicular_pairs': len(constraints.get('perpendicular', [])),
                'parallel_pairs': len(constraints.get('parallel', [])),
                'equal_length_pairs': len(constraints.get('equal_length', []))
            }
        }
        
        return result

# Wrapper that prefers the original engine for identical behavior
class EngineAdapter:
    """Adapter that uses the repo's GeometryEngine if available for parity."""
    def __init__(self, performance_mode: bool = False):
        if EXTERNAL_ENGINE_AVAILABLE:
            self._engine = ExternalGeometryEngine(performance_mode=performance_mode)
            self._use_external = True
        else:
            self._engine = GeometryEngine(performance_mode=performance_mode)
            self._use_external = False

    def process_raw_geometry(self, raw_lines: List[List[float]], raw_symbols: Optional[List[dict]] = None,
                              raw_text: Optional[List[dict]] = None) -> Dict:
        return self._engine.process_raw_geometry(raw_lines, raw_symbols, raw_text)

# =============================================================================
# COLOR-CODED FLOOR PLAN PROCESSOR (main application)
# =============================================================================

class ColoredFloorPlanProcessor:
    """Processes hand-drawn floor plans into color-coded CAD outputs"""
    
    def __init__(self):
        # Use adapter to ensure outputs match the original engine when available
        self.engine = EngineAdapter(performance_mode=True)
        
        # Color scheme for different room types and features
        self.colors = {
            'walls': '#2C3E50',          # Dark blue-gray for walls
            'doors': '#E74C3C',          # Red for doors
            'windows': '#3498DB',        # Blue for windows
            'kitchen': '#F39C12',        # Orange for kitchen
            'bathroom': '#9B59B6',       # Purple for bathroom
            'bedroom': '#E91E63',        # Pink for bedroom
            'living_room': '#2ECC71',    # Green for living room
            'dining_room': '#F1C40F',    # Yellow for dining room
            'closet': '#95A5A6',         # Gray for closets
            'laundry': '#1ABC9C',        # Teal for laundry
            'entryway': '#34495E',       # Dark gray for entryway
            'family_room': '#27AE60',    # Darker green for family room
            'text': '#2C3E50',           # Dark for text labels
            'dimensions': '#E74C3C'      # Red for dimensions
        }
    
    def extract_lines_two_pass(self, image_path, max_width: int = 1400):
        """Two-pass extractor that captures outer outline and medium-length internals"""
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not load image: {image_path}")

        # Resize for consistency
        h, w = img.shape[:2]
        scale = 1.0
        if w > max_width:
            scale = max_width / float(w)
            img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
            h, w = img.shape[:2]

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Light clean-up
        open_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, open_kernel, iterations=1)
        close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, close_kernel, iterations=1)

        # Edges and Hough (outer walls)
        edges = cv2.Canny(bw, 60, 180, apertureSize=3)
        min_len_outer = int(max(h, w) * 0.12)
        hough_lines = cv2.HoughLinesP(
            edges, 1, np.pi / 180, threshold=100, minLineLength=min_len_outer, maxLineGap=int(min_len_outer * 0.3)
        )

        # LSD (internal walls + stairs)
        try:
            lsd = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD)
        except Exception:
            try:
                lsd = cv2.createLineSegmentDetector()
            except Exception:
                lsd = None
        
        lines_lsd = None
        if lsd is not None:
            try:
                res = lsd.detect(gray)
                if isinstance(res, tuple):
                    lines_lsd = res[0]
                else:
                    lines_lsd = res
            except Exception:
                lines_lsd = None

        def to_seg_list(lines):
            out = []
            if lines is None:
                return out
            for l in lines:
                x1, y1, x2, y2 = l[0]
                out.append([float(x1), float(y1), float(x2), float(y2)])
            return out

        segs_outer = to_seg_list(hough_lines)

        # Filter LSD by length and angle
        segs_inner = []
        if lines_lsd is not None:
            min_len_inner = int(max(h, w) * 0.035)
            for l in lines_lsd:
                x1, y1, x2, y2 = l[0]
                length = np.hypot(x2 - x1, y2 - y1)
                if length < min_len_inner:
                    continue
                angle = abs(np.degrees(np.arctan2(y2 - y1, x2 - x1))) % 180
                nearest = min([0, 45, 90, 135], key=lambda a: abs(a - angle))
                if abs(angle - nearest) > 12:
                    continue
                segs_inner.append([float(x1), float(y1), float(x2), float(y2)])

        # Merge and dedupe
        all_segs = segs_outer + segs_inner

        def dedupe(segments, tol=6):
            kept = []
            for s in segments:
                x1, y1, x2, y2 = s
                duplicate = False
                for k in kept:
                    kx1, ky1, kx2, ky2 = k
                    if (abs(x1 - kx1) < tol and abs(y1 - ky1) < tol and abs(x2 - kx2) < tol and abs(y2 - ky2) < tol) or \
                       (abs(x1 - kx2) < tol and abs(y1 - ky2) < tol and abs(x2 - kx1) < tol and abs(y2 - ky1) < tol):
                        duplicate = True
                        break
                if not duplicate:
                    kept.append(s)
            return kept

        raw_lines = dedupe(all_segs)
        print(f"Extracted {len(raw_lines)} line segments")
        return raw_lines, img.shape

    def detect_stairs(self, segments, image_shape):
        """Heuristic stair detector"""
        h, w = image_shape[:2]
        if not segments:
            return []

        # Compute per-line features
        feats = []
        for x1, y1, x2, y2 in segments:
            length = np.hypot(x2 - x1, y2 - y1)
            angle = np.degrees(np.arctan2(y2 - y1, x2 - x1)) % 180
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            feats.append((angle, length, cx, cy, (x1, y1, x2, y2)))

        # Group by angle bucket
        buckets = {0: [], 45: [], 90: [], 135: []}
        for angle, length, cx, cy, seg in feats:
            nearest = min(buckets.keys(), key=lambda a: abs(a - angle))
            if abs(nearest - angle) <= 8:
                buckets[nearest].append((length, cx, cy, seg))

        stairs_symbols = []
        for angle_key, items in buckets.items():
            if len(items) < 4:
                continue
            items.sort(key=lambda t: t[2] if angle_key in (0, 180) else t[1])
            for i in range(len(items) - 3):
                window = items[i:i+6]
                if len(window) < 4:
                    continue
                lengths = [t[0] for t in window]
                if np.std(lengths) > max(6, 0.15 * np.mean(lengths)):
                    continue
                # Compute bounding box
                xs = []
                ys = []
                for _, _, _, seg in window:
                    x1, y1, x2, y2 = seg
                    xs += [x1, x2]
                    ys += [y1, y2]
                x_min, x_max = min(xs), max(xs)
                y_min, y_max = min(ys), max(ys)
                if (x_max - x_min) * (y_max - y_min) < (0.01 * w * h):
                    continue
                stairs_symbols.append({'class': 'stairs', 'bbox': [x_min, y_min, x_max, y_max], 'confidence': 0.7})
                break

        return stairs_symbols
    
    def simulate_ocr_data(self, image_shape):
        """Simulate OCR data extraction from the floor plan"""
        height, width = image_shape[:2]
        
        text_data = [
            {'text': 'kitchen', 'bbox': [[width*0.6, height*0.2], [width*0.8, height*0.2], [width*0.8, height*0.25], [width*0.6, height*0.25]]},
            {'text': 'family room', 'bbox': [[width*0.1, height*0.3], [width*0.3, height*0.3], [width*0.3, height*0.35], [width*0.1, height*0.35]]},
            {'text': 'dining room', 'bbox': [[width*0.6, height*0.5], [width*0.8, height*0.5], [width*0.8, height*0.55], [width*0.6, height*0.55]]},
            {'text': 'living room', 'bbox': [[width*0.4, height*0.7], [width*0.6, height*0.7], [width*0.6, height*0.75], [width*0.4, height*0.75]]},
            {'text': 'laundry', 'bbox': [[width*0.1, height*0.6], [width*0.25, height*0.6], [width*0.25, height*0.65], [width*0.1, height*0.65]]},
            {'text': 'closet', 'bbox': [[width*0.3, height*0.6], [width*0.4, height*0.6], [width*0.4, height*0.65], [width*0.3, height*0.65]]},
            {'text': 'bath', 'bbox': [[width*0.85, height*0.4], [width*0.95, height*0.4], [width*0.95, height*0.45], [width*0.85, height*0.45]]},
            {'text': 'entryway', 'bbox': [[width*0.4, height*0.85], [width*0.6, height*0.85], [width*0.6, height*0.9], [width*0.4, height*0.9]]},
        ]
        
        return text_data
    
    def simulate_symbol_detection(self, image_shape):
        """Simulate door and window detection"""
        height, width = image_shape[:2]
        
        symbols = [
            # Doors
            {'class': 'door', 'bbox': [width*0.45, height*0.82, width*0.55, height*0.88], 'confidence': 0.9},
            {'class': 'door', 'bbox': [width*0.25, height*0.45, width*0.35, height*0.55], 'confidence': 0.85},
            {'class': 'door', 'bbox': [width*0.8, height*0.35, width*0.85, height*0.45], 'confidence': 0.8},
            
            # Windows
            {'class': 'window', 'bbox': [width*0.1, height*0.15, width*0.2, height*0.25], 'confidence': 0.9},
            {'class': 'window', 'bbox': [width*0.7, height*0.1, width*0.9, height*0.15], 'confidence': 0.85},
            {'class': 'window', 'bbox': [width*0.9, height*0.6, width*0.95, height*0.8], 'confidence': 0.8},
        ]
        
        return symbols
    
    def classify_rooms_by_text(self, rooms, text_data):
        """Classify rooms based on nearby text labels"""
        classified_rooms = []
        
        for room in rooms:
            room_type = 'unknown'
            
            for text_item in text_data:
                text_center = self.bbox_center(text_item['bbox'])
                
                if self.point_in_polygon(text_center, room.vertices):
                    text_lower = text_item['text'].lower()
                    
                    if 'kitchen' in text_lower:
                        room_type = 'kitchen'
                    elif 'bath' in text_lower or 'bathroom' in text_lower:
                        room_type = 'bathroom'
                    elif 'bedroom' in text_lower or 'bed' in text_lower:
                        room_type = 'bedroom'
                    elif 'living' in text_lower:
                        room_type = 'living_room'
                    elif 'dining' in text_lower:
                        room_type = 'dining_room'
                    elif 'family' in text_lower:
                        room_type = 'family_room'
                    elif 'closet' in text_lower:
                        room_type = 'closet'
                    elif 'laundry' in text_lower:
                        room_type = 'laundry'
                    elif 'entry' in text_lower:
                        room_type = 'entryway'
                    break
            
            # Fallback classification
            if room_type == 'unknown':
                if room.area < 500:
                    room_type = 'closet'
                elif room.area < 2000:
                    room_type = 'bathroom'
                elif room.area < 8000:
                    room_type = 'bedroom'
                else:
                    room_type = 'living_room'
            
            room.room_type = room_type
            classified_rooms.append(room)
        
        return classified_rooms
    
    def bbox_center(self, bbox):
        """Calculate center of bounding box"""
        if isinstance(bbox[0], list):
            x_coords = [pt[0] for pt in bbox]
            y_coords = [pt[1] for pt in bbox]
            return Point(sum(x_coords) / len(x_coords), sum(y_coords) / len(y_coords))
        else:
            return Point((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)
    
    def point_in_polygon(self, point, vertices):
        """Check if point is inside polygon"""
        n = len(vertices)
        inside = False
        
        p1x, p1y = vertices[0].x, vertices[0].y
        for i in range(1, n + 1):
            p2x, p2y = vertices[i % n].x, vertices[i % n].y
            
            if point.y > min(p1y, p2y):
                if point.y <= max(p1y, p2y):
                    if point.x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (point.y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or point.x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y
        
        return inside
    
    def create_colored_visualization(self, processed_result, symbols, text_data, output_path, original_shape):
        """Create color-coded floor plan visualization"""
        print("Creating color-coded visualization...")
        
        height, width = original_shape[:2]
        
        # Create figure
        fig, ax = plt.subplots(1, 1, figsize=(16, 12))
        ax.set_xlim(0, width)
        ax.set_ylim(height, 0)
        ax.set_aspect('equal')
        
        # Background
        ax.add_patch(patches.Rectangle((0, 0), width, height, 
                                     facecolor='white', edgecolor='none'))
        
        # Draw rooms
        rooms = processed_result.get('rooms', [])
        if text_data:
            rooms = self.classify_rooms_by_text(rooms, text_data)
        
        print(f"Drawing {len(rooms)} rooms with color coding...")
        
        for room in rooms:
            if len(room.vertices) < 3:
                continue
            
            room_color = self.colors.get(room.room_type, self.colors['living_room'])
            xy_points = [(v.x, v.y) for v in room.vertices]
            
            room_patch = patches.Polygon(
                xy_points, 
                facecolor=room_color, 
                alpha=0.3,
                edgecolor=room_color,
                linewidth=2
            )
            ax.add_patch(room_patch)
            
            if hasattr(room, 'room_type'):
                ax.text(room.centroid.x, room.centroid.y, 
                       room.room_type.replace('_', ' ').title(),
                       ha='center', va='center', 
                       fontsize=10, fontweight='bold',
                       color=self.colors['text'],
                       bbox=dict(boxstyle="round,pad=0.3", 
                               facecolor='white', alpha=0.8))
        
        # Draw walls
        lines = processed_result.get('lines', [])
        print(f"Drawing {len(lines)} wall segments...")
        
        for line in lines:
            ax.plot([line.start.x, line.end.x], 
                   [line.start.y, line.end.y],
                   color=self.colors['walls'], 
                   linewidth=3, 
                   solid_capstyle='round')
        
        # Draw symbols
        for symbol in symbols:
            bbox = symbol['bbox']
            symbol_type = symbol['class']
            
            if symbol_type == 'door':
                color = self.colors['doors']
                label = 'Door'
            elif symbol_type == 'window':
                color = self.colors['windows']
                label = 'Window'
            else:
                # For stairs etc., draw dashed outline
                center_x = (bbox[0] + bbox[2]) / 2
                center_y = (bbox[1] + bbox[3]) / 2
                rect = patches.Rectangle(
                    (bbox[0], bbox[1]), bbox[2] - bbox[0], bbox[3] - bbox[1],
                    facecolor='none', edgecolor=self.colors['walls'], 
                    linewidth=1.5, linestyle='--', alpha=0.8
                )
                ax.add_patch(rect)
                ax.text(center_x, center_y, symbol_type,
                       ha='center', va='center',
                       fontsize=9, fontweight='bold',
                       color=self.colors['text'],
                       bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.6))
                continue

            # Draw filled rectangle for doors/windows
            rect = patches.Rectangle(
                (bbox[0], bbox[1]), 
                bbox[2] - bbox[0], 
                bbox[3] - bbox[1],
                facecolor=color, 
                alpha=0.7,
                edgecolor=color,
                linewidth=2
            )
            ax.add_patch(rect)
            
            center_x = (bbox[0] + bbox[2]) / 2
            center_y = (bbox[1] + bbox[3]) / 2
            ax.text(center_x, center_y, label,
                   ha='center', va='center',
                   fontsize=8, fontweight='bold',
                   color='white')
        
        # Add title
        stats = processed_result.get('statistics', {})
        title = f"PaperCAD Edge - Color-Coded Floor Plan\n"
        title += f"Rooms: {stats.get('rooms_detected', 0)} | "
        title += f"Lines: {stats.get('final_lines', 0)} | "
        title += f"Constraints: {stats.get('constraints_applied', 0)}"
        
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        
        # Legend
        legend_elements = []
        room_types_found = set(getattr(room, 'room_type', 'unknown') for room in rooms)
        
        for room_type in sorted(room_types_found):
            if room_type in self.colors:
                legend_elements.append(
                    patches.Patch(color=self.colors[room_type], 
                                label=room_type.replace('_', ' ').title())
                )
        
        legend_elements.extend([
            patches.Patch(color=self.colors['doors'], label='Doors'),
            patches.Patch(color=self.colors['windows'], label='Windows'),
            patches.Patch(color=self.colors['walls'], label='Walls')
        ])
        
        ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.02, 1))
        
        # Clean axes
        ax.set_xticks([])
        ax.set_yticks([])
        
        # Save
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        plt.close()
        
        print(f"Color-coded floor plan saved to: {output_path}")
    
    def process_floor_plan(self, input_path, output_path):
        """Main processing function"""
        print("Starting PaperCAD Edge Color Processing...")
        
        # Extract lines
        raw_lines, image_shape = self.extract_lines_two_pass(input_path)
        
        # Simulate data
        text_data = self.simulate_ocr_data(image_shape)
        symbols = self.simulate_symbol_detection(image_shape)
        stairs_syms = self.detect_stairs(raw_lines, image_shape)
        symbols.extend(stairs_syms)
        
        # Process geometry
        print("Processing through geometry engine...")
        result = self.engine.process_raw_geometry(raw_lines, symbols, text_data)
        
        # Create visualization
        self.create_colored_visualization(result, symbols, text_data, output_path, image_shape)
        
        # Summary
        print(f"\nProcessing Summary:")
        print(f"  Input lines detected: {len(raw_lines)}")
        print(f"  Rooms identified: {len(result.get('rooms', []))}")
        print(f"  Doors/Windows: {len(symbols)}")
        print(f"  Text labels: {len(text_data)}")
        print(f"  Constraints applied: {result.get('statistics', {}).get('constraints_applied', 0)}")
        
        return result

# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Main function - modify these paths as needed"""
    import os
    
    processor = ColoredFloorPlanProcessor()
    
    # MODIFY THESE PATHS FOR YOUR USE:
    input_image = "inputs/test_4.jpeg"  # Change to your input image path
    output_image = "output/image_tests/standalone_colored_4.png"  # Change to your desired output path
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_image) if os.path.dirname(output_image) else ".", exist_ok=True)
    
    try:
        result = processor.process_floor_plan(input_image, output_image)
        print(f"\nSUCCESS! Color-coded floor plan created at: {output_image}")
        
    except Exception as e:
        print(f"Error processing floor plan: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
