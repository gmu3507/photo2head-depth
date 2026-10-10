"""3D head mesh generation from depth and landmarks."""

import numpy as np
from scipy.interpolate import PchipInterpolator

def smooth_curve(values, sigma=1.0):
    """Smooth a 1D curve using Gaussian filter."""
    try:
        from scipy import ndimage
        return ndimage.gaussian_filter1d(values, sigma, mode="nearest")
    except Exception:
        return values

def extract_contour_from_depth(depth, mask, axis=0):
    """Extract front/back or left/right contour from depth and mask."""
    h, w = depth.shape
    contour = []
    
    if axis == 0:  # horizontal rows
        for y in range(h):
            row_mask = mask[y]
            if row_mask.any():
                x_min, x_max = np.where(row_mask)[0][[0, -1]]
                contour.append((float(depth[y, x_min]), float(depth[y, x_max])))
            else:
                contour.append((0.0, 0.0))
    else:  # vertical columns
        for x in range(w):
            col_mask = mask[:, x]
            if col_mask.any():
                y_min, y_max = np.where(col_mask)[0][[0, -1]]
                contour.append((float(depth[y_min, x]), float(depth[y_max, x])))
            else:
                contour.append((0.0, 0.0))
    
    return np.array(contour, dtype=np.float32)

def build_eye_socket(eye_points, depth_map, eye_depth=0.008):
    """Build eye socket geometry from eye landmark points and depth.
    
    Returns vertices and face indices for one eye socket.
    """
    if len(eye_points) < 4:
        return np.zeros((0, 3)), []
    
    eye_center = eye_points.mean(axis=0)
    center_z = float(np.interp(eye_center[1], np.arange(depth_map.shape[0]), 
                               depth_map.mean(axis=1)))
    
    # Create socket rim
    rim_points = []
    n_rim = 16
    for i in range(n_rim):
        angle = 2 * np.pi * i / n_rim
        r = 0.015  # ~1.5 cm radius
        x = eye_center[0] + r * np.cos(angle)
        y = eye_center[1] + r * np.sin(angle)
        z = center_z - eye_depth * (0.5 + 0.5 * np.cos(angle))
        rim_points.append([x, y, z])
    
    # Center point (back of socket)
    center_point = [eye_center[0], eye_center[1], center_z - eye_depth]
    
    verts = np.array(rim_points + [center_point], dtype=np.float32)
    faces = [(i, (i+1) % n_rim, n_rim) for i in range(n_rim)]
    
    return verts, faces

def build_nostril_hole(nostril_points, depth_map, detail_strength=0.7):
    """Build nostril hole geometry."""
    if len(nostril_points) < 2:
        return np.zeros((0, 3)), []
    
    center = nostril_points.mean(axis=0)
    center_z = float(np.interp(center[1], np.arange(depth_map.shape[0]), 
                               depth_map.mean(axis=1)))
    
    # Create hole rim
    rim = []
    n_rim = 8
    r = 0.004  # ~4mm nostril radius
    hole_depth = 0.003 * detail_strength
    
    for i in range(n_rim):
        angle = 2 * np.pi * i / n_rim
        x = center[0] + r * np.cos(angle)
        y = center[1] + r * np.sin(angle)
        z = center_z - hole_depth
        rim.append([x, y, z])
    
    center_hole = [center[0], center[1], center_z - hole_depth * 1.5]
    
    verts = np.array(rim + [center_hole], dtype=np.float32)
    faces = [(i, (i+1) % n_rim, n_rim) for i in range(n_rim)]
    
    return verts, faces

def build_lip_crease(lip_upper, lip_lower, depth_map, sharpness=0.8):
    """Build sharp crease lines for lips."""
    if len(lip_upper) < 2 or len(lip_lower) < 2:
        return np.zeros((0, 3)), []
    
    verts = []
    faces = []
    
    # Crease between lips
    for i in range(min(len(lip_upper), len(lip_lower))):
        u = lip_upper[i]
        l = lip_lower[i]
        z_u = float(np.interp(u[1], np.arange(depth_map.shape[0]), depth_map.mean(axis=1)))
        z_l = float(np.interp(l[1], np.arange(depth_map.shape[0]), depth_map.mean(axis=1)))
        
        crease_depth = 0.002 * sharpness
        mid_z = (z_u + z_l) / 2.0 - crease_depth
        
        verts.append([u[0], u[1], z_u])
        verts.append([l[0], l[1], z_l])
        verts.append([u[0], u[1], mid_z])
        verts.append([l[0], l[1], mid_z])
    
    return np.array(verts, dtype=np.float32), faces

def build_head_mesh(depth_front, depth_side, lm_front, lm_side, head_height=0.23,
                     eye_depth=0.008, lip_sharpness=0.8, nose_detail=0.7,
                     rings=250, segments=120):
    """Build complete head mesh from front/side depth and landmarks.
    
    Returns:
        (verts, faces) - numpy arrays of vertices and face indices
    """
    try:
        from landmark_detector import extract_front_features, extract_side_features
    except ImportError:
        return None, None
    
    if depth_front is None or lm_front is None:
        return None, None
    
    # Extract features
    front_feat = extract_front_features(lm_front, depth_front.shape)
    side_feat = extract_side_features(lm_side, depth_side.shape) if lm_side is not None else {}
    
    # Build base head surface
    h_front, w_front = depth_front.shape
    verts = []
    faces = []
    vert_map = {}
    
    # Create main surface grid
    for y in range(0, h_front, max(1, h_front // rings)):
        for x in range(0, w_front, max(1, w_front // segments)):
            depth_val = float(depth_front[y, x])
            # Normalize to 3D space
            vx = (x - w_front / 2.0) / w_front * head_height
            vy = (y - h_front / 2.0) / h_front * head_height
            vz = depth_val * head_height
            verts.append([vx, vy, vz])
            vert_map[(y, x)] = len(verts) - 1
    
    # Build eye sockets
    eye_l_verts, eye_l_faces = build_eye_socket(front_feat["eye_l"], depth_front, eye_depth)
    eye_r_verts, eye_r_faces = build_eye_socket(front_feat["eye_r"], depth_front, eye_depth)
    
    eye_l_offset = len(verts)
    verts.extend(eye_l_verts.tolist())
    faces.extend([(f[0] + eye_l_offset, f[1] + eye_l_offset, f[2] + eye_l_offset) 
                  for f in eye_l_faces])
    
    eye_r_offset = len(verts)
    verts.extend(eye_r_verts.tolist())
    faces.extend([(f[0] + eye_r_offset, f[1] + eye_r_offset, f[2] + eye_r_offset) 
                  for f in eye_r_faces])
    
    # Build nostril holes
    nostril_verts, nostril_faces = build_nostril_hole(front_feat["nostrils"], depth_front, nose_detail)
    if len(nostril_verts) > 0:
        nostril_offset = len(verts)
        verts.extend(nostril_verts.tolist())
        faces.extend([(f[0] + nostril_offset, f[1] + nostril_offset, f[2] + nostril_offset) 
                      for f in nostril_faces])
    
    # Build lip creases
    lip_verts, lip_faces = build_lip_crease(front_feat["lips_upper"], front_feat["lips_lower"],
                                            depth_front, lip_sharpness)
    if len(lip_verts) > 0:
        lip_offset = len(verts)
        verts.extend(lip_verts.tolist())
        faces.extend([(f[0] + lip_offset, f[1] + lip_offset, f[2] + lip_offset) 
                      for f in lip_faces])
    
    # Create quad faces for main surface
    grid_y = list(range(0, h_front, max(1, h_front // rings)))
    grid_x = list(range(0, w_front, max(1, w_front // segments)))
    
    for i in range(len(grid_y) - 1):
        for j in range(len(grid_x) - 1):
            y1, y2 = grid_y[i], grid_y[i + 1]
            x1, x2 = grid_x[j], grid_x[j + 1]
            
            if (y1, x1) in vert_map and (y1, x2) in vert_map and \
               (y2, x1) in vert_map and (y2, x2) in vert_map:
                v00 = vert_map[(y1, x1)]
                v01 = vert_map[(y1, x2)]
                v10 = vert_map[(y2, x1)]
                v11 = vert_map[(y2, x2)]
                faces.append((v00, v01, v11, v10))
    
    return np.array(verts, dtype=np.float32), faces
