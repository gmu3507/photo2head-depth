"""Facial landmark detection using MediaPipe."""

import numpy as np
import os

_LANDMARKER = None
_MP = None

def ensure_landmark_deps():
    """Check if MediaPipe is available."""
    try:
        import mediapipe
        return True
    except ImportError:
        return False

def detect_landmarks(rgb, use_tasks_api=False):
    """Detect 468 face landmarks from RGB image.
    
    Args:
        rgb: RGB image (H, W, 3) as uint8 numpy array
        use_tasks_api: Use new Tasks API instead of solutions
    
    Returns:
        Landmark array (468, 3) with (x, y, z) in normalized coords [0, 1]
    """
    try:
        import mediapipe as mp
        global _LANDMARKER, _MP
        
        if use_tasks_api:
            if _LANDMARKER is None:
                from mediapipe.tasks.python import BaseOptions
                from mediapipe.tasks.python import vision
                opts = vision.FaceLandmarkerOptions(
                    base_options=BaseOptions(model_asset_path="face_landmarker.task"),
                    running_mode=vision.RunningMode.IMAGE,
                    num_faces=1
                )
                _LANDMARKER = vision.FaceLandmarker.create_from_options(opts)
            
            img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            res = _LANDMARKER.detect(img)
            lm = res.face_landmarks[0] if res.face_landmarks else None
        else:
            fm = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=True,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5
            )
            try:
                res = fm.process(rgb)
            finally:
                fm.close()
            lm = res.multi_face_landmarks[0].landmark if res.multi_face_landmarks else None
        
        if lm is None:
            return None
        
        h, w = rgb.shape[:2]
        return np.array([[p.x, p.y, p.z] for p in lm], dtype=np.float32)
    
    except Exception as e:
        print(f"Photo2Head: Landmark detection failed: {e}")
        return None

# MediaPipe face mesh index groups
FACE_OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378,
             400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21,
             54, 103, 67, 109]
NOSE_INDICES = [1, 2, 4, 5, 6, 248, 275, 305, 362, 454]
NOSTRIL_INDICES = [195, 197, 174, 201, 419, 420, 421, 399, 456, 457, 458, 459]
LIPS_UPPER = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 375, 321, 405]
LIPS_LOWER = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409]
EYE_L = [33, 246, 161, 160, 159, 158, 157, 173, 133, 155, 154, 153, 145, 144, 163, 7]
EYE_R = [263, 466, 388, 387, 386, 385, 384, 398, 362, 382, 381, 380, 374, 373, 390, 249]
BROW_L = [70, 63, 105, 66, 107]
BROW_R = [300, 293, 334, 296, 336]

def extract_front_features(lm, rgb_shape):
    """Extract front-facing feature points from landmarks.
    
    Returns dict with feature groups and key anatomical heights.
    """
    h, w = rgb_shape[:2]
    
    return {
        "face_oval": lm[FACE_OVAL],
        "nose": lm[NOSE_INDICES],
        "nostrils": lm[NOSTRIL_INDICES],
        "lips_upper": lm[LIPS_UPPER],
        "lips_lower": lm[LIPS_LOWER],
        "eye_l": lm[EYE_L],
        "eye_r": lm[EYE_R],
        "brow_l": lm[BROW_L],
        "brow_r": lm[BROW_R],
        "nasion": lm[168],  # bridge of nose (top)
        "nose_tip": lm[4],   # tip of nose
        "menton": lm[152],   # chin/jaw point
        "lip_top": lm[0],    # top of upper lip
        "lip_bottom": lm[17],  # bottom of lower lip
    }

def extract_side_features(lm, rgb_shape, profile="LEFT"):
    """Extract side profile features. profile='LEFT' or 'RIGHT'."""
    h, w = rgb_shape[:2]
    
    if profile == "LEFT":
        indices = {
            "profile_outline": list(range(10, 11)) + list(range(338, 350)) + list(range(450, 468)),
            "nose_profile": [1, 2, 4, 5],
            "nostril_profile": [419, 420, 421, 399],
            "lip_profile": [61, 185, 40, 39, 37, 0],
            "chin_profile": [152, 148, 176, 149, 150],
        }
    else:
        indices = {
            "profile_outline": list(range(268, 279)) + list(range(168, 180)),
            "nose_profile": [1, 2, 4, 5],
            "nostril_profile": [456, 457, 458, 459],
            "lip_profile": [291, 405, 270, 269, 267, 0],
            "chin_profile": [152, 148, 176, 149, 150],
        }
    
    return {k: lm[np.array(idx)] for k, idx in indices.items()}
