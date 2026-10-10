"""Monocular depth estimation using MiDaS or ZoeDepth."""

import numpy as np
import os
import sys

_DEPTH_MODEL = None
_MODEL_TYPE = "midas"  # or "zoedepth"

def ensure_depth_deps():
    """Install depth estimation dependencies if needed."""
    try:
        import torch
        import cv2
        return True
    except ImportError:
        pass
    return False

def load_depth_model(model_type="midas"):
    """Load MiDaS or ZoeDepth model."""
    global _DEPTH_MODEL, _MODEL_TYPE
    
    if _DEPTH_MODEL is not None and _MODEL_TYPE == model_type:
        return _DEPTH_MODEL
    
    try:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        if model_type == "midas":
            torch.hub.set_dir(os.path.expanduser("~/.torch/models"))
            model = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
            model.to(device)
            model.eval()
            transform = torch.hub.load("intel-isl/MiDaS", "transforms").small_transform
            _DEPTH_MODEL = {"model": model, "transform": transform, "device": device}
            _MODEL_TYPE = "midas"
        elif model_type == "zoedepth":
            import zoedepth
            model = zoedepth.ZoeDepth.build_from_config_file(torch.hub.get_dir() + "/zoedepth_config.yaml")
            model.to(device)
            model.eval()
            _DEPTH_MODEL = {"model": model, "device": device}
            _MODEL_TYPE = "zoedepth"
        
        return _DEPTH_MODEL
    except Exception as e:
        print(f"Photo2Head: Failed to load depth model: {e}")
        return None

def estimate_depth(rgb, model_type="midas", scale=1.0):
    """Estimate depth map from RGB image.
    
    Args:
        rgb: RGB image (H, W, 3) as uint8 numpy array
        model_type: "midas" or "zoedepth"
        scale: depth scale multiplier
    
    Returns:
        Normalized depth map (0-1) as float32, same shape as input
    """
    try:
        import torch
        import cv2
        
        model_dict = load_depth_model(model_type)
        if model_dict is None:
            return None
        
        h, w = rgb.shape[:2]
        
        if model_type == "midas":
            model = model_dict["model"]
            transform = model_dict["transform"]
            device = model_dict["device"]
            
            with torch.no_grad():
                input_batch = transform(rgb).to(device)
                prediction = model(input_batch)
                prediction = torch.nn.functional.interpolate(
                    prediction.unsqueeze(1), size=(h, w), mode="bicubic", align_corners=False
                ).squeeze()
            
            depth = prediction.cpu().numpy()
            depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
            depth = np.clip(depth * scale, 0, 1).astype(np.float32)
        
        elif model_type == "zoedepth":
            model = model_dict["model"]
            device = model_dict["device"]
            rgb_tensor = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).float().to(device) / 255.0
            
            with torch.no_grad():
                depth = model(rgb_tensor).squeeze().cpu().numpy()
            
            depth = cv2.resize(depth, (w, h), interpolation=cv2.INTER_LINEAR)
            depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
            depth = np.clip(depth * scale, 0, 1).astype(np.float32)
        
        return depth
    
    except Exception as e:
        print(f"Photo2Head: Depth estimation failed: {e}")
        return None

def estimate_depth_bilateral(depth, sigma_s=3, sigma_r=0.15):
    """Smooth depth map while preserving edges (bilateral filter)."""
    try:
        import cv2
        depth_8u = (depth * 255).astype(np.uint8)
        smoothed = cv2.bilateralFilter(depth_8u, 9, sigma_r * 255, sigma_s)
        return smoothed.astype(np.float32) / 255.0
    except Exception as e:
        print(f"Photo2Head: Depth smoothing failed: {e}")
        return depth
