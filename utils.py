"""Utility functions for Photo2Head Depth addon."""

import bpy
import json
import os
import math
import numpy as np
from bpy.props import StringProperty, FloatProperty, IntProperty, BoolProperty

def register_properties():
    """Register scene properties."""
    Scene = bpy.types.Scene
    
    Scene.p2h_front_image = StringProperty(
        name="Front Image",
        description="Front view reference photo",
        subtype='FILE_PATH',
        default=""
    )
    
    Scene.p2h_side_image = StringProperty(
        name="Side Image",
        description="Side view reference photo",
        subtype='FILE_PATH',
        default=""
    )
    
    Scene.p2h_head_height = FloatProperty(
        name="Head Height (m)",
        description="Total height of head from chin to top of skull",
        default=0.23,
        min=0.15,
        max=0.35,
        step=0.01,
        unit='LENGTH'
    )
    
    Scene.p2h_neck_length = FloatProperty(
        name="Neck Length (m)",
        description="Maximum neck length below chin",
        default=0.06,
        min=0.0,
        max=0.2,
        step=0.01,
        unit='LENGTH'
    )
    
    Scene.p2h_depth_scale = FloatProperty(
        name="Depth Scale",
        description="Multiplier for depth estimation (1.0 = natural)",
        default=1.0,
        min=0.5,
        max=2.0,
        step=0.1
    )
    
    Scene.p2h_eye_depth = FloatProperty(
        name="Eye Socket Depth (m)",
        description="Depth of eye sockets",
        default=0.008,
        min=0.003,
        max=0.015,
        step=0.001,
        unit='LENGTH'
    )
    
    Scene.p2h_lip_sharpness = FloatProperty(
        name="Lip Sharpness",
        description="Crease sharpness for lip boundaries (0=smooth, 1=sharp)",
        default=0.8,
        min=0.0,
        max=1.0,
        step=0.1
    )
    
    Scene.p2h_nose_detail = FloatProperty(
        name="Nose Detail",
        description="Nostril hole depth and bridge definition (0=smooth, 1=sharp)",
        default=0.7,
        min=0.0,
        max=1.0,
        step=0.1
    )
    
    Scene.p2h_rings = IntProperty(
        name="Mesh Rings",
        description="Horizontal rings in mesh",
        default=250,
        min=100,
        max=500
    )
    
    Scene.p2h_segments = IntProperty(
        name="Segments (half)",
        description="Segments per ring (half-face)",
        default=120,
        min=40,
        max=200
    )
    
    Scene.p2h_depth_model = StringProperty(
        name="Depth Model",
        description="Which depth estimation model to use",
        default="midas"
    )
    
    Scene.p2h_auto_detect = BoolProperty(
        name="Auto Detect",
        description="Automatically detect landmarks and depth",
        default=True
    )

def unregister_properties():
    """Unregister scene properties."""
    Scene = bpy.types.Scene
    props = [
        'p2h_front_image', 'p2h_side_image', 'p2h_head_height', 'p2h_neck_length',
        'p2h_depth_scale', 'p2h_eye_depth', 'p2h_lip_sharpness', 'p2h_nose_detail',
        'p2h_rings', 'p2h_segments', 'p2h_depth_model', 'p2h_auto_detect'
    ]
    for prop in props:
        if hasattr(Scene, prop):
            delattr(Scene, prop)

def load_image(path, max_dim=1280):
    """Load image and return RGB numpy array."""
    if not path or not isinstance(path, str):
        return None
    try:
        import cv2
        abs_path = bpy.path.abspath(path)
        if not os.path.isfile(abs_path):
            return None
        bgr = cv2.imdecode(np.fromfile(abs_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        if bgr is None:
            return None
        h, w = bgr.shape[:2]
        k = max_dim / float(max(h, w))
        if k < 1.0:
            bgr = cv2.resize(bgr, (int(round(w * k)), int(round(h * k))), interpolation=cv2.INTER_AREA)
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    except Exception as e:
        print(f"Photo2Head: Failed to load image: {e}")
        return None

def get_image_collection(context):
    """Get or create the image collection."""
    coll_name = "Photo2Head_Images"
    c = bpy.data.collections.get(coll_name)
    if c is None:
        c = bpy.data.collections.new(coll_name)
        context.scene.collection.children.link(c)
    return c

def get_mesh_collection(context):
    """Get or create the mesh collection."""
    coll_name = "Photo2Head_Mesh"
    c = bpy.data.collections.get(coll_name)
    if c is None:
        c = bpy.data.collections.new(coll_name)
        context.scene.collection.children.link(c)
    return c

def remove_object(name):
    """Remove object and its data."""
    ob = bpy.data.objects.get(name)
    if ob is None:
        return
    data = ob.data
    bpy.data.objects.remove(ob, do_unlink=True)
    if data is not None and getattr(data, "users", 1) == 0:
        try:
            if isinstance(data, bpy.types.Mesh):
                bpy.data.meshes.remove(data)
        except Exception:
            pass
