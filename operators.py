"""Blender operators for Photo2Head Depth addon."""

import bpy
import os
import numpy as np
from bpy.props import EnumProperty
from . import utils, depth_estimator, landmark_detector, mesh_builder

class P2H_OT_setup_check(bpy.types.Operator):
    """Check if all dependencies are installed."""
    bl_idname = "p2h.setup_check"
    bl_label = "Check Dependencies"
    bl_description = "Verify that MediaPipe, PyTorch, and depth models are installed"
    bl_options = {'REGISTER'}

    def execute(self, context):
        missing = []
        
        try:
            import cv2
        except ImportError:
            missing.append("OpenCV")
        
        try:
            import mediapipe
        except ImportError:
            missing.append("MediaPipe")
        
        try:
            import torch
        except ImportError:
            missing.append("PyTorch")
        
        try:
            from scipy import ndimage
        except ImportError:
            missing.append("SciPy")
        
        if missing:
            self.report({'ERROR'}, f"Missing: {', '.join(missing)}. Run pip install in the system console.")
            return {'CANCELLED'}
        
        self.report({'INFO'}, "All dependencies OK!")
        return {'FINISHED'}

class P2H_OT_auto_detect(bpy.types.Operator):
    """Auto-detect landmarks and depth from both images."""
    bl_idname = "p2h.auto_detect"
    bl_label = "Auto Detect Front & Side"
    bl_description = "Automatically detect facial landmarks and depth maps"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        p = context.scene.p2h
        
        try:
            if not p.p2h_front_image or not p.p2h_side_image:
                self.report({'ERROR'}, 'Select both front and side images first')
                return {'CANCELLED'}
            
            # Load images
            front_rgb = utils.load_image(p.p2h_front_image)
            side_rgb = utils.load_image(p.p2h_side_image)
            
            if front_rgb is None or side_rgb is None:
                self.report({'ERROR'}, 'Failed to load images')
                return {'CANCELLED'}
            
            self.report({'INFO'}, 'Detecting landmarks... (this may take a moment)')
            
            # Detect landmarks
            lm_front = landmark_detector.detect_landmarks(front_rgb)
            lm_side = landmark_detector.detect_landmarks(side_rgb)
            
            if lm_front is None:
                self.report({'ERROR'}, 'No face detected in front image')
                return {'CANCELLED'}
            
            self.report({'INFO'}, 'Estimating depth maps...')
            
            # Estimate depth
            depth_front = depth_estimator.estimate_depth(front_rgb, model_type="midas", 
                                                         scale=p.p2h_depth_scale)
            depth_side = depth_estimator.estimate_depth(side_rgb, model_type="midas", 
                                                        scale=p.p2h_depth_scale) if lm_side is not None else None
            
            if depth_front is None:
                self.report({'ERROR'}, 'Depth estimation failed')
                return {'CANCELLED'}
            
            # Store in scene custom properties for later use
            context.scene["p2h_landmarks_front"] = lm_front.tolist()
            context.scene["p2h_landmarks_side"] = lm_side.tolist() if lm_side is not None else []
            context.scene["p2h_depth_front"] = depth_front.tolist()
            context.scene["p2h_depth_side"] = depth_side.tolist() if depth_side is not None else []
            
            self.report({'INFO'}, f'Detected {len(lm_front)} front landmarks and depth map')
            return {'FINISHED'}
        
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.report({'ERROR'}, str(e)[:200])
            return {'CANCELLED'}

class P2H_OT_build_mesh(bpy.types.Operator):
    """Build the final 3D head mesh."""
    bl_idname = "p2h.build_mesh"
    bl_label = "Build Head Mesh"
    bl_description = "Generate detailed 3D head with eyes, lips, nose holes, and sharp features"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        p = context.scene.p2h
        
        try:
            # Retrieve stored data
            if "p2h_depth_front" not in context.scene:
                self.report({'ERROR'}, 'Run Auto Detect first')
                return {'CANCELLED'}
            
            lm_front = np.array(context.scene.get("p2h_landmarks_front", []), dtype=np.float32)
            lm_side = np.array(context.scene.get("p2h_landmarks_side", []), dtype=np.float32)
            depth_front_list = context.scene.get("p2h_depth_front", [])
            depth_side_list = context.scene.get("p2h_depth_side", [])
            
            # Reconstruct numpy arrays
            if not depth_front_list:
                self.report({'ERROR'}, 'No depth data found')
                return {'CANCELLED'}
            
            depth_front = np.array(depth_front_list, dtype=np.float32)
            if len(depth_front.shape) == 1:
                # Need to reshape - this is a limitation of storing in scene
                w = int(len(depth_front_list) ** 0.5)
                depth_front = depth_front.reshape(w, w)
            
            depth_side = np.array(depth_side_list, dtype=np.float32) if depth_side_list else None
            
            self.report({'INFO'}, 'Building mesh... (generating geometry)')
            
            verts, faces = mesh_builder.build_head_mesh(
                depth_front, depth_side, lm_front, lm_side if len(lm_side) > 0 else None,
                head_height=p.p2h_head_height,
                eye_depth=p.p2h_eye_depth,
                lip_sharpness=p.p2h_lip_sharpness,
                nose_detail=p.p2h_nose_detail,
                rings=p.p2h_rings,
                segments=p.p2h_segments
            )
            
            if verts is None or len(verts) == 0:
                self.report({'ERROR'}, 'Mesh generation failed')
                return {'CANCELLED'}
            
            # Create mesh object
            mesh = bpy.data.meshes.new("Photo2Head_Mesh")
            mesh.from_pydata(verts.tolist(), [], faces)
            mesh.update()
            
            # Smooth shading
            for poly in mesh.polygons:
                poly.use_smooth = True
            
            # Create object
            obj = bpy.data.objects.get("Photo2Head")
            if obj is None:
                obj = bpy.data.objects.new("Photo2Head", mesh)
                utils.get_mesh_collection(context).objects.link(obj)
            else:
                old_mesh = obj.data
                obj.data = mesh
                if old_mesh.users == 0:
                    bpy.data.meshes.remove(old_mesh)
            
            # Add modifiers
            for mod in list(obj.modifiers):
                obj.modifiers.remove(mod)
            
            mirror = obj.modifiers.new(name="Mirror", type='MIRROR')
            mirror.use_axis[0] = True
            mirror.use_clip = True
            mirror.use_mirror_merge = True
            
            subsurf = obj.modifiers.new(name="Subdivision", type='SUBSURF')
            subsurf.levels = 2
            subsurf.render_levels = 3
            
            # Select and focus
            for o in context.selected_objects:
                o.select_set(False)
            obj.select_set(True)
            context.view_layer.objects.active = obj
            
            self.report({'INFO'}, f'Head mesh built: {len(verts)} vertices with detailed eyes, lips, and nose')
            return {'FINISHED'}
        
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.report({'ERROR'}, str(e)[:200])
            return {'CANCELLED'}

class P2H_PT_panel(bpy.types.Panel):
    """Main Photo2Head Depth panel."""
    bl_label = "Photo2Head Depth"
    bl_idname = "P2H_PT_main"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'Photo2Head'

    def draw(self, context):
        layout = self.layout
        p = context.scene.p2h
        
        # Images section
        box = layout.box()
        box.label(text="1. Reference Images", icon='IMAGE_DATA')
        box.prop(p, "p2h_front_image", text="Front")
        box.prop(p, "p2h_side_image", text="Side")
        box.prop(p, "p2h_head_height", text="Head Height")
        box.prop(p, "p2h_neck_length", text="Neck Length")
        
        # Detection section
        box2 = layout.box()
        box2.label(text="2. Detection", icon='FACE_MAPS')
        box2.operator("p2h.setup_check", text="Check Setup", icon='CHECKMARK')
        row = box2.row()
        row.scale_y = 1.5
        row.operator("p2h.auto_detect", text="Auto Detect Landmarks & Depth", icon='ZOOM_ALL')
        
        # Mesh settings
        box3 = layout.box()
        box3.label(text="3. Mesh Settings", icon='MOD_SUBSURF')
        box3.prop(p, "p2h_depth_scale", text="Depth Scale")
        box3.prop(p, "p2h_eye_depth", text="Eye Socket Depth")
        box3.prop(p, "p2h_lip_sharpness", text="Lip Sharpness")
        box3.prop(p, "p2h_nose_detail", text="Nose Detail")
        box3.prop(p, "p2h_rings", text="Rings")
        box3.prop(p, "p2h_segments", text="Segments")
        
        # Build section
        box4 = layout.box()
        box4.label(text="4. Generate", icon='MESH_UVSPHERE')
        row = box4.row()
        row.scale_y = 2.0
        row.operator("p2h.build_mesh", text="Build Head Mesh", icon='SHADING_SOLID')
        
        layout.label(text="Detailed 3D head with eyes, lips, nose holes, sharp creases", icon='INFO')

_CLASSES = (
    P2H_OT_setup_check,
    P2H_OT_auto_detect,
    P2H_OT_build_mesh,
    P2H_PT_panel,
)

def register():
    for cls in _CLASSES:
        bpy.utils.register_class(cls)

def unregister():
    for cls in reversed(_CLASSES):
        bpy.utils.unregister_class(cls)
