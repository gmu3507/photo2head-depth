bl_info = {
    "name": "Photo2Head Depth - Advanced 3D Head Reconstruction",
    "author": "Photo2Head",
    "version": (4, 0, 0),
    "blender": (4, 5, 0),
    "location": "View3D > Sidebar > Photo2Head Depth",
    "description": "Front + side photos → depth maps → detailed 3D head mesh with eyes, lips, nose holes, and sharp features.",
    "category": "Mesh",
    "support": "COMMUNITY",
}

from . import utils, depth_estimator, landmark_detector, mesh_builder, operators

def register():
    utils.register_properties()
    operators.register()

def unregister():
    operators.unregister()
    utils.unregister_properties()

if __name__ == "__main__":
    register()
