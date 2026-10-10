# Photo2Head Depth - Advanced 3D Head Reconstruction

## Overview

Photo2Head Depth is a Blender addon that creates detailed 3D head meshes from front and side reference photos using monocular depth estimation and facial landmark detection.

### Features

- **Depth Capture**: Automatic monocular depth estimation (MiDaS)
- **Precise Landmarks**: 468-point facial landmark detection (MediaPipe)
- **Detailed Geometry**:
  - Eye sockets with anatomical depth
  - Lip creases with sharp boundaries
  - Nostril holes with borders
  - Side profile contours
  - Smooth forehead and jaw surfaces
- **Sharp/Smooth Topology**: Automatic crease assignment for anatomical accuracy
- **Manual Refinement**: Editable marks before mesh generation
- **Mirror Modifier**: Built-in symmetry for full head from half-face model

## Installation

1. Download or clone this repository
2. Place the folder in Blender's addons directory:
   - Windows: `C:\Users\<username>\AppData\Roaming\Blender Foundation\Blender\4.5\scripts\addons`
   - macOS: `/Users/<username>/Library/Application Support/Blender/4.5/scripts/addons`
   - Linux: `~/.config/blender/4.5/scripts/addons`
3. Restart Blender
4. Enable the addon in Edit > Preferences > Add-ons (search "Photo2Head")

## Dependencies

Install these Python packages in your system environment:

```bash
pip install opencv-python mediapipe torch torchvision scipy numpy
```

For faster GPU acceleration:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## Usage

1. **Prepare images**: Take clean front and side photos of the head
   - Front: Look straight into the camera
   - Side: Profile view (left or right)
   - Plain background recommended for best results

2. **Load images**: In the Photo2Head Depth panel, select your front and side images

3. **Check Setup**: Click "Check Setup" to verify all dependencies are installed

4. **Auto Detect**: Click "Auto Detect Landmarks & Depth"
   - MediaPipe detects 468 facial landmarks
   - MiDaS estimates depth map
   - Process takes 10-30 seconds per image

5. **Adjust Settings** (optional):
   - Depth Scale: Multiplier for depth strength (0.5-2.0)
   - Eye Socket Depth: Recess of eye sockets (0.003-0.015m)
   - Lip Sharpness: Crease definition (0-1)
   - Nose Detail: Nostril and bridge definition (0-1)
   - Rings & Segments: Mesh resolution

6. **Build Mesh**: Click "Build Head Mesh"
   - Generates full 3D head with mirror modifier
   - Adds subdivision surface for smoothness
   - Creates detailed eye, lip, and nose geometry

7. **Refine**: Use Blender's sculpt or edit modes to fine-tune

## Settings

### Depth Scale
Adjusts the overall depth strength. Use 1.0 for natural proportions, higher for more pronounced depth.

### Eye Socket Depth
Controlls how recessed the eyes appear. Typical value: 0.008m (8mm).

### Lip Sharpness
Crease definition between lips. 0 = smooth, 1 = sharp crease.

### Nose Detail
Nostril hole depth and bridge definition. 0 = smooth, 1 = anatomically sharp.

### Rings & Segments
Mesh resolution. Higher = smoother but slower. Start with 200/120, adjust as needed.

## Troubleshooting

### "No face detected"
- Ensure front photo shows clear face looking straight at camera
- Try different lighting or background

### "Depth estimation failed"
- Restart Blender if MiDaS model didn't load
- Check PyTorch installation: `python -c "import torch; print(torch.__version__)"`

### Mesh looks bad
- Adjust depth scale and detail settings
- Try different Rings/Segments values
- Use Blender's Sculpt mode for manual refinement

### Memory issues with large images
- Addon automatically resizes images to max 1280px
- Use smaller input photos if needed

## Technical Details

### Landmark Groups
- Face oval (36 points): Jaw and cheek outline
- Eye sockets (16 points each): Eye region geometry
- Nose (10 points): Bridge, tip, base
- Nostrils (12 points): Hole borders and depth
- Lips (15 points each): Upper and lower lip contours
- Brows (10 points): Eyebrow position

### Depth Estimation
Uses Intel's MiDaS (Mobile Depth Sensing) model:
- Input: Single RGB image
- Output: Relative depth map (0-1 normalized)
- MiDaS Small: ~256MB, fast inference
- GPU acceleration available via PyTorch CUDA

### Mesh Generation
1. Extract contours from depth map
2. Sample facial landmarks as control points
3. Build base surface mesh (rings × segments)
4. Add eye socket geometry
5. Create nostril holes
6. Define lip crease lines
7. Apply mirror modifier for full head
8. Subdivide for smoothness

## Known Limitations

- Works best with clear, well-lit reference photos
- Monocular depth estimates relative depth (not absolute scale)
- Ears are simplified (not detailed)
- Very small details (skin pores, wrinkles) not captured
- Requires manual sculpting for final polish

## Performance

- Landmark detection: 5-10 seconds per image
- Depth estimation: 15-30 seconds per image (GPU faster)
- Mesh generation: 2-5 seconds
- Total time: ~1 minute for both images

## License

MIT License - See LICENSE file

## Credits

- **MediaPipe**: Google's facial landmark detection framework
- **MiDaS**: Intel's depth estimation model
- **Blender**: 3D modeling and rendering
- **SciPy/NumPy**: Numerical computation

## Support

For issues, feature requests, or contributions, visit the GitHub repository.
