"""
Renders a 360-degree rotating animation of the 3D STL model into an optimized animated WebP.
Uses pure numpy vector math + Pillow for software z-buffering.
"""

import struct
import numpy as np
from PIL import Image, ImageDraw
import os

def load_stl(filename):
    with open(filename, 'rb') as f:
        header = f.read(80)
        n_triangles = struct.unpack('<I', f.read(4))[0]
        normals = []
        triangles = []
        for _ in range(n_triangles):
            data = f.read(50)
            n = struct.unpack('<3f', data[0:12])
            v1 = struct.unpack('<3f', data[12:24])
            v2 = struct.unpack('<3f', data[24:36])
            v3 = struct.unpack('<3f', data[36:48])
            normals.append(n)
            triangles.append((v1, v2, v3))
    return np.array(normals), np.array(triangles)

def render_model_rotation(stl_path, out_path, width=480, height=480, n_frames=36):
    normals, triangles = load_stl(stl_path)
    print(f"Loaded {len(triangles)} triangles from {stl_path}")
    
    # Center and scale model to fit in [-1.5, 1.5]
    all_verts = triangles.reshape(-1, 3)
    min_v = all_verts.min(axis=0)
    max_v = all_verts.max(axis=0)
    center = (min_v + max_v) / 2.0
    scale = 3.2 / np.max(max_v - min_v)
    
    norm_tris = (triangles - center) * scale
    
    # Camera settings
    cam_dist = 4.5
    light_dir = np.array([0.4, 0.8, 1.0])
    light_dir = light_dir / np.linalg.norm(light_dir)
    
    # Colors
    bg_color = (8, 9, 13)
    base_color_cyan = np.array([6, 182, 212], dtype=float)
    base_color_purple = np.array([124, 58, 237], dtype=float)

    frames = []
    
    angles = np.linspace(0, 360, n_frames, endpoint=False)
    for frame_idx, ang in enumerate(angles):
        rad = np.radians(ang)
        pitch = np.radians(22)  # slight isometric tilt
        
        # Rotation matrix: Y-axis rotation then slight X-axis tilt
        cos_y, sin_y = np.cos(rad), np.sin(rad)
        cos_x, sin_x = np.cos(pitch), np.sin(pitch)
        
        Ry = np.array([[cos_y, 0, sin_y], [0, 1, 0], [-sin_y, 0, cos_y]])
        Rx = np.array([[1, 0, 0], [0, cos_x, -sin_x], [0, sin_x, cos_x]])
        R = Rx @ Ry
        
        # Transform vertices
        v_rot = np.einsum('ij,klj->kli', R, norm_tris)
        
        # Compute face depths and face normals
        # Face normal from transformed vertices
        e1 = v_rot[:, 1] - v_rot[:, 0]
        e2 = v_rot[:, 2] - v_rot[:, 0]
        fn = np.cross(e1, e2)
        fn_norm = np.linalg.norm(fn, axis=1, keepdims=True)
        fn_norm[fn_norm == 0] = 1.0
        fn = fn / fn_norm
        
        # Back-face culling (faces pointing away from Z+ camera)
        # Camera looks down -Z towards origin, so normal.z > 0 is facing camera
        visible_mask = fn[:, 2] > -0.05
        
        vis_tris = v_rot[visible_mask]
        vis_fn = fn[visible_mask]
        
        # Compute depths for painter's algorithm (Z sort)
        face_depths = np.mean(vis_tris[:, :, 2], axis=1)
        sort_order = np.argsort(face_depths) # back to front
        
        # Perspective projection
        # Camera is at (0, 0, cam_dist)
        fov = width * 1.0
        z_denom = cam_dist - vis_tris[:, :, 2]
        z_denom[z_denom < 0.1] = 0.1
        
        px = (vis_tris[:, :, 0] / z_denom) * fov + width / 2.0
        py = (-vis_tris[:, :, 1] / z_denom) * fov + height / 2.0
        
        # Render image
        img = Image.new("RGB", (width, height), bg_color)
        draw = ImageDraw.Draw(img)
        
        # Draw grid horizon on bottom
        for gx in range(0, width, 40):
            draw.line([(gx, height - 30), (width // 2, height - 120)], fill=(17, 24, 39), width=1)
        
        for idx in sort_order:
            tri_px = list(zip(px[idx], py[idx]))
            normal = vis_fn[idx]
            
            # Lighting calculation
            diff = max(0.1, np.dot(normal, light_dir))
            spec = max(0.0, np.dot(normal, np.array([0, 0, 1]))) ** 6 * 0.4
            
            # Dual color gradient based on Y position
            y_ratio = max(0.0, min(1.0, (vis_tris[idx, :, 1].mean() + 1.2) / 2.4))
            col = (1.0 - y_ratio) * base_color_cyan + y_ratio * base_color_purple
            final_col = np.clip(col * diff + 255.0 * spec, 0, 255).astype(int)
            
            outline_col = tuple(np.clip(final_col * 0.6 + 40, 0, 255).astype(int))
            draw.polygon(tri_px, fill=tuple(final_col), outline=outline_col)
            
        frames.append(img)
        if frame_idx % 9 == 0:
            print(f"Rendered frame {frame_idx + 1}/{n_frames}")
            
    # Save as animated WebP
    frames[0].save(
        out_path,
        save_all=True,
        append_images=frames[1:],
        duration=60, # ~16 fps
        loop=0,
        quality=90,
        method=4
    )
    print(f"Saved animated 3D render to: {out_path} ({os.path.getsize(out_path)} bytes)")

if __name__ == "__main__":
    models_to_render = [
        ("assets/models/ai-core.stl", "assets/models/ai-core-3d-rotating.webp", 440, 440, 36),
        ("assets/models/developer-workspace.stl", "assets/models/developer-workspace-3d-rotating.webp", 440, 440, 36)
    ]
    for stl_path, out_path, w, h, frames in models_to_render:
        render_model_rotation(stl_path, out_path, width=w, height=h, n_frames=frames)
