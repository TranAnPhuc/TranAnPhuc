"""
3D Model Generator for Tran An Phuc GitHub Profile
Generates production 3D models in STL and Wavefront OBJ formats:
1. ai-core.stl / ai-core.obj
2. developer-workspace.stl / developer-workspace.obj
3. tech-matrix.stl / tech-matrix.obj
"""

import struct
import numpy as np
import os

class Mesh:
    def __init__(self, name="Mesh"):
        self.name = name
        self.vertices = []  # list of (x, y, z)
        self.normals = []   # list of (nx, ny, nz)
        self.faces = []     # list of (v1, v2, v3) 0-indexed

    def add_triangle(self, v1, v2, v3, normal=None):
        idx = len(self.vertices)
        self.vertices.extend([v1, v2, v3])
        if normal is None:
            # calculate normal
            a = np.array(v2) - np.array(v1)
            b = np.array(v3) - np.array(v1)
            n = np.cross(a, b)
            norm = np.linalg.norm(n)
            if norm > 1e-6:
                n = n / norm
            else:
                n = np.array([0.0, 0.0, 1.0])
            normal = tuple(n)
        self.normals.extend([normal, normal, normal])
        self.faces.append((idx, idx + 1, idx + 2))

    def add_quad(self, v1, v2, v3, v4, normal=None):
        self.add_triangle(v1, v2, v3, normal)
        self.add_triangle(v1, v3, v4, normal)

    def add_box(self, center, size, rot_y=0.0):
        cx, cy, cz = center
        sx, sy, sz = [s / 2.0 for s in size]
        
        # 8 box corners relative to center
        corners = [
            [-sx, -sy, -sz],
            [ sx, -sy, -sz],
            [ sx,  sy, -sz],
            [-sx,  sy, -sz],
            [-sx, -sy,  sz],
            [ sx, -sy,  sz],
            [ sx,  sy,  sz],
            [-sx,  sy,  sz],
        ]
        
        # Apply Y rotation if any
        if abs(rot_y) > 1e-6:
            rad = np.radians(rot_y)
            cos_r, sin_r = np.cos(rad), np.sin(rad)
            rotated = []
            for x, y, z in corners:
                rx = x * cos_r + z * sin_r
                rz = -x * sin_r + z * cos_r
                rotated.append([rx, y, rz])
            corners = rotated

        pts = [[cx + x, cy + y, cz + z] for x, y, z in corners]

        # Front (Z+)
        self.add_quad(pts[4], pts[5], pts[6], pts[7])
        # Back (Z-)
        self.add_quad(pts[1], pts[0], pts[3], pts[2])
        # Top (Y+)
        self.add_quad(pts[7], pts[6], pts[2], pts[3])
        # Bottom (Y-)
        self.add_quad(pts[0], pts[1], pts[5], pts[4])
        # Right (X+)
        self.add_quad(pts[5], pts[1], pts[2], pts[6])
        # Left (X-)
        self.add_quad(pts[0], pts[4], pts[7], pts[3])

    def add_cylinder(self, center, radius, height, segments=24, axis='y'):
        cx, cy, cz = center
        h2 = height / 2.0
        angles = np.linspace(0, 2 * np.pi, segments, endpoint=False)
        
        top_pts = []
        bot_pts = []
        for a in angles:
            x = radius * np.cos(a)
            z = radius * np.sin(a)
            if axis == 'y':
                top_pts.append([cx + x, cy + h2, cz + z])
                bot_pts.append([cx + x, cy - h2, cz + z])
            elif axis == 'z':
                top_pts.append([cx + x, cy + z, cz + h2])
                bot_pts.append([cx + x, cy + z, cz - h2])

        top_center = [cx, cy + h2 if axis == 'y' else cy, cz if axis == 'y' else cz + h2]
        bot_center = [cx, cy - h2 if axis == 'y' else cy, cz if axis == 'y' else cz - h2]

        for i in range(segments):
            next_i = (i + 1) % segments
            # Top cap
            self.add_triangle(top_center, top_pts[i], top_pts[next_i])
            # Bottom cap
            self.add_triangle(bot_center, bot_pts[next_i], bot_pts[i])
            # Side quads
            self.add_quad(bot_pts[i], bot_pts[next_i], top_pts[next_i], top_pts[i])

    def add_torus(self, center, major_r, minor_r, major_segs=32, minor_segs=12, rot=(0,0,0)):
        cx, cy, cz = center
        rx, ry, rz = [np.radians(a) for a in rot]
        
        # Rotation matrices
        Rx = np.array([[1, 0, 0], [0, np.cos(rx), -np.sin(rx)], [0, np.sin(rx), np.cos(rx)]])
        Ry = np.array([[np.cos(ry), 0, np.sin(ry)], [0, 1, 0], [-np.sin(ry), 0, np.cos(ry)]])
        Rz = np.array([[np.cos(rz), -np.sin(rz), 0], [np.sin(rz), np.cos(rz), 0], [0, 0, 1]])
        R = Rz @ Ry @ Rx

        u = np.linspace(0, 2 * np.pi, major_segs, endpoint=False)
        v = np.linspace(0, 2 * np.pi, minor_segs, endpoint=False)

        grid = []
        for i in range(major_segs):
            row = []
            for j in range(minor_segs):
                x = (major_r + minor_r * np.cos(v[j])) * np.cos(u[i])
                y = minor_r * np.sin(v[j])
                z = (major_r + minor_r * np.cos(v[j])) * np.sin(u[i])
                p = R @ np.array([x, y, z]) + np.array([cx, cy, cz])
                row.append(p.tolist())
            grid.append(row)

        for i in range(major_segs):
            next_i = (i + 1) % major_segs
            for j in range(minor_segs):
                next_j = (j + 1) % minor_segs
                self.add_quad(grid[i][j], grid[next_i][j], grid[next_i][next_j], grid[i][next_j])

    def add_octahedron(self, center, size):
        cx, cy, cz = center
        s = size / 2.0
        top = [cx, cy + s * 1.4, cz]
        bot = [cx, cy - s * 1.4, cz]
        e1 = [cx + s, cy, cz]
        e2 = [cx, cy, cz + s]
        e3 = [cx - s, cy, cz]
        e4 = [cx, cy, cz - s]

        # Top 4 faces
        self.add_triangle(top, e1, e2)
        self.add_triangle(top, e2, e3)
        self.add_triangle(top, e3, e4)
        self.add_triangle(top, e4, e1)
        # Bottom 4 faces
        self.add_triangle(bot, e2, e1)
        self.add_triangle(bot, e3, e2)
        self.add_triangle(bot, e4, e3)
        self.add_triangle(bot, e1, e4)

    def export_stl_binary(self, filename):
        with open(filename, 'wb') as f:
            # 80-byte header
            header = f"3D Model: {self.name} | Tran An Phuc AI Workspace".encode('ascii')
            header = header[:80].ljust(80, b'\0')
            f.write(header)
            
            # Number of triangles (uint32)
            n_triangles = len(self.faces)
            f.write(struct.pack('<I', n_triangles))
            
            # Triangle records (50 bytes each)
            for face in self.faces:
                v1 = self.vertices[face[0]]
                v2 = self.vertices[face[1]]
                v3 = self.vertices[face[2]]
                n = self.normals[face[0]]
                
                f.write(struct.pack('<3f', float(n[0]), float(n[1]), float(n[2])))
                f.write(struct.pack('<3f', float(v1[0]), float(v1[1]), float(v1[2])))
                f.write(struct.pack('<3f', float(v2[0]), float(v2[1]), float(v2[2])))
                f.write(struct.pack('<3f', float(v3[0]), float(v3[1]), float(v3[2])))
                f.write(struct.pack('<H', 0)) # attribute byte count

    def export_obj(self, filename, mtl_filename=None):
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(f"# Wavefront OBJ File: {self.name}\n")
            f.write(f"# Designed for Tran An Phuc (anphuctran2005) 3D Workspace\n")
            if mtl_filename:
                f.write(f"mtllib {os.path.basename(mtl_filename)}\n")
            f.write(f"o {self.name}\n\n")

            for v in self.vertices:
                f.write(f"v {v[0]:.4f} {v[1]:.4f} {v[2]:.4f}\n")
            f.write("\n")

            for n in self.normals:
                f.write(f"vn {n[0]:.4f} {n[1]:.4f} {n[2]:.4f}\n")
            f.write("\n")

            if mtl_filename:
                f.write("usemtl CyberMaterial\ns 1\n")

            for face in self.faces:
                # 1-indexed for OBJ
                i1, i2, i3 = face[0] + 1, face[1] + 1, face[2] + 1
                f.write(f"f {i1}//{i1} {i2}//{i2} {i3}//{i3}\n")


def generate_ai_core():
    mesh = Mesh("AICore")
    
    # 1. Base pedestal
    mesh.add_cylinder(center=(0, -2.5, 0), radius=2.2, height=0.4, segments=32)
    mesh.add_cylinder(center=(0, -2.2, 0), radius=1.7, height=0.3, segments=24)
    mesh.add_cylinder(center=(0, -1.8, 0), radius=1.1, height=0.5, segments=16)

    # 2. Central Polyhedral AI Crystal (Multi-layered Octahedron & Star Core)
    mesh.add_octahedron(center=(0, 0, 0), size=2.0)
    mesh.add_octahedron(center=(0, 0, 0), size=1.4)
    mesh.add_octahedron(center=(0, 0, 0), size=0.8)

    # 3. Triple Orbital Gimbal Rings
    # Outer Ring 1 (tilted 35 deg X, 15 deg Z)
    mesh.add_torus(center=(0, 0, 0), major_r=2.5, minor_r=0.08, major_segs=48, minor_segs=10, rot=(35, 0, 15))
    # Middle Ring 2 (tilted -45 deg X, 40 deg Y)
    mesh.add_torus(center=(0, 0, 0), major_r=3.2, minor_r=0.09, major_segs=48, minor_segs=10, rot=(-45, 40, 0))
    # Equator Ring 3 (horizontal with slight pitch)
    mesh.add_torus(center=(0, 0, 0), major_r=3.8, minor_r=0.07, major_segs=48, minor_segs=10, rot=(10, 0, -20))

    # 4. Four Floating Satellite Tech Modules
    sats = [
        # (x, y, z), size, name
        ((-2.8, 1.2, 0.8), (0.7, 0.7, 0.7), "TS_Module"),
        ((2.8, 1.4, -0.6), (0.7, 0.7, 0.7), "React_Module"),
        ((-2.4, -1.5, 1.2), (0.65, 0.65, 0.65), "Node_Module"),
        ((2.6, -1.3, -1.0), (0.65, 0.65, 0.65), "Postgres_Module")
    ]
    for pos, sz, sname in sats:
        mesh.add_box(center=pos, size=sz)
        # Antenna / Beacon pin on each satellite
        mesh.add_cylinder(center=(pos[0], pos[1] + sz[1]*0.6, pos[2]), radius=0.04, height=0.3, segments=8)
        # Small orbital node
        mesh.add_octahedron(center=(pos[0], pos[1] + sz[1]*0.9, pos[2]), size=0.15)

    return mesh


def generate_developer_workspace():
    mesh = Mesh("DeveloperWorkspace")

    # 1. Main Desk Top
    mesh.add_box(center=(0, 0, 0), size=(6.0, 0.15, 2.8))
    # Desk bevel trim
    mesh.add_box(center=(0, -0.1, 0), size=(5.8, 0.1, 2.6))
    
    # 2. Cybernetic Desk Legs / Truss Supports
    mesh.add_box(center=(-2.6, -1.5, 0), size=(0.2, 2.9, 2.2))
    mesh.add_box(center=(2.6, -1.5, 0), size=(0.2, 2.9, 2.2))
    # Crossbar reinforcement
    mesh.add_cylinder(center=(0, -1.8, 0), radius=0.08, height=5.2, segments=12, axis='z')

    # 3. Triple Curved Monitor Display Setup
    # Center Ultrawide Display
    mesh.add_box(center=(0, 1.5, -0.6), size=(2.8, 1.3, 0.08))
    mesh.add_box(center=(0, 1.5, -0.62), size=(2.86, 1.36, 0.04)) # Bezel
    mesh.add_cylinder(center=(0, 0.75, -0.7), radius=0.06, height=1.3, segments=12) # Stand arm
    mesh.add_cylinder(center=(0, 0.1, -0.7), radius=0.4, height=0.05, segments=16) # Stand base

    # Left Monitor (Angled 28 deg toward center)
    mesh.add_box(center=(-2.1, 1.5, -0.3), size=(1.4, 1.2, 0.08), rot_y=28.0)
    mesh.add_cylinder(center=(-2.0, 0.75, -0.45), radius=0.05, height=1.3, segments=12)

    # Right Monitor (Angled -28 deg toward center)
    mesh.add_box(center=(2.1, 1.5, -0.3), size=(1.4, 1.2, 0.08), rot_y=-28.0)
    mesh.add_cylinder(center=(2.0, 0.75, -0.45), radius=0.05, height=1.3, segments=12)

    # 4. Cyber Mechanical Keyboard & Trackpad
    mesh.add_box(center=(-0.3, 0.12, 0.5), size=(1.5, 0.06, 0.6))
    # Keycap array block
    mesh.add_box(center=(-0.3, 0.17, 0.5), size=(1.42, 0.04, 0.54))
    # Precision Trackpad / Mouse
    mesh.add_box(center=(0.85, 0.11, 0.5), size=(0.3, 0.04, 0.45))

    # 5. AI Neural Compute Tower / Rig (Under or on side of desk)
    mesh.add_box(center=(2.2, 0.8, 0.5), size=(0.6, 1.4, 1.3))
    # Ventilation grilles & chassis glass panel
    mesh.add_box(center=(2.2, 0.8, -0.16), size=(0.54, 1.3, 0.02))
    mesh.add_box(center=(1.89, 0.8, 0.5), size=(0.02, 1.3, 1.24))

    return mesh


def generate_tech_matrix():
    mesh = Mesh("TechMatrix")

    # 4 Modular High-Tech Computing Nodes arranged in 2x2 with vertical elevation
    nodes = [
        ((-1.5, 1.0, 0), (1.4, 1.2, 1.4), "Frontend_Node"),
        (( 1.5, 1.0, 0), (1.4, 1.2, 1.4), "Backend_Node"),
        ((-1.5, -1.0, 0), (1.4, 1.2, 1.4), "Tooling_Node"),
        (( 1.5, -1.0, 0), (1.4, 1.2, 1.4), "AI_Agent_Node")
    ]

    for pos, sz, name in nodes:
        # Main Node Chassis
        mesh.add_box(center=pos, size=sz)
        # Inner glowing core window
        mesh.add_box(center=(pos[0], pos[1], pos[2] + sz[2]*0.52), size=(sz[0]*0.7, sz[1]*0.7, 0.04))
        # Top heat sink fins
        for k in [-0.3, 0.0, 0.3]:
            mesh.add_box(center=(pos[0] + k, pos[1] + sz[1]*0.55, pos[2]), size=(0.1, 0.1, sz[2]*0.8))

    # Connecting High-Speed Data Bus Conduit Tubes
    # Horizontal Top (Frontend <-> Backend)
    mesh.add_cylinder(center=(0, 1.0, 0), radius=0.12, height=3.0, segments=16, axis='z')
    # Horizontal Bottom (Tooling <-> AI)
    mesh.add_cylinder(center=(0, -1.0, 0), radius=0.12, height=3.0, segments=16, axis='z')
    # Vertical Left (Frontend <-> Tooling)
    mesh.add_cylinder(center=(-1.5, 0, 0), radius=0.12, height=2.0, segments=16, axis='y')
    # Vertical Right (Backend <-> AI)
    mesh.add_cylinder(center=(1.5, 0, 0), radius=0.12, height=2.0, segments=16, axis='y')

    # Central Data Hub Core
    mesh.add_octahedron(center=(0, 0, 0), size=0.9)
    mesh.add_torus(center=(0, 0, 0), major_r=0.8, minor_r=0.06, major_segs=24, minor_segs=8)

    return mesh


def write_mtl_file(filepath):
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write("# Cyber Material Definitions for 3D AI Workspace\n")
        f.write("newmtl CyberMaterial\n")
        f.write("Ka 0.05 0.06 0.09\n")       # Ambient (Deep obsidian)
        f.write("Kd 0.08 0.71 0.83\n")       # Diffuse (Cyan #06B6D4)
        f.write("Ks 0.49 0.23 0.93\n")       # Specular (Electric Violet #7C3AED)
        f.write("Ke 0.02 0.20 0.25\n")       # Emissive glow
        f.write("Ns 60.0\n")                 # Shininess
        f.write("d 1.0\n")                   # Opacity
        f.write("illum 2\n")


def main():
    out_dir = "assets/models"
    os.makedirs(out_dir, exist_ok=True)
    
    mtl_file = os.path.join(out_dir, "cyber-materials.mtl")
    write_mtl_file(mtl_file)
    print(f"Created: {mtl_file}")

    models = [
        ("ai-core", generate_ai_core()),
        ("developer-workspace", generate_developer_workspace()),
        ("tech-matrix", generate_tech_matrix())
    ]

    for name, mesh in models:
        stl_path = os.path.join(out_dir, f"{name}.stl")
        obj_path = os.path.join(out_dir, f"{name}.obj")
        
        mesh.export_stl_binary(stl_path)
        mesh.export_obj(obj_path, mtl_filename="cyber-materials.mtl")
        
        file_size_stl = os.path.getsize(stl_path)
        file_size_obj = os.path.getsize(obj_path)
        print(f"Exported {name}:")
        print(f"  - STL: {stl_path} ({file_size_stl} bytes, {len(mesh.faces)} triangles)")
        print(f"  - OBJ: {obj_path} ({file_size_obj} bytes, {len(mesh.vertices)} vertices)")

if __name__ == "__main__":
    main()
