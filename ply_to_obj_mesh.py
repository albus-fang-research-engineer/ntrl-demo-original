import open3d as o3d
import sys
import os

def convert_ply_to_obj(input_path, output_path):
    # Try reading as triangle mesh first
    mesh = o3d.io.read_triangle_mesh(input_path)

    if mesh.is_empty():
        print("Input is not a mesh, trying as point cloud...")
        pcd = o3d.io.read_point_cloud(input_path)

        if pcd.is_empty():
            raise ValueError("Failed to read PLY file.")

        # Convert point cloud to mesh (basic Poisson reconstruction)
        print("Running Poisson surface reconstruction...")
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcd, depth=8
        )

        # Optional: remove low-density vertices
        vertices_to_remove = densities < densities.mean()
        mesh.remove_vertices_by_mask(vertices_to_remove)

    # Save as OBJ
    o3d.io.write_triangle_mesh(output_path, mesh)
    print(f"Saved OBJ to: {output_path}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python convert.py input.ply output.obj")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]

    if not os.path.exists(input_file):
        raise FileNotFoundError(f"{input_file} not found")

    convert_ply_to_obj(input_file, output_file)