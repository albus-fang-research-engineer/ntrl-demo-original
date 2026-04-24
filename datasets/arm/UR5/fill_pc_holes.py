import open3d as o3d
import numpy as np

# =========================
# User settings
# =========================
input_obj_path = "fused_all_denoise.obj"
output_pointcloud_path = "filled_cloud.ply"
output_mesh_path = "poisson_mesh.ply"

voxel_size = 0.003
normal_radius = 0.01
normal_max_nn = 30

poisson_depth = 9
density_quantile = 0.02
num_sample_points = 150000

# =========================
# Load vertices-only OBJ as point cloud
# =========================
def load_obj_vertices_as_point_cloud(path):
    points = []
    with open(path, "r") as f:
        for line in f:
            if line.startswith("v "):
                parts = line.strip().split()
                if len(parts) >= 4:
                    x, y, z = map(float, parts[1:4])
                    points.append([x, y, z])

    if len(points) == 0:
        raise ValueError("No vertex lines found in OBJ.")

    points = np.asarray(points, dtype=np.float64)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    return pcd

pcd = load_obj_vertices_as_point_cloud(input_obj_path)
print(f"Loaded {len(pcd.points)} points from OBJ")

# =========================
# Preprocess
# =========================
pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
print(f"After voxel downsampling: {len(pcd.points)} points")

pcd, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
print(f"After outlier removal: {len(pcd.points)} points")

# =========================
# Estimate normals
# =========================
pcd.estimate_normals(
    search_param=o3d.geometry.KDTreeSearchParamHybrid(
        radius=normal_radius,
        max_nn=normal_max_nn
    )
)

pcd.orient_normals_consistent_tangent_plane(50)

# =========================
# Poisson reconstruction
# =========================
mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
    pcd, depth=poisson_depth
)

print(f"Poisson mesh has {len(mesh.vertices)} vertices and {len(mesh.triangles)} triangles")

densities = np.asarray(densities)
density_threshold = np.quantile(densities, density_quantile)
vertices_to_remove = densities < density_threshold
mesh.remove_vertices_by_mask(vertices_to_remove)

print(f"Trimmed mesh has {len(mesh.vertices)} vertices and {len(mesh.triangles)} triangles")

# Crop to original bbox
bbox = pcd.get_axis_aligned_bounding_box()
bbox = bbox.scale(1.02, bbox.get_center())
mesh = mesh.crop(bbox)

print(f"Cropped mesh has {len(mesh.vertices)} vertices and {len(mesh.triangles)} triangles")

# =========================
# Sample back to point cloud
# =========================
filled_pcd = mesh.sample_points_poisson_disk(number_of_points=num_sample_points)
print(f"Sampled filled point cloud has {len(filled_pcd.points)} points")

filled_pcd.estimate_normals(
    search_param=o3d.geometry.KDTreeSearchParamHybrid(
        radius=normal_radius,
        max_nn=normal_max_nn
    )
)

# =========================
# Save outputs
# =========================
o3d.io.write_triangle_mesh(output_mesh_path, mesh)
o3d.io.write_point_cloud(output_pointcloud_path, filled_pcd)

print(f"Saved mesh to: {output_mesh_path}")
print(f"Saved filled point cloud to: {output_pointcloud_path}")

o3d.visualization.draw_geometries([filled_pcd], window_name="Filled Point Cloud")