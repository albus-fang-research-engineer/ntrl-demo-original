"""
Load pre-optimized UR5 trajectories and run a per-waypoint collision check
using actual URDF link meshes (vertices) instead of bounding spheres.

Approach:
  - Parse URDF to find visual/collision mesh files per link
  - Load mesh vertices with trimesh
  - Apply FK transforms (pytorch_kinematics) to get world-frame vertices
  - Query obstacle KD-tree; collision if any vertex is within threshold
"""

import sys, math, re
import numpy as np
import torch
import igl
import trimesh
import pytorch_kinematics as pk
import torch_kdtree
import xml.etree.ElementTree as ET
from pathlib import Path

# ------------------------------------------------------------------ #
# Paths / config
# ------------------------------------------------------------------ #
IN_DIR     = Path("Evaluations/Arm/optimized_paths")
MESH_FILE  = "./datasets/arm/UR5/realpc_scaled.off"
URDF_PATH  = "datasets/arm/UR5/ur5e.urdf"
NUM_PATHS  = 100

# Positive = clearance gap treated as collision (set 0.0 for exact surface)
COLLISION_THRESHOLD = 0.0

# Links to include (indices 2-7 in the FK chain, same window as before)
LINK_IDX_RANGE = range(2, 8)

# Subsample vertices per link to keep GPU memory reasonable (None = use all)
MAX_VERTS_PER_LINK = 512


# ------------------------------------------------------------------ #
# Parse URDF → {link_name: absolute mesh path}
# ------------------------------------------------------------------ #
def parse_urdf_meshes(urdf_path: str) -> dict:
    """
    Returns {link_name: Path} for every link that has a collision mesh.
    Resolves 'package://' and relative paths relative to the URDF directory.
    """
    urdf_dir = Path(urdf_path).parent
    tree     = ET.parse(urdf_path)
    root     = tree.getroot()

    link_meshes = {}
    for link in root.iter("link"):
        name = link.get("name")
        # prefer <collision> mesh; fall back to <visual>
        for tag in ("collision", "visual"):
            geo = link.find(f"{tag}/geometry/mesh")
            if geo is not None:
                fname = geo.get("filename", "")
                # resolve package:// URIs
                fname = re.sub(r"^package://[^/]+/", "", fname)
                full  = (urdf_dir / fname).resolve()
                if full.exists():
                    link_meshes[name] = full
                break  # found for this link, move on

    return link_meshes


# ------------------------------------------------------------------ #
# Build FK chain + per-link mesh vertex tensors
# ------------------------------------------------------------------ #
def build_chain_and_meshes(device: str):
    end_link = "wrist_3_link"
    chain    = pk.build_serial_chain_from_urdf(
        open(URDF_PATH).read(), end_link)
    chain    = chain.to(dtype=torch.float32, device=device)

    # One forward pass to discover link names in order
    dummy   = torch.zeros(1, 6, device=device)
    tg      = chain.forward_kinematics(dummy, end_only=False)
    ordered_links = list(tg.keys())          # FK order

    link_mesh_paths = parse_urdf_meshes(URDF_PATH)

    mesh_tensors = {}   # link_name → (V, 3) float32 on device
    for idx, link_name in enumerate(ordered_links):
        if idx not in LINK_IDX_RANGE:
            continue
        if link_name not in link_mesh_paths:
            print(f"  [warn] no mesh for link '{link_name}', skipping")
            continue

        mesh = trimesh.load(str(link_mesh_paths[link_name]),
                            force="mesh", process=False)
        verts = np.array(mesh.vertices, dtype=np.float32)

        # Optional: subsample to cap memory usage
        if MAX_VERTS_PER_LINK and len(verts) > MAX_VERTS_PER_LINK:
            idx_s  = np.random.choice(len(verts), MAX_VERTS_PER_LINK,
                                      replace=False)
            verts  = verts[idx_s]

        mesh_tensors[link_name] = torch.tensor(verts, device=device)
        print(f"  link '{link_name}': {len(verts)} vertices  "
              f"(path: {link_mesh_paths[link_name].name})")

    return chain, ordered_links, mesh_tensors


# ------------------------------------------------------------------ #
# FK → world-frame vertices for all links, all configs
# ------------------------------------------------------------------ #
def fk_mesh_vertices(joint_angles_rad: torch.Tensor,
                     chain, ordered_links, mesh_tensors) -> torch.Tensor:
    """
    joint_angles_rad : (N, 6)
    Returns          : (N, V_total, 3)  world-frame vertices
    """
    N  = joint_angles_rad.shape[0]
    tg = chain.forward_kinematics(joint_angles_rad, end_only=False)

    all_verts = []
    for idx, link_name in enumerate(ordered_links):
        if idx not in LINK_IDX_RANGE:
            continue
        if link_name not in mesh_tensors:
            continue

        verts_local = mesh_tensors[link_name]   # (V, 3)
        V           = verts_local.shape[0]

        # Homogeneous: (V, 4)
        ones        = torch.ones(V, 1, device=verts_local.device)
        verts_h     = torch.cat([verts_local, ones], dim=1)   # (V, 4)

        # Transform matrix: (N, 4, 4)
        T           = tg[link_name].get_matrix()              # (N, 4, 4)

        # Broadcast: (N, V, 4) = (N, 4, 4) @ (1, 4, V) → permute
        verts_world = torch.bmm(
            T,
            verts_h.T.unsqueeze(0).expand(N, -1, -1)         # (N, 4, V)
        ).permute(0, 2, 1)[..., :3]                           # (N, V, 3)

        all_verts.append(verts_world)

    return torch.cat(all_verts, dim=1)   # (N, V_total, 3)


# ------------------------------------------------------------------ #
# Collision distance for a batch of configs
# ------------------------------------------------------------------ #
def arm_collision_distance(joint_batch_rad: torch.Tensor,
                           chain, ordered_links,
                           mesh_tensors, kdtree) -> torch.Tensor:
    """
    Returns (N,) minimum distance to obstacle surface.
    Positive = free space, ≤ 0 = in collision (or within threshold).
    """
    verts          = fk_mesh_vertices(
        joint_batch_rad, chain, ordered_links, mesh_tensors)  # (N, V, 3)
    N, V, _        = verts.shape

    flat           = verts.reshape(-1, 3)                     # (N*V, 3)

    # KD-tree returns *squared* distances
    sq_dists, _    = kdtree.query(flat, nr_nns_searches=1)
    dists          = torch.sqrt(sq_dists.squeeze())           # (N*V,)

    dists          = dists.reshape(N, V)
    min_dist, _    = dists.min(dim=1)                         # (N,)
    return min_dist


# ------------------------------------------------------------------ #
# Main
# ------------------------------------------------------------------ #
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}\n")

    print("=== Building FK chain + loading link meshes ...")
    chain, ordered_links, mesh_tensors = build_chain_and_meshes(device)
    total_verts = sum(v.shape[0] for v in mesh_tensors.values())
    print(f"  Total robot vertices per config: {total_verts}\n")

    print("=== Loading obstacle mesh ...")
    v, _   = igl.read_triangle_mesh(MESH_FILE)
    v_obs  = torch.tensor(v, dtype=torch.float32, device=device)
    kdtree = torch_kdtree.build_kd_tree(v_obs)
    print(f"  Obstacle mesh vertices: {v_obs.shape[0]}\n")

    collision_free_count = 0
    checked_count        = 0
    skipped_count        = 0
    min_distances        = []

    for i in range(NUM_PATHS):
        traj_path = IN_DIR / f"path_{i:03d}.npy"

        if not traj_path.exists():
            print(f"[{i+1:03d}/{NUM_PATHS}]  MISSING — skipping")
            skipped_count += 1
            continue

        traj_rad = np.asarray(np.load(traj_path, allow_pickle=True))
        waypoints = torch.tensor(traj_rad, dtype=torch.float32, device=device)
        print(f"[{i+1:03d}/{NUM_PATHS}]  shape={traj_rad.shape}", end="  ")

        dist         = arm_collision_distance(
            waypoints, chain, ordered_links, mesh_tensors, kdtree)
        in_collision = dist <= COLLISION_THRESHOLD

        n_col    = int(in_collision.sum().item())
        min_dist = float(dist.min().item())
        min_distances.append(min_dist)
        checked_count += 1

        if n_col == 0:
            collision_free_count += 1
            print(f"COLLISION-FREE   min_dist={min_dist:.4f}")
        else:
            col_idx = in_collision.nonzero(as_tuple=True)[0].tolist()
            print(f"COLLISION        min_dist={min_dist:.4f}  "
                  f"colliding waypoints={col_idx}")

    print("\n" + "=" * 60)
    print(f"Trajectories checked    : {checked_count}/{NUM_PATHS}")
    print(f"Skipped (missing file)  : {skipped_count}/{NUM_PATHS}")
    print(f"Collision-free          : {collision_free_count}/{checked_count}")
    print(f"In collision            : {checked_count - collision_free_count}/{checked_count}")
    if min_distances:
        print(f"Min distance across all : {min(min_distances):.4f}")
        print(f"Mean min distance       : {np.mean(min_distances):.4f}")


if __name__ == "__main__":
    main()