import numpy as np
import matplotlib.pyplot as plt

path = "./maze"

# -------------------------------
# Load obstacle point cloud
# -------------------------------
pts = np.load(f"{path}/obstacle_points.npy")
x_obs = pts[:,0]
y_obs = pts[:,1]

# -------------------------------
# Load samples + MC normals
# -------------------------------
samples = np.load(f"{path}/sampled_points.npy")
normal  = np.load(f"{path}/normal.npy")

# start points
x0 = samples[:,0]
y0 = samples[:,1]
n0x = normal[:,0]
n0y = normal[:,1]

# goal points
x1 = samples[:,2]
y1 = samples[:,3]
n1x = normal[:,2]
n1y = normal[:,3]

# -------------------------------
# tuning knobs
# -------------------------------
skip  = 80
scale = 40
width = 0.003

# ===============================
# START NORMALS
# ===============================
plt.figure(figsize=(6,6))
plt.scatter(x_obs, y_obs, s=0.3, alpha=0.5, color="black")

plt.quiver(
    x0[::skip], y0[::skip],
    n0x[::skip], n0y[::skip],
    angles="xy", scale_units="xy", scale=scale,
    width=width, color="red", alpha=0.8
)

plt.axis("equal")
plt.title("Start-Point Monte-Carlo Mean Normals")
plt.xlabel("x")
plt.ylabel("y")
plt.tight_layout()
plt.show()

# ===============================
# GOAL NORMALS
# ===============================
plt.figure(figsize=(6,6))
plt.scatter(x_obs, y_obs, s=0.3, alpha=0.5, color="black")

plt.quiver(
    x1[::skip], y1[::skip],
    n1x[::skip], n1y[::skip],
    angles="xy", scale_units="xy", scale=scale,
    width=width, color="blue", alpha=0.8
)

plt.axis("equal")
plt.title("Goal-Point Monte-Carlo Mean Normals")
plt.xlabel("x")
plt.ylabel("y")
plt.tight_layout()
plt.show()
