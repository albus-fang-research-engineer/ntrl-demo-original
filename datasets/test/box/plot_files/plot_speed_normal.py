import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------
# ----------- Load the dataset ----------
# ---------------------------------------

path = "datasets/test/box"
path = "."

pts   = np.load(f"{path}/sampled_points.npy")
speed = np.load(f"{path}/speed.npy")
normal = np.load(f"{path}/normal.npy")

print("points:", pts.shape)
print("speed:", speed.shape)
print("normal:", normal.shape)

x = pts[:,0]
y = pts[:,1]

nx = normal[:,0]
ny = normal[:,1]

# ---------------------------------------
# -----   Visualize Speed (start)   -----
# ---------------------------------------

plt.figure(figsize=(6,5))
sc = plt.scatter(x, y, c=speed[:,0], s=3, cmap="viridis")
plt.colorbar(sc, label="speed (start)")
plt.title("Speed Field (start point)")
plt.axis("equal")
plt.show()

# ---------------------------------------
# -------- Visualize Normals -----------
# ---------------------------------------

plt.figure(figsize=(6,6))
plt.quiver(x, y, nx, ny, angles="xy", scale_units="xy", scale=10)
plt.title("Normal Vectors")
plt.axis("equal")
plt.show()

# ---------------------------------------
# ---- Overlay: Speed + Normal Arrows ---
# ---------------------------------------

plt.figure(figsize=(6,6))
sc = plt.scatter(x, y, c=speed[:,0], s=3, cmap="viridis")
plt.quiver(x, y, nx, ny, color="red", scale=20)
plt.colorbar(sc, label="speed (start)")
plt.title("Speed + Normals")
plt.axis("equal")
plt.show()
