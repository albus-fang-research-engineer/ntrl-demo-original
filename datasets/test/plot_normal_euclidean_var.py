import numpy as np
import matplotlib.pyplot as plt

path = "./maze"

pts  = np.load(f"{path}/sampled_points.npy")
var  = np.load(f"{path}/normal_euclidean_var.npy")

x0 = pts[:,0]
y0 = pts[:,1]
x1 = pts[:,2]
y1 = pts[:,3]

vx0 = var[:,0]
vy0 = var[:,1]
vx1 = var[:,2]
vy1 = var[:,3]

# -------------------------------
# Start point: Var(nx)
# -------------------------------
plt.figure(figsize=(6,5))
sc = plt.scatter(x0, y0, c=vx0, s=0.5, cmap="viridis")
plt.colorbar(sc, label="Var(nx)")
plt.title("Start: Var(nx)")
plt.axis("equal")
plt.tight_layout()
plt.show()

# -------------------------------
# Start point: Var(ny)
# -------------------------------
plt.figure(figsize=(6,5))
sc = plt.scatter(x0, y0, c=vy0, s=0.5, cmap="viridis")
plt.colorbar(sc, label="Var(ny)")
plt.title("Start: Var(ny)")
plt.axis("equal")
plt.tight_layout()
plt.show()

# -------------------------------
# Goal: Var(nx)
# -------------------------------
plt.figure(figsize=(6,5))
sc = plt.scatter(x1, y1, c=vx1, s=0.5, cmap="viridis")
plt.colorbar(sc, label="Var(nx)")
plt.title("Goal: Var(nx)")
plt.axis("equal")
plt.tight_layout()
plt.show()

# -------------------------------
# Goal: Var(ny)
# -------------------------------
plt.figure(figsize=(6,5))
sc = plt.scatter(x1, y1, c=vy1, s=0.5, cmap="viridis")
plt.colorbar(sc, label="Var(ny)")
plt.title("Goal: Var(ny)")
plt.axis("equal")
plt.tight_layout()
plt.show()
