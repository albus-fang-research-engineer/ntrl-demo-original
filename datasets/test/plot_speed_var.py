import numpy as np
import matplotlib.pyplot as plt

path = "./maze"

pts   = np.load(f"{path}/sampled_points.npy")
speed = np.load(f"{path}/speed_var.npy")

print("points:", pts.shape)
print("speed:", speed.shape)



x_start = pts[:,0]
y_start = pts[:,1]
x_goal = pts[:,2]
y_goal = pts[:,3]

plt.figure(figsize=(6,5))

# --- very transparent background point cloud ---
# plt.scatter(x, y, s=0.01, color="black", alpha=0.9)

# --- speed field on top ---
sc = plt.scatter(x_start, y_start, c=speed[:,0], s=0.5, cmap="viridis")
plt.colorbar(sc, label="speed (start)")

plt.title("Variance of Speed Field + Sampled Points")
plt.axis("equal")
plt.tight_layout()
plt.show()

plt.figure(figsize=(6,5))

# --- very transparent background point cloud ---
# plt.scatter(x, y, s=0.01, color="black", alpha=0.9)

# --- speed field on top ---
sc = plt.scatter(x_goal, y_goal, c=speed[:,1], s=0.5, cmap="viridis")
plt.colorbar(sc, label="speed (goal)")

plt.title("Speed Field + Sampled Points")
plt.axis("equal")
plt.tight_layout()
plt.show()