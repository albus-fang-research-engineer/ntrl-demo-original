import numpy as np
import matplotlib.pyplot as plt

# Load obstacle point cloud
pts = np.load("maze/obstacle_points.npy")

print("Obstacle points shape:", pts.shape)

# 2D or 3D?
if pts.shape[1] == 3:
    x = pts[:, 0]
    y = pts[:, 1]
    z = pts[:, 2]
else:
    x = pts[:, 0]
    y = pts[:, 1]

plt.figure(figsize=(6,6))
plt.scatter(x, y, s=0.2, alpha=0.5)
plt.axis("equal")
plt.title("Obstacle Surface Point Cloud")
plt.xlabel("x")
plt.ylabel("y")
plt.tight_layout()
plt.show()
