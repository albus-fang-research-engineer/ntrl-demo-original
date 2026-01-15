import numpy as np
import matplotlib.pyplot as plt

path = "./maze"

pts   = np.load(f"{path}/sampled_points.npy")
normal_var = np.load(f"{path}/normal_angular_var.npy")

print("points:", pts.shape)
print("normal_var:", normal_var.shape)

x_start = pts[:,0]
y_start = pts[:,1]
x_goal  = pts[:,2]
y_goal  = pts[:,3]

# -------------------------------
# Start point normal variance
# -------------------------------
plt.figure(figsize=(6,5))

sc = plt.scatter(
    x_start, y_start,
    c=normal_var[:,0],
    s=0.5,
    cmap="viridis"
)

plt.colorbar(sc, label="Normal angular variance (start)")
plt.title("Normal Angular Variance Field (Start Points)")
plt.axis("equal")
plt.tight_layout()
plt.show()

# -------------------------------
# Goal point normal variance
# -------------------------------
plt.figure(figsize=(6,5))

sc = plt.scatter(
    x_goal, y_goal,
    c=normal_var[:,1],
    s=0.5,
    cmap="viridis"
)

plt.colorbar(sc, label="Normal angular variance (goal)")
plt.title("Normal Angular Variance Field (Goal Points)")
plt.axis("equal")
plt.tight_layout()
plt.show()
