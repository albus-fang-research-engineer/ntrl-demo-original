import numpy as np
import matplotlib.pyplot as plt

# load the file
points = np.load("sampled_points.npy")     # <-- change filename

print(points.shape)                # sanity-check (should be N x 2)

# plot
plt.figure(figsize=(6,6))
plt.scatter(points[:,0], points[:,1], s=0.05, color='blue')
plt.scatter(points[:,2], points[:,3], s=0.05, color='red')
plt.axis('equal')
plt.title("Sampled 2D Points")
plt.xlabel("x")
plt.ylabel("y")
plt.savefig("sampled points.png")
plt.show()
