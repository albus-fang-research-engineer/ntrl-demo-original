import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

d = np.load("lower_half_debug_epoch500.npz")

X = d["X"]
Y = d["Y"]
S = d["Speed"]
T = d["Tau"]

print("Speed min/max:", S.min(), S.max())
print("Tau min/max:", T.min(), T.max())

# ---- Scatter plot of raw evaluated points ----
plt.figure(figsize=(6,5))

sc = plt.scatter(
    X, Y,
    c=S,
    cmap="plasma",
    norm=LogNorm(vmin=S.min(), vmax=S.max()),
    s=500,
    alpha=0.9
)

plt.colorbar(sc, label="Speed (log scale)")
plt.title("Raw NN speed samples (lower half)")
plt.axis("equal")
plt.xlabel("x")
plt.ylabel("y")
plt.tight_layout()
plt.show()
