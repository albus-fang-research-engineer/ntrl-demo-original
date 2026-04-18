import numpy as np
import sys

def npy_to_obj(input_path, output_path=None):
    if output_path is None:
        output_path = input_path.replace(".npy", ".obj")

    points = np.load(input_path)

    # Handle (N, 3) or (N, 2) arrays
    if points.ndim != 2 or points.shape[1] not in (2, 3):
        raise ValueError(f"Expected array of shape (N, 2) or (N, 3), got {points.shape}")

    with open(output_path, "w") as f:
        for point in points:
            if len(point) == 2:
                f.write(f"v {point[0]} {point[1]} 0.0\n")
            else:
                f.write(f"v {point[0]} {point[1]} {point[2]}\n")

    print(f"Wrote {len(points)} vertices to {output_path}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python npy_to_obj.py <input.npy> [output.obj]")
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else None
    npy_to_obj(input_path, output_path)