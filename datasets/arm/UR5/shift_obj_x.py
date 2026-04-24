import sys

def center_x(input_file, output_file):
    with open(input_file, 'r') as f:
        lines = f.readlines()

    # First pass: find X bounding box
    x_values = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith('v '):
            parts = stripped.split()
            x_values.append(float(parts[1]))

    if not x_values:
        print("No vertices found.")
        return

    x_min = min(x_values)
    x_max = max(x_values)
    x_center = (x_min + x_max) / 2
    offset = -x_center

    print(f"X range: [{x_min}, {x_max}]")
    print(f"Bounding box center X: {x_center}")
    print(f"Applying offset: {offset}")

    # Second pass: write with shifted X
    with open(output_file, 'w') as out:
        for line in lines:
            stripped = line.strip()
            if stripped.startswith('v '):
                parts = stripped.split()
                x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                x += offset
                out.write(f'v {x} {y} {z}\n')
            else:
                out.write(line)

    print(f"Done. Written to '{output_file}'")

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: python center_x.py <input.obj> <output.obj>")
        sys.exit(1)
    center_x(sys.argv[1], sys.argv[2])