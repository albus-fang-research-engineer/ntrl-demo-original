import sys


def shift_y(input_file, output_file, offset=-0.2):
    with open(input_file, 'r') as f:
        lines = f.readlines()

    with open(output_file, 'w') as out:
        for line in lines:
            stripped = line.strip()
            if stripped.startswith('v '):
                parts = stripped.split()
                x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                y += offset
                out.write(f'v {x} {y} {z}\n')
            else:
                out.write(line)

    print(f"Done. Y values shifted by {offset}. Written to '{output_file}'")


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: python shift_y.py <input.obj> <output.obj>")
        sys.exit(1)

    shift_y(sys.argv[1], sys.argv[2])