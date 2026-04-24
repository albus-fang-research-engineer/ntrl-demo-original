import sys


def shift_y_off(input_file, output_file, offset=-0.2):
    with open(input_file, 'r') as f:
        lines = f.readlines()

    if not lines:
        raise ValueError("Empty file.")

    # --- Parse header ---
    idx = 0

    # Skip blank/comment lines before OFF
    while lines[idx].strip() == "" or lines[idx].strip().startswith("#"):
        idx += 1

    if not lines[idx].strip().startswith("OFF"):
        raise ValueError("Not a valid OFF file (missing OFF header).")

    header_idx = idx
    idx += 1

    # Skip comments/blank lines before counts
    while lines[idx].strip() == "" or lines[idx].strip().startswith("#"):
        idx += 1

    counts_idx = idx
    parts = lines[counts_idx].strip().split()
    if len(parts) < 3:
        raise ValueError("Invalid OFF counts line.")

    num_vertices = int(parts[0])
    num_faces = int(parts[1])
    # num_edges = int(parts[2])  # not needed

    idx += 1
    vertex_start_idx = idx

    # --- Write output ---
    with open(output_file, 'w') as out:
        # Write everything up to vertex list unchanged
        for i in range(vertex_start_idx):
            out.write(lines[i])

        # --- Process vertices ---
        for i in range(num_vertices):
            line = lines[vertex_start_idx + i].strip()

            if line == "" or line.startswith("#"):
                out.write(lines[vertex_start_idx + i])
                continue

            parts = line.split()
            if len(parts) < 3:
                raise ValueError(f"Invalid vertex line at index {i}")

            x = float(parts[0])
            y = float(parts[1]) + offset
            z = float(parts[2])

            # Preserve extra attributes if present (e.g., colors)
            rest = parts[3:]
            new_line = f"{x} {y} {z}"
            if rest:
                new_line += " " + " ".join(rest)

            out.write(new_line + "\n")

        # --- Write faces and rest unchanged ---
        remaining_start = vertex_start_idx + num_vertices
        for i in range(remaining_start, len(lines)):
            out.write(lines[i])

    print(f"Done. Shifted Y by {offset}. Output: '{output_file}'")


if __name__ == '__main__':
    if len(sys.argv) not in [3, 4]:
        print("Usage: python shift_y_off.py <input.off> <output.off> [offset]")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]
    offset = float(sys.argv[3]) if len(sys.argv) == 4 else -0.2

    shift_y_off(input_file, output_file, offset)