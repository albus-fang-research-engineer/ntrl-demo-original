#!/usr/bin/env python3

import sys
import os


def shift_off_y(input_path, output_path, y_offset=-0.2):
    with open(input_path, "r") as f:
        lines = f.readlines()

    if not lines:
        raise ValueError("Input file is empty.")

    if lines[0].strip() != "OFF":
        raise ValueError("Input file does not start with OFF header.")

    if len(lines) < 2:
        raise ValueError("OFF file is missing the counts line.")

    counts = lines[1].strip().split()
    if len(counts) < 3:
        raise ValueError("Invalid OFF counts line.")

    num_vertices = int(counts[0])
    num_faces = int(counts[1])
    num_edges = int(counts[2])

    output_lines = []
    output_lines.append(lines[0])  # OFF
    output_lines.append(lines[1])  # counts line

    # Process vertex lines
    for i in range(2, 2 + num_vertices):
        parts = lines[i].strip().split()
        if len(parts) < 3:
            raise ValueError(f"Invalid vertex line at line {i+1}: {lines[i].strip()}")

        x = float(parts[0])
        y = float(parts[1]) + y_offset
        z = float(parts[2])

        # Preserve any extra columns after xyz
        extras = parts[3:]
        new_line = f"{x} {y} {z}"
        if extras:
            new_line += " " + " ".join(extras)
        output_lines.append(new_line + "\n")

    # Copy the rest unchanged
    for i in range(2 + num_vertices, len(lines)):
        output_lines.append(lines[i])

    with open(output_path, "w") as f:
        f.writelines(output_lines)

    print(f"Saved shifted OFF file to: {output_path}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python shift_off_y.py input.off output.off")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]

    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file not found: {input_file}")

    shift_off_y(input_file, output_file, y_offset=-0.2)