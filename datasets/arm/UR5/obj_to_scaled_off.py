import sys


def read_obj_vertices(input_file):
    vertices = []
    with open(input_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('v '):
                parts = line.split()
                vertices.append((float(parts[1]), float(parts[2]), float(parts[3])))
    return vertices


def write_off_vertices_only(output_file, vertices):
    with open(output_file, 'w') as out:
        out.write('OFF\n')
        out.write(f'{len(vertices)} 0 0\n')
        for vert in vertices:
            out.write(f'{vert[0]} {vert[1]} {vert[2]}\n')


def convert(input_file, output_file):
    vertices = read_obj_vertices(input_file)
    print(f"Read {len(vertices)} vertices from '{input_file}'")
    write_off_vertices_only(output_file, vertices)
    print(f"Written to '{output_file}'")


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: python obj_to_off.py <input.obj> <output.off>")
        sys.exit(1)

    convert(sys.argv[1], sys.argv[2])