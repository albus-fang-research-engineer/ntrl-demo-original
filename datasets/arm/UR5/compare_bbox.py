import sys

def parse_obj_vertices(path):
    vertices = []
    with open(path, 'r') as f:
        for line in f:
            if line.startswith('v '):
                _, x, y, z = line.split()
                vertices.append((float(x), float(y), float(z)))
    return vertices

def bounding_box(vertices):
    xs, ys, zs = zip(*vertices)
    return {
        'min': (min(xs), min(ys), min(zs)),
        'max': (max(xs), max(ys), max(zs)),
        'size': (max(xs)-min(xs), max(ys)-min(ys), max(zs)-min(zs))
    }

def print_bbox(label, bb):
    print(f"\n{label}")
    print(f"  Min : ({bb['min'][0]:.4f}, {bb['min'][1]:.4f}, {bb['min'][2]:.4f})")
    print(f"  Max : ({bb['max'][0]:.4f}, {bb['max'][1]:.4f}, {bb['max'][2]:.4f})")
    print(f"  Size: ({bb['size'][0]:.4f}, {bb['size'][1]:.4f}, {bb['size'][2]:.4f})")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python compare_bbox.py file1.obj file2.obj")
        sys.exit(1)

    for path in sys.argv[1:]:
        verts = parse_obj_vertices(path)
        bb = bounding_box(verts)
        print_bbox(path, bb)