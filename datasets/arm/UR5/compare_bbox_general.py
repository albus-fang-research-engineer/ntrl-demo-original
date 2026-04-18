import sys
import os
import math


def parse_obj(path):
    vertices = []
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('v '):
                parts = line.split()
                if len(parts) >= 4:
                    try:
                        vertices.append((float(parts[1]), float(parts[2]), float(parts[3])))
                    except ValueError:
                        pass
    return vertices


def parse_off(path):
    vertices = []
    with open(path, 'r') as f:
        lines = [l.strip() for l in f if l.strip() and not l.startswith('#')]

    i = 0
    # Handle OFF header
    if lines[i].upper().startswith('OFF'):
        parts = lines[i].split()
        if len(parts) == 4:          # "OFF 9055 0 0" — counts on same line
            n_verts = int(parts[1])
            i += 1
        else:                        # bare "OFF" — counts on next line
            i += 1
            n_verts = int(lines[i].split()[0])
            i += 1
    else:                            # no OFF keyword, straight to counts
        n_verts = int(lines[i].split()[0])
        i += 1

    # Read exactly n_verts vertex lines
    for _ in range(n_verts):
        parts = lines[i].split()
        vertices.append((float(parts[0]), float(parts[1]), float(parts[2])))
        i += 1

    return vertices


def parse_file(path):
    ext = os.path.splitext(path)[1].lower()
    if ext == '.off':
        return parse_off(path)
    elif ext == '.obj':
        return parse_obj(path)
    else:
        raise ValueError(f"Unsupported file type: {ext}")


def bounding_box(vertices):
    if not vertices:
        raise ValueError("No vertices found.")
    xs, ys, zs = zip(*vertices)
    return {
        'min':    (min(xs),           min(ys),           min(zs)),
        'max':    (max(xs),           max(ys),           max(zs)),
        'size':   (max(xs) - min(xs), max(ys) - min(ys), max(zs) - min(zs)),
        'center': ((max(xs) + min(xs)) / 2,
                   (max(ys) + min(ys)) / 2,
                   (max(zs) + min(zs)) / 2),
    }


def print_bbox(label, bb, n_verts):
    print(f"\n{'='*50}")
    print(f"  {label}  ({n_verts} vertices)")
    print(f"{'='*50}")
    print(f"  Min   : ({bb['min'][0]:+.4f}, {bb['min'][1]:+.4f}, {bb['min'][2]:+.4f})")
    print(f"  Max   : ({bb['max'][0]:+.4f}, {bb['max'][1]:+.4f}, {bb['max'][2]:+.4f})")
    print(f"  Size  : ({bb['size'][0]:.4f},  {bb['size'][1]:.4f},  {bb['size'][2]:.4f})")
    print(f"  Center: ({bb['center'][0]:+.4f}, {bb['center'][1]:+.4f}, {bb['center'][2]:+.4f})")


def compare_bboxes(bb1, bb2):
    print(f"\n{'='*50}")
    print("  COMPARISON  (file2 - file1)")
    print(f"{'='*50}")
    for i, ax in enumerate(['X', 'Y', 'Z']):
        print(f"  {ax}  size diff  : {bb2['size'][i]   - bb1['size'][i]:+.4f}")
        print(f"  {ax}  center diff: {bb2['center'][i] - bb1['center'][i]:+.4f}")

    diag1 = math.sqrt(sum(s**2 for s in bb1['size']))
    diag2 = math.sqrt(sum(s**2 for s in bb2['size']))
    print(f"\n  Diagonal (file1): {diag1:.4f}")
    print(f"  Diagonal (file2): {diag2:.4f}")
    print(f"  Diagonal ratio  : {diag2/diag1:.4f}  (file2 / file1)")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python compare_bbox.py file1 file2")
        sys.exit(1)

    paths = sys.argv[1:]
    bbs = []
    for path in paths:
        verts = parse_file(path)
        bb = bounding_box(verts)
        print_bbox(os.path.basename(path), bb, len(verts))
        bbs.append(bb)

    compare_bboxes(bbs[0], bbs[1])