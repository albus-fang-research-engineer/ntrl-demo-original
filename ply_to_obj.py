import sys
import os
from plyfile import PlyData

def convert_ply_to_obj(input_path, output_path):
    plydata = PlyData.read(input_path)
    vertex = plydata['vertex']
    
    with open(output_path, 'w') as f:
        for p in zip(vertex['x'], vertex['y'], vertex['z']):
            f.write(f"v {p[0]} {p[1]} {p[2]}\n")
    
    print(f"Saved {len(vertex)} points to: {output_path}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python convert.py input.ply output.obj")
        sys.exit(1)
    input_file, output_file = sys.argv[1], sys.argv[2]
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"{input_file} not found")
    convert_ply_to_obj(input_file, output_file)