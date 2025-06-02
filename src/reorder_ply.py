import os
import argparse
import numpy as np
from plyfile import PlyData, PlyElement

def reorder(ply, savepath):
    vertices = PlyData.read(ply)['vertex'].data
    reorder_vertices = vertices.copy()

    reorder_vertices['x'] = vertices['y']
    reorder_vertices['y'] = -1 * vertices['z']
    reorder_vertices['z'] = -1 * vertices['x']

    ply_elem = PlyElement.describe(reorder_vertices, "vertex")
    savepath = os.path.join(savepath, "reorder.ply")
    PlyData([ply_elem], text=True).write(savepath)
    return

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data", required=True, help="original pointcloud")
    parser.add_argument("-l", "--save_loc", required=False, help="directory to save reorder to", default=os.getcwd())
    args = parser.parse_args()

    os.makedirs(args.save_loc, exist_ok=True)
    reorder(args.data, args.save_loc)

    print(f"saved to {args.save_loc}reorder.ply")