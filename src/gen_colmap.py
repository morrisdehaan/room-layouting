import os
import json
import time
import hashlib
import argparse
import numpy as np
from plyfile import PlyData
from scipy.spatial.transform import Rotation as R

def views_data(dataset, target, start=None, end=None):
    views = open(os.path.join(dataset, "views.json")).readlines()[start:end]

    cameras_path = os.path.join(target, "sparse", "cameras.txt")
    images_path = os.path.join(target, "sparse", "images.txt")

    with open(cameras_path, 'w') as cam_txt, open(images_path, 'w') as img_txt:
        for i, view in enumerate(views, start=1):
            v = json.loads(view)

            camera_line = f"{i} PINHOLE {v['width']} {v['height']} {v['fx']} {v['fy']} {v['cx']} {v['cy']}" 
            cam_txt.write(camera_line + "\n")

            fname = v['name'] + '.jpg'
            extrinsic = np.array(v["viewmat"])
            rotation = extrinsic[:-1, :-1]
            tx, ty, tz = extrinsic[:-1, -1]
            qx, qy, qz, qw = R.from_matrix(rotation).as_quat()
            image_line = f"{i} {qw} {qx} {qy} {qz} {tx} {ty} {tz} {i} {fname}\n0 0 -1"
            # 0 0 -1 = no keypoint matches to 3d points
            img_txt.write(image_line + "\n")
    return

def ply_data(plyfile, target, reorder_dims=False):
    p3d_path = os.path.join(target, "sparse", "points3D.txt")

    vertices = PlyData.read(plyfile)['vertex'].data
    with open(p3d_path, 'w') as p3d_file:
        # no rgb
        if (len(vertices[0]) == 3):
            for i,(x,y,z) in enumerate(vertices, start=1):
                if reorder_dims:
                    y,z = z,y
                p3d_line = f"{i} {x} {y} {z} 255 255 255 0.0"
                p3d_file.write(p3d_line + "\n")
        # rgb
        elif (len(vertices[0]) == 6):
            for i,(x,y,z,r,g,b) in enumerate(vertices, start=1):
                if reorder_dims:
                    y,z = z,y
                p3d_line = f"{i} {x} {y} {z} {r} {g} {b} 0.0"
                p3d_file.write(p3d_line + "\n")
    return

def copy_images(dataset, target, start=None, end=None):
    images = sorted(os.listdir(os.path.join(dataset, "images")))
    depths = sorted(os.listdir(os.path.join(dataset, "depth")))

    img_loc = os.path.join(target, "images")
    dpt_loc = os.path.join(target, "depth")

    for i in range(start, end):
        img_path = os.path.join(dataset, "images", images[i])
        dpt_path = os.path.join(dataset, "depth", depths[i])
        # hmm
        os.system(f"cp {img_path} {img_loc}")
        os.system(f"cp {dpt_path} {dpt_loc}")
    return
        

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", required=True, help="dataset location, contains: depth/; images/; views.json")
    parser.add_argument("-p", "--pointcloud", required=True, help="pointcloud in .ply format")
    parser.add_argument("-s", "--start", type=int, required=False, help="create colmap starting at image N", default=None)
    parser.add_argument("-e", "--end", type=int, required=False, help="create colmap ending at image N (exclusive)", default=None)
    parser.add_argument("-l", "--save_location", required=False, help="location to save colmap", default=os.getcwd())

    args = parser.parse_args()

    H = "colmap_" + hashlib.md5(f"{time.time()}".encode()).hexdigest()[:8]
    unique_save = os.path.join(args.save_location, H)

    for dir_ in ["sparse", "images", "depth"]:
        os.makedirs(os.path.join(unique_save, dir_), exist_ok=False)

    views_data(args.dataset, unique_save, args.start, args.end)
    ply_data(args.pointcloud, unique_save)
    copy_images(args.dataset, unique_save, args.start, args.end)

    print(f"saved to {unique_save}")
