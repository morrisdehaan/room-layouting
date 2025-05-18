import os
import json
import argparse
import numpy as np
from plyfile import PlyData
from scipy.spatial.transform import Rotation as R

def views_data(dataset, target, start=None, end=None):
    views = open(os.path.join(dataset, "views.json")).readlines()[start:end]

    cameras_path = os.path.join(target, "colmap", "sparse", "cameras.txt")
    images_path = os.path.join(target, "colmap", "sparse", "images.txt")

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
            # \n0 0 -1 = no keypoint matches to 3d points
            img_txt.write(image_line + "\n")

def ply_data(plyfile, target, start=None, end=None):
    p3d_path = os.path.join(target, "colmap", "sparse", "points3D.txt")
    vertices = PlyData.read(plyfile)['vertex'].data [start:end]
    with open(p3d_path, 'w') as p3d_file:
        for i,(x,y,z) in enumerate(vertices, start=1):
            p3d_line = f"{i} {x} {y} {z} 255 255 255 0.0"
            p3d_file.write(p3d_line + "\n")


def copy_images(dataset, target, start=None, end=None):
    images = sorted(os.listdir(os.path.join(dataset, "images")))
    copy_loc = os.path.join(target, "colmap", "images")
    if os.listdir(copy_loc):
        os.system(f"rm {copy_loc}/*")

    for img in images[start:end]:
        img_path = os.path.join(dataset, "images", img)
        os.system(f"cp {img_path} {copy_loc}")
        

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", required=True, help="dataset location, contains: depth/; images/; views.json")
    parser.add_argument("-p", "--pointcloud", required=True, help="pointcloud in .ply format")
    parser.add_argument("-s", "--start", type=int, required=False, help="create colmap starting at image N", default=None)
    parser.add_argument("-e", "--end", type=int, required=False, help="create colmap ending at image N", default=None)
    parser.add_argument("-l", "--save_location", required=False, help="location to save colmap", default=os.getcwd())

    args = parser.parse_args()

    os.makedirs(os.path.join(args.save_location, "colmap", "sparse"), exist_ok=True)
    os.makedirs(os.path.join(args.save_location, "colmap", "images"), exist_ok=True)

    views_data(args.dataset, args.save_location, args.start, args.end)
    ply_data(args.pointcloud, args.save_location, args.start, args.end)
    copy_images(args.dataset, args.save_location, args.start, args.end)
