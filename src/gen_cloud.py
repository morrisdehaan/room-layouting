from PIL import Image
import numpy as np
import json
import matplotlib.pyplot as plt
from collections import Counter
from typing import Dict, Any, Tuple, List, Callable, Optional
import argparse
from plyfile import PlyData, PlyElement

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Generate point cloud script")
    # input data adirectory
    parser.add_argument(
        "-i",
        "--data_dir",
        type=str,
        default="../res/scans",
    )
    # directory where output is stored
    parser.add_argument(
        "-o",
        "--out_path",
        type=str,
        default="../res/out/room.ply"
    )
    # which views to consider
    parser.add_argument(
        "-v",
        "--views",
        type=int,
        nargs="*", # take start, end (exclusive), and offset value,
        default=None,
        help="Please provide a start, end (exclusive), and offset value"
    )
    # sample rate of each depth map
    parser.add_argument(
        "-dsr",
        "--depth_sample_rate",
        type=int,
        default=16
    )
    # whether to add color to points
    parser.add_argument(
        "-c",
        "--add_rgb",
        action="store_true",
        help="Use this flag to project RGB values to each point"
    )

    args = parser.parse_args()

    if args.views is not None:
        if len(args.views) != 3:
            parser.error("Argument '--views' requires exactly 3 integers!")
        elif args.views[2] <= 0:
            parser.error("Argument '--views' offset value should be at least 1!")

    return args

def load_views(views_path: str, view_range: Optional[List[int]]) -> List:
    """
    Loads camera information per view punt (e.g., camera in- and extrinsics).

    # Params
    * `view_range`: A start, end (exclusive), and offset value that determine which views are loaded
    """

    views = []
    with open(views_path) as f:
        for i, line in enumerate(f):
            if view_range is not None:
                if i < view_range[0] or i % view_range[2] != 0:
                    continue
                if i >= view_range[1]:
                    break

            views.append(
                json.loads(line.strip())
            )
    return views

def load_rgbd(depth_path: str, col_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Loads depth and color images from given paths.

    # Returns
    Returns a depth and color image.
    """
    # load depth
    depth_img = Image.open(depth_path)
    # convert from mm to meters
    depth_img = np.array(depth_img).astype(np.float32) / 1000.0

    # load color
    col_img = Image.open(col_path).convert("RGB")
    col_img = np.array(col_img)
    
    return depth_img, col_img

def depth_to_points(
        uv_coords: np.ndarray, depths: np.ndarray, K: np.ndarray, E: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Recover 3D world space coordinates from a depth-map ignoring 0 depth values.
    
    # Params
    * `uv_coords`: of shape [N x 2]
    * `depths`: depth values matching `uv_coords` of shape [N]
    * `K`: intrinsic camera matrix of shape [3 x 3]
    * `E`: extrinsic camera matrix of shape [4 x 4]
    # Returns
    An array of shape [M x 3] denoting 3D coordinates and corresponding uv-coordinates of shape [M x 2] in the color image.
    """

    # ignore 0 depth values
    nonzero_inds = np.where(depths != 0.)[0]
    
    N = len(nonzero_inds)
    if N == 0:
        return np.zeros((0, 3))

    # convert to homogeneous coordinates
    points = (
        np.concatenate(
            (uv_coords[nonzero_inds], np.ones((N, 1))), axis=1
        ) * depths[nonzero_inds].reshape(-1, 1)
    ).T

    # undo intrinsic camera transformation, then camera rotation, then camera translation
    points = np.linalg.inv(K) @ points
    points = np.concatenate((points, np.ones((1, N))), axis=0)
    points = (np.linalg.inv(E) @ points).T[:, :3]

    return points, uv_coords[nonzero_inds]

def rgbd_to_points(
        depth_img: np.ndarray, col_img: np.ndarray, K: np.ndarray, E: np.ndarray,
        sample_rate=16,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Recover 3D world space coordinates from a depth and color image ignoring 0 depth values.
    
    # Params
    * `depth_img`: of shape [Dh x Dw]
    * `col_img`: of shape [Ch x Cw x 3]
    * `K`: intrinsic camera matrix of shape [3 x 3]
    * `E`: extrinsic camera matrix of shape [4 x 4]
    * `sample_rate`: downsampling rate for the depth image
    # Returns
    An array of shape [M x 3] denoting 3D coordinates and corresponding uv-coordinates of shape [M x 2] in the color image.
    """    
    # uv-map that accounts for the fact that depth and color images are of different sizes
    xs, ys = np.meshgrid(
        np.linspace(0.5, col_img.shape[1]-0.5, depth_img.shape[1]),
        np.linspace(0.5, col_img.shape[0]-0.5, depth_img.shape[0]),
    )
    uv_grid = np.stack((xs, ys), axis=2)

    return depth_to_points(
        uv_grid[::sample_rate, ::sample_rate].reshape(-1, 2),
        depth_img[::sample_rate, ::sample_rate].reshape(-1),
        K,
        E,
    )

def views_info_to_points(
    views: List[Dict[str, Any]], file_dir: str, sample_rate=16, add_rgb=False
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Recover 3D world space coordinates from a list of view meta-data.

    # Params
    * `views`: list of dictionaries containing view meta-data
    * `file_dir`: directory containing the depth and color images
    * `sample_rate`: downsampling rate for the depth image

    # Returns
    An array of shape [M x 3] denoting 3D coordinates, corresponding uv-coordinates of shape [M x 2] in the color image
    and the number of samples for each image of shape [I].
    """

    if add_rgb:
        points = np.array([]).reshape(0, 6)
    else:
        points = np.array([]).reshape(0, 3)

    uv_coords = np.array([]).reshape(0, 2)
    num_samples = np.zeros(len(views)).astype(np.int32)
    
    for i, view in enumerate(views):
        # load images
        depth_img, col_img = load_rgbd(
            f"{file_dir}/{view["depth_file_name"]}",
            f"{file_dir}/{view["file_name"]}",
        )

        # load point cloud for view
        cloud, uvs = rgbd_to_points(
            depth_img, col_img, np.array(view["K"]), np.array(view["viewmat"]), sample_rate
        )
        if add_rgb:
            # use nearest neighbor interpolation
            uvs = (uvs + 0.5).astype(int)
            cloud_cols = col_img[uvs[:, 1], uvs[:, 0]]
            cloud = np.concatenate((cloud, cloud_cols), axis=1)

        num_samples[i] = cloud.shape[0]
        points = np.concatenate((points, cloud), axis=0)
        uv_coords = np.concatenate((uv_coords, uvs), axis=0)
    return points, uv_coords, num_samples

if __name__ == "__main__":
    parser = parse_args()

    print("Args:", parser)

    # load camera in- and extrinsics for different view points
    views = load_views(f"{parser.data_dir}/views.json", parser.views)

    # load point cloud from selected views
    cloud, _, _ = views_info_to_points(
        views,
        parser.data_dir,
        parser.depth_sample_rate,
        parser.add_rgb,
    )

    print(f"Generated cloud with {cloud.shape[0]} points of dimensionality {cloud.shape[1]}!")

    # save results
    # first, make sure z-axis is up (necessary for SpatialLm) by shifting coordinates
    ply_points = cloud
    ply_points[:, :3] = np.roll(cloud[:, :3], 1, axis=1)

    # second, convert points to the correct datatype
    if parser.add_rgb:
        ply_points = np.asarray(list(map(tuple, ply_points)), dtype=[
            ("x", "f4"), ("y", "f4"), ("z", "f4"),
            ("red", np.uint8), ("green", np.uint8), ("blue", np.uint8),
        ])
    else:
        ply_points = np.asarray(list(map(tuple, ply_points)), dtype=[("x", "f4"), ("y", "f4"), ("z", "f4")])
    # third, save
    ply_el = PlyElement.describe(
        ply_points,
        "vertex"
    )
    PlyData([ply_el]).write(parser.out_path)

    # TODO: remove or refactor
    # if parser.add_rgb:
    #     json.dump([{
    #             "pos": [float(v[0]), float(v[1]), float(v[2])],
    #             "col": [float(v[3]) / 255, float(v[4]) / 255, float(v[5]) / 255],
    #             "size": 0.05
    #         } for v in ply_points],
    #         open(f"../res/out/pretty_point_cloud.json", "w"),
    #     )
    
