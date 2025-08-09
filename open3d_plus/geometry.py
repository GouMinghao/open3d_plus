import open3d as o3d
import numpy as np
from PIL import Image
from typing import Union


def array2pcd(points: np.ndarray, colors: Union[None, np.ndarray] = None):
    """
    Convert points and colors into open3d point cloud.

    Args:
        points(np.ndarray): coordinates of the points.
        colors(np.ndarray): RGB values of the points.

    Returns:
        open3d.geometry.PointCloud: the point cloud.
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    if colors is not None:
        pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd


def pcd2array(pcd):
    """
    Convert open3d point cloud into points and colors.

    Args:
        pcd(open3d.geometry.PointCloud): the point cloud.

    Returns:
        np.array, np.array: coordinates of the points, RGB values of the points.
    """
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)
    return points, colors


def merge_pcds(pcds, merge_color: bool = True):
    """
    Merge several point cloud into a single one.

    Args:
        pcds(list): list of point cloud.
        merge_color(bool): if merge color

    Returns:
        open3d.geometry.PointCloud: the merged point cloud.
    """
    ret_pcd = o3d.geometry.PointCloud()
    if len(pcds) == 0:
        return ret_pcd
    points_list = []
    if merge_color:
        colors_list = []
    for pcd in pcds:
        points = np.asarray(pcd.points)
        points_list.append(points)
        if merge_color:
            colors = np.asarray(pcd.colors)
            colors_list.append(colors)
    merge_points = np.vstack(points_list)
    if merge_color:
        merge_colors = np.vstack(colors_list)
        return array2pcd(merge_points, merge_colors)
    else:
        return array2pcd(merge_points)


def generate_scene_pointcloud(depth, rgb, intrinsics, depth_scale, use_mask=True):
    """Generate point cloud from depth image and color image

    Args:
        depth(str / np.array): Depth image path or depth.
        rgb(str / np.array): RGB image path or RGB values.
        intrinsics(np.array): Camera intrinsics matrix.
        depth_scale(float): The depth factor.
        use_mask(bool): Whether to use mask for pointcloud.

    Returns:
        open3d.geometry.PointCloud: the point cloud
    """
    if type(depth) == str and type(rgb) == str:
        colors = np.array(Image.open(rgb), dtype=np.float32) / 255.0
        depths = np.array(Image.open(depth))

    elif type(depth) == np.ndarray and type(rgb) == np.ndarray:
        colors = rgb
        depths = depth

    else:
        raise ValueError("The type of depth and rgb must be str or np.ndarray")

    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]

    xmap, ymap = np.arange(colors.shape[1]), np.arange(colors.shape[0])
    xmap, ymap = np.meshgrid(xmap, ymap)

    points_z = depths / depth_scale
    points_x = (xmap - cx) / fx * points_z
    points_y = (ymap - cy) / fy * points_z

    mask = points_z > 0
    points = np.stack([points_x, points_y, points_z], axis=-1)
    if use_mask:
        points = points[mask]
        colors = colors[mask]
    else:
        points = points.reshape((-1, 3))
        colors = colors.reshape((-1, 3))
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points)
    cloud.colors = o3d.utility.Vector3dVector(colors)
    return cloud


def create_axis(length: float, color=[1, 0, 0], rot=[0, 0, 1], translation=[0, 0, 0]):
    cylinder_radius_ratio = 0.01
    cone_radius_ratio = 0.02
    cyliner_height_ratio = 0.95
    cone_height_ratio = 0.05
    axis = o3d.geometry.TriangleMesh.create_arrow(
        cylinder_radius=length * cylinder_radius_ratio,
        cone_radius=length * cone_radius_ratio,
        cylinder_height=length * cyliner_height_ratio,
        cone_height=length * cone_height_ratio,
    )
    axis.paint_uniform_color(np.array(color, dtype=np.float64))
    dummy_axis_1 = np.ones(3, dtype=np.float64) * np.sqrt(3) / 3
    dummy_axis_2 = dummy_axis_1.copy()
    dummy_axis_2[0] = -dummy_axis_2[0]
    rot = np.array(rot, dtype=np.float64)
    assert np.allclose(np.linalg.norm(rot), np.array(1)), "rot norm not close to 1"
    if np.allclose(rot, dummy_axis_1):
        dummy_axis = dummy_axis_2
    else:
        dummy_axis = dummy_axis_1
    dummy_x = np.cross(rot, dummy_axis)
    dummy_x = dummy_x / np.linalg.norm(dummy_x)
    dummy_y = np.cross(rot, dummy_x)
    dummy_y = dummy_y / np.linalg.norm(dummy_y)
    rot_mat = np.vstack([dummy_x, dummy_y, rot]).T
    axis.rotate(rot_mat, center=np.zeros(3, dtype=np.float64))
    axis.translate(np.array(translation, dtype=np.float64))
    return axis
