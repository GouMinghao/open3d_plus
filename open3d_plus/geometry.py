import open3d as o3d
import numpy as np

def array2pcd(points, colors):
    """
    Convert points and colors into open3d point cloud.

    Args:
        points(np.array): coordinates of the points.
        colors(np.array): RGB values of the points.

    Returns:
        open3d.geometry.PointCloud: the point cloud.
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
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

def merge_pcds(pcds):
    """
    Merge several point cloud into a single one.

    Args:
        pcds(list): list of point cloud.

    Returns:
        open3d.geometry.PointCloud: the merged point cloud.
    """
    ret_pcd = o3d.geometry.PointCloud()
    if len(pcds) == 0:
        return ret_pcd
    old_points, old_colors = pcd2array(pcds[0])
    for i in range(1, len(pcds)):
        points, colors = pcd2array(pcds[i])
        old_points = np.concatenate((old_points, points))
        old_colors = np.concatenate((old_colors, colors)) 
    return array2pcd(old_points, old_colors)