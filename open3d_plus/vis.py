import open3d as o3d
import numpy as np

import matplotlib.cm as cm

try:
    import torch

    if torch.cuda.is_available():
        USE_TORCH_CUDA = True
except:
    print("torch cuda not avaliable")
    USE_TORCH_CUDA = False


def paint_pcd_by_axis(pcd: o3d.geometry.PointCloud, axis: int, var: float = 2):
    points = np.asarray(pcd.points)
    vals = points[:, axis]
    mean_val = np.mean(vals)
    var_val = np.sqrt(np.var(vals))
    min_val = mean_val - var * var_val
    max_val = mean_val + var * var_val
    norm_range = (vals - min_val) / (max_val - min_val)
    clip_norm_range = np.clip(norm_range, 0, 1)
    colors = cm.jet(clip_norm_range)[:, :3]
    pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd


def render_o3d_pointcloud(
    pcd: o3d.geometry.PointCloud,
    pinhole_camera_param: o3d.camera.PinholeCameraParameters,
    point_radius: int = 1,
    use_gpu: bool = False,
) -> np.ndarray:
    """render point cloud with numpy or torch

    Args:
        pcd (Union[np.ndarray, torch.Tensor, o3d.geometry]): point cloud in either np/torch/open3d format
        pinhole_camera_param (o3d.camera.PinholeCameraParameters): camera params
        point_radius(int, optional): render point size. Defaults to 1
        use_gpu (bool, optional): if use gpu. Defaults to False.

    Returns:
        np.ndarray: image in uint8 (H, W, C), BGR format
    """
    np_points = np.asarray(pcd.points)
    np_colors = np.asarray(pcd.colors)
    if use_gpu and USE_TORCH_CUDA:
        torch_points = torch.from_numpy(np_points).cuda()
        torch_colors = torch.from_numpy(np_colors).cuda()
        return render_pointcloud_torch(
            torch_points, torch_colors, pinhole_camera_param=pinhole_camera_param, point_radius=point_radius
        )
    else:
        return render_pointcloud_numpy(
            np_points, np_colors, pinhole_camera_param=pinhole_camera_param, point_radius=point_radius
        )


def project_to_2d_torch(fx, fy, cx, cy, points_3d):
    Z = points_3d[:, 2]
    x_normalized = points_3d[:, 0] / Z
    y_normalized = points_3d[:, 1] / Z
    u = (fx * x_normalized + cx).to(torch.int64)
    v = (fy * y_normalized + cy).to(torch.int64)
    return torch.column_stack((u, v))


def render_pointcloud_torch(
    points: torch.Tensor,
    colors: torch.Tensor,
    pinhole_camera_param: o3d.camera.PinholeCameraParameters,
    point_radius: int = 1,
) -> np.ndarray:
    """render point cloud with torch

    Args:
        points (torch.Tensor): points coords
        colors (torch.Tensor): colors of the points, [0 - 1], RGB
        pinhole_camera_param (o3d.camera.PinholeCameraParameters): _description_
        point_radius (int): point rendering radius

    Returns:
        np.ndarray: image in uint8 (H, W, C), BGR format
    """
    intrin = pinhole_camera_param.intrinsic
    vis = 255 * torch.ones((intrin.height, intrin.width, 3), dtype=torch.uint8).cuda()
    bgr_colors = (colors * 255).to(torch.uint8)[:, [2, 1, 0]]
    num_pts = points.shape[0]
    # transform points to camera frame
    camera_points = torch.matmul(
        torch.linalg.inv(torch.from_numpy(pinhole_camera_param.extrinsic).cuda()),
        torch.hstack((points, torch.ones((num_pts, 1), dtype=points.dtype).cuda())).T,
    ).T[:, :3]
    valid_mask = camera_points[:, 2] > 0
    valid_pts = camera_points[valid_mask]
    valid_colors = bgr_colors[valid_mask]
    dist_down_idxs = torch.flip(torch.argsort(torch.linalg.norm(valid_pts, axis=1)), [0])
    sorted_pts = valid_pts[dist_down_idxs]  # from longest to shortest
    sorted_colors = valid_colors[dist_down_idxs]
    fx, fy = intrin.get_focal_length()
    cx, cy = intrin.get_principal_point()
    uvs = project_to_2d_torch(fx, fy, cx, cy, sorted_pts)  # [n, 2]
    offsets = torch.arange(-point_radius, point_radius + 1)
    grid_size = 2 * point_radius + 1
    all_colors = torch.einsum("ij,k->ikj", sorted_colors, torch.ones(grid_size**2, dtype=torch.uint8).cuda()).reshape(
        -1, 3
    )
    meshgrids = torch.from_numpy(np.array(np.meshgrid(offsets, offsets)).T.reshape((-1, 2))).cuda()
    all_uvs = (
        torch.einsum("ij,k->ikj", uvs, torch.ones(grid_size**2, dtype=torch.int64).cuda()) + meshgrids
    ).reshape((-1, 2))
    all_uvs[:, 0] = torch.clip(all_uvs[:, 0], 0, intrin.width - 1)
    all_uvs[:, 1] = torch.clip(all_uvs[:, 1], 0, intrin.height - 1)
    vis[all_uvs[:, 1], all_uvs[:, 0]] = all_colors
    return vis.cpu().numpy()


def project_to_2d_numpy(fx, fy, cx, cy, points_3d):
    Z = points_3d[:, 2]
    x_normalized = points_3d[:, 0] / Z
    y_normalized = points_3d[:, 1] / Z
    u = (fx * x_normalized + cx).astype(np.int64)
    v = (fy * y_normalized + cy).astype(np.int64)
    return np.column_stack((u, v))


def render_pointcloud_numpy(
    points: np.ndarray,
    colors: np.ndarray,
    pinhole_camera_param: o3d.camera.PinholeCameraParameters,
    point_radius: int = 1,
) -> np.ndarray:
    """render point cloud with numpy

    Args:
        points (np.ndarray): points coords
        colors (np.ndarray): colors of the points, [0 - 1], RGB
        pinhole_camera_param (o3d.camera.PinholeCameraParameters): _description_
        point_radius (int): point rendering radius

    Returns:
        np.ndarray: image in uint8 (H, W, C), BGR format
    """
    intrin = pinhole_camera_param.intrinsic
    vis = 255 * np.ones((intrin.height, intrin.width, 3), dtype=np.uint8)
    bgr_colors = (colors * 255).astype(np.uint8)[:, [2, 1, 0]]
    num_pts = points.shape[0]
    # transform points to camera frame
    camera_points = np.matmul(
        np.linalg.inv(pinhole_camera_param.extrinsic), np.hstack((points, np.ones((num_pts, 1), dtype=points.dtype))).T
    ).T[:, :3]
    valid_mask = camera_points[:, 2] > 0
    valid_pts = camera_points[valid_mask]
    valid_colors = bgr_colors[valid_mask]
    dist_down_idxs = np.argsort(np.linalg.norm(valid_pts, axis=1))[::-1]
    sorted_pts = valid_pts[dist_down_idxs]  # from longest to shortest
    sorted_colors = valid_colors[dist_down_idxs]
    fx, fy = intrin.get_focal_length()
    cx, cy = intrin.get_principal_point()
    uvs = project_to_2d_numpy(fx, fy, cx, cy, sorted_pts)  # [n, 2]
    offsets = np.arange(-point_radius, point_radius + 1)
    grid_size = 2 * point_radius + 1
    all_colors = np.einsum("ij,k->ikj", sorted_colors, np.ones(grid_size**2, dtype=int)).reshape(-1, 3)
    all_uvs = (
        np.einsum("ij,k->ikj", uvs, np.ones(grid_size**2, dtype=int))
        + np.array(np.meshgrid(offsets, offsets)).T.reshape(-1, 2)
    ).reshape((-1, 2))
    all_uvs[:, 0] = np.clip(all_uvs[:, 0], 0, intrin.width - 1)
    all_uvs[:, 1] = np.clip(all_uvs[:, 1], 0, intrin.height - 1)
    vis[all_uvs[:, 1], all_uvs[:, 0]] = all_colors
    return vis
