import open3d as o3d
import numpy as np
from pathlib import Path
from typing import Union

try:
    import torch
    if torch.cuda.is_available():
        USE_TORCH_CUDA = True
except:
    print("torch cuda not avaliable")
    USE_TORCH_CUDA = False

def render_pointcloud(pcd: Union[np.ndarray, torch.Tensor, o3d.geometry.PointCloud], pinhole_camera_param: o3d.camera.PinholeCameraParameters, use_gpu: bool=False) -> np.ndarray:
    """render point cloud with numpy or torch

    Args:
        pcd (Union[np.ndarray, torch.Tensor, o3d.geometry]): point cloud in either np/torch/open3d format
        pinhole_camera_param (o3d.camera.PinholeCameraParameters): camera params
        use_gpu (bool, optional): if use gpu. Defaults to False.

    Returns:
        np.ndarray: image in uint8 (H, W, C), BGR format
    """
    if isinstance(pcd, o3d.geometry.PointCloud):
        np_points = np.asarray(pcd.points)
        if use_gpu and USE_TORCH_CUDA:
            backend = "torch"
            torch_points = torch.from_numpy(np_points).cuda()
        else:
            backend = "numpy"
    elif isinstance(pcd, np.ndarray):
        np_points = pcd
        if use_gpu and USE_TORCH_CUDA:
            backend = "torch"
            torch_points = torch.from_numpy(np_points).cuda()
        else:
            backend = "numpy"
    elif isinstance(pcd, torch.Tensor):
        assert USE_TORCH_CUDA
        torch_points = pcd
        backend = "torch"
    else:
        raise ValueError(f"Unsupported pcd type: {type(pcd)}")
    if backend == "numpy":
        return render_pointcloud_numpy(np_points, pinhole_camera_param)
    elif backend == "torch":
        return render_pointcloud_torch(torch_points, pinhole_camera_param=pinhole_camera_param)

def render_pointcloud_torch(pcd_tensor: torch.Tensor, pinhole_camera_param: o3d.camera.PinholeCameraParameters) -> np.ndarray:
    """_summary_

    Args:
        pcd_tensor (torch.Tensor): _description_
        pinhole_camera_param (o3d.camera.PinholeCameraParameters): _description_

    Returns:
        np.ndarray: _description_
    """
    pass

def render_pointcloud_numpy(np_points: np.ndarray, pinhole_camera_param: o3d.camera.PinholeCameraParameters) -> np.ndarray:
    """_summary_

    Args:
        np_points (np.ndarray): _description_
        pinhole_camera_param (o3d.camera.PinholeCameraParameters): _description_

    Returns:
        np.ndarray: _description_
    """
    pass
