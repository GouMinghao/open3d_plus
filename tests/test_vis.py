from pathlib import Path
from typing import Union
import cv2

import numpy as np
import torch
import open3d as o3d

from open3d_plus import render_pointcloud_torch, render_pointcloud_numpy, paint_pcd_by_axis, render_o3d_pointcloud

if __name__ == "__main__":
    import time

    pcd = o3d.io.read_point_cloud("../tests/data/down_sample.pcd")
    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(100)
    points = np.asarray(pcd.points)
    pcd = pcd.translate(-np.mean(points, axis=0))
    pcd = paint_pcd_by_axis(pcd, 1, 2)
    intrinsic = o3d.camera.PinholeCameraIntrinsic(fx=500, fy=500, cx=639.9, cy=359.5, width=1280, height=720)
    extrinsic = np.array(([1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, -1000], [0, 0, 0, 1]), dtype=np.float64)
    pinhole_camera_param = o3d.camera.PinholeCameraParameters()
    pinhole_camera_param.intrinsic = intrinsic
    pinhole_camera_param.extrinsic = extrinsic
    np_points = np.asarray(pcd.points)
    np_colors = np.asarray(pcd.colors)
    torch_points = torch.from_numpy(np_points).cuda()
    torch_colors = torch.from_numpy(np_colors).cuda()
    t1 = time.time()
    for _ in range(10):
        vis_gpu = render_pointcloud_torch(torch_points, torch_colors, pinhole_camera_param, 1)
    t2 = time.time()

    t3 = time.time()
    for _ in range(10):
        vis_cpu = render_pointcloud_numpy(np_points, np_colors, pinhole_camera_param, 1)
    t4 = time.time()

    t5 = time.time()
    for _ in range(10):
        vis_o3d_gpu = render_o3d_pointcloud(pcd, pinhole_camera_param, 1, True)
    t6 = time.time()

    t7 = time.time()
    for _ in range(10):
        vis_o3d_cpu = render_o3d_pointcloud(pcd, pinhole_camera_param, 1, False)
    t8 = time.time()

    print(f"gpu time: {(t2 - t1) / 10}s")
    print(f"cpu time: {(t4 - t3) / 10}s")
    print(f"o3d-gpu time: {(t6 - t5) / 10}s")
    print(f"o3d-cpu time: {(t8 - t7) / 10}s")
    cv2.imwrite("vis_cpu.png", vis_cpu)
    cv2.imwrite("vis_gpu.png", vis_gpu)
    cv2.imwrite("vis_o3d_gpu.png", vis_o3d_gpu)
    cv2.imwrite("vis_o3d_cpu.png", vis_o3d_cpu)
