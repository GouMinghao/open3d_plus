import cv2
import time

import numpy as np
import torch
import open3d as o3d

from open3d_plus import (
    render_pointcloud_torch,
    render_pointcloud_numpy,
    paint_pcd_by_axis,
    render_o3d_pointcloud,
    render_pcd_around_axis,
)


def test_render_views():
    world = o3d.geometry.TriangleMesh.create_coordinate_frame(10)
    pcd = o3d.io.read_point_cloud("data/down_color.pcd")
    pcd.translate(-pcd.get_center())
    o3d.visualization.draw_geometries([world, pcd])
    intrinsic = o3d.camera.PinholeCameraIntrinsic(width=1920, height=1080, fx=200, fy=200, cx=959.5, cy=539.5)
    vis_list = render_pcd_around_axis(pcd, intrinsic, "x-", var=5)
    for vis in vis_list:
        cv2.imshow("vis", vis)
        cv2.waitKey(1)
    cv2.destroyAllWindows()


def test_custom_render():
    pcd = o3d.io.read_point_cloud("data/down_color.pcd")
    points = np.asarray(pcd.points)
    pcd = pcd.translate(-np.mean(points, axis=0))
    intrinsic = o3d.camera.PinholeCameraIntrinsic(fx=500, fy=500, cx=639.9, cy=359.5, width=1280, height=720)
    extrinsic = np.linalg.inv(np.array(([1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, -10], [0, 0, 0, 1]), dtype=np.float64))
    pinhole_camera_param = o3d.camera.PinholeCameraParameters()
    pinhole_camera_param.intrinsic = intrinsic
    pinhole_camera_param.extrinsic = extrinsic
    np_points = np.asarray(pcd.points)
    np_colors = np.asarray(pcd.colors)
    torch_points = torch.from_numpy(np_points).cuda()
    torch_colors = torch.from_numpy(np_colors).cuda()

    TEST_TIME = 10
    t1 = time.time()
    for _ in range(TEST_TIME):
        vis_gpu = render_pointcloud_torch(torch_points, torch_colors, pinhole_camera_param, 1)
    t2 = time.time()

    t3 = time.time()
    for _ in range(TEST_TIME):
        vis_cpu = render_pointcloud_numpy(np_points, np_colors, pinhole_camera_param, 1)
    t4 = time.time()

    t5 = time.time()
    for _ in range(TEST_TIME):
        vis_o3d_gpu = render_o3d_pointcloud(pcd, pinhole_camera_param, 1, True)
    t6 = time.time()

    t7 = time.time()
    for _ in range(TEST_TIME):
        vis_o3d_cpu = render_o3d_pointcloud(pcd, pinhole_camera_param, 1, False)
    t8 = time.time()

    print(f"gpu time: {(t2 - t1) / TEST_TIME}s")
    print(f"cpu time: {(t4 - t3) / TEST_TIME}s")
    print(f"o3d-gpu time: {(t6 - t5) / TEST_TIME}s")
    print(f"o3d-cpu time: {(t8 - t7) / TEST_TIME}s")
    cv2.imwrite("vis_cpu.png", vis_cpu)
    cv2.imwrite("vis_gpu.png", vis_gpu)
    cv2.imwrite("vis_o3d_gpu.png", vis_o3d_gpu)
    cv2.imwrite("vis_o3d_cpu.png", vis_o3d_cpu)


if __name__ == "__main__":
    # test_custom_render()
    test_render_views()
