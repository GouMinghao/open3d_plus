from open3d_plus.geometry import create_axis
import open3d as o3d
import math

if __name__ == "__main__":
    axis = create_axis(5, [1, 0, 0], [math.sqrt(2) / 2, math.sqrt(2) / 2, 0], [-1, 1, 1])
    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(1)
    o3d.visualization.draw_geometries([axis, frame])
