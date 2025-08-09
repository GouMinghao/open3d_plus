from open3d_plus import merge_pcds, array2pcd
import open3d as o3d
import numpy as np

if __name__ == "__main__":
    pt1 = np.array([[0, 0, 0]], dtype=np.float64)
    color1 = np.array([[1, 1, 1]], dtype=np.float64)
    pt2 = np.array([[2, 2, 2]], dtype=np.float64)
    color2 = np.array([[3, 4, 4]], dtype=np.float64)
    pcd1 = array2pcd(pt1, color1)
    pcd2 = array2pcd(pt2, color2)
    merge_color_pcd = merge_pcds([pcd1, pcd2], True)
    merge_nocolor_pcd = merge_pcds([pcd1, pcd2], False)
    pcd3 = array2pcd(pt1, None)
    pcd4 = array2pcd(pt2, None)
    merge_nocolor_pcd2 = merge_pcds([pcd3, pcd4], False)
    print(merge_color_pcd)
    print(merge_nocolor_pcd)
    print(merge_nocolor_pcd2)
