import open3d as o3d
import numpy as np
# import cupy as cp
from scipy.spatial import cKDTree
import imageio

# 读取PLY文件
pcd = o3d.io.read_point_cloud("result.ply")
points = np.asarray(pcd.points)[:, :2]  # 只取XY平面
colors = np.asarray(pcd.colors)

# 分离RGB通道的点
r_mask = colors[:, 0] > 0
g_mask = colors[:, 1] > 0
b_mask = colors[:, 2] > 0

r_points = points[r_mask]
g_points = points[g_mask]
b_points = points[b_mask]

# 计算边界
min_xy = np.min(points, axis=0)
max_xy = np.max(points, axis=0)


print(min_xy)
print(max_xy)




