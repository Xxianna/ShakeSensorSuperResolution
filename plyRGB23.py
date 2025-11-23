import open3d as o3d
import numpy as np

# 读取点云
pcd = o3d.io.read_point_cloud("result.ply")
points = np.asarray(pcd.points)
colors = np.asarray(pcd.colors)

# 创建布尔索引
idx_r = (colors[:, 0] > 0) & (colors[:, 1] == 0) & (colors[:, 2] == 0)
idx_g = (colors[:, 1] > 0) & (colors[:, 0] == 0) & (colors[:, 2] == 0)
idx_b = (colors[:, 2] > 0) & (colors[:, 0] == 0) & (colors[:, 1] == 0)

# 拆分并保存
for name, idx in [("R", idx_r), ("G", idx_g), ("B", idx_b)]:
    pcd_new = o3d.geometry.PointCloud()
    pcd_new.points = o3d.utility.Vector3dVector(points[idx])
    pcd_new.colors = o3d.utility.Vector3dVector(colors[idx])
    o3d.io.write_point_cloud(f"result_{name}.ply", pcd_new)
