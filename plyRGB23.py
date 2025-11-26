import open3d as o3d
import numpy as np

# 读取点云
pcd = o3d.io.read_point_cloud("result.ply")
points = np.asarray(pcd.points)
normals = np.asarray(pcd.normals)  # 从法线读取颜色信息

# 创建布尔索引（基于法线中的颜色值）
idx_r = (normals[:, 0] > 0) & (normals[:, 1] == 0) & (normals[:, 2] == 0)
idx_g = (normals[:, 1] > 0) & (normals[:, 0] == 0) & (normals[:, 2] == 0)
idx_b = (normals[:, 2] > 0) & (normals[:, 0] == 0) & (normals[:, 1] == 0)

# 拆分并保存
for name, idx in [("R", idx_r), ("G", idx_g), ("B", idx_b)]:
    pcd_new = o3d.geometry.PointCloud()
    pcd_new.points = o3d.utility.Vector3dVector(points[idx])
    pcd_new.normals = o3d.utility.Vector3dVector(normals[idx])  # 仍存回法线
    o3d.io.write_point_cloud(f"result_{name}.ply", pcd_new)
