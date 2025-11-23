import open3d as o3d

# 读取点云文件
pcd = o3d.io.read_point_cloud("result.ply")

# 可视化
o3d.visualization.draw_geometries([pcd])
