import open3d as o3d
import numpy as np

def check_ply_color_range(ply_path):
    pcd = o3d.io.read_point_cloud(ply_path)
    if not pcd.has_colors():
        print("点云无颜色信息")
        return
    
    colors = np.asarray(pcd.colors)
    print(f"颜色数据类型: {colors.dtype}")
    print(f"颜色范围: [{colors.min():.6f}, {colors.max():.6f}]")
    print(f"理论16位整数范围: [{int(colors.min()*65535)}, {int(colors.max()*65535)}]")

# 使用示例
check_ply_color_range("result_R.ply")
