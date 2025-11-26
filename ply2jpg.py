import open3d as o3d
import numpy as np
import cv2
from tqdm import tqdm

# 读取三个单色点云
pcd_r = o3d.io.read_point_cloud("result_R.ply")
pcd_g = o3d.io.read_point_cloud("result_G.ply")
pcd_b = o3d.io.read_point_cloud("result_B.ply")

# 构建KD树
kdtree_r = o3d.geometry.KDTreeFlann(pcd_r)
kdtree_g = o3d.geometry.KDTreeFlann(pcd_g)
kdtree_b = o3d.geometry.KDTreeFlann(pcd_b)

# 计算xy包围盒
points_r = np.asarray(pcd_r.points)
points_g = np.asarray(pcd_g.points)
points_b = np.asarray(pcd_b.points)

# 获取所有点的xy范围
all_points = np.vstack([points_r, points_g, points_b])
xy_min = np.min(all_points[:, :2], axis=0)
xy_max = np.max(all_points[:, :2], axis=0)

# 包围盒向内取整
x_min = int(np.ceil(xy_min[0]))
y_min = int(np.ceil(xy_min[1]))
x_max = int(np.floor(xy_max[0])) - 1
y_max = int(np.floor(xy_max[1])) - 1

print(f"取整后XY包围盒范围: x=[{x_min}, {x_max}], y=[{y_min}, {y_max}]")

# 构建二维图像，分辨率为包围盒尺寸的二倍
width = (x_max - x_min + 1) * 2
height = (y_max - y_min + 1) * 2

# 创建opencv图像
# 创建 OpenCV 图像（使用 int16）
image = np.zeros((height, width, 3), dtype=np.int16)

# 计算保存间隔
save_interval = max(1, height // 20)  # 每5%保存一次

# 添加进度条
for y in tqdm(range(height), desc="Processing rows"):
    for x in range(width):
        # 计算三维采样点坐标
        sample_x = x_min + (x / (width - 1)) * (x_max - x_min)
        sample_y = y_min + (y / (height - 1)) * (y_max - y_min)
        sample_point = np.array([sample_x, sample_y, 0.0])
        
        # 在R点云中找最近点
        _, indices, _ = kdtree_r.search_knn_vector_3d(sample_point, 1)
        if len(indices) > 0:
            image[y, x, 0] = int(np.asarray(pcd_r.normals)[indices[0], 0])  # 直接取值，不乘65535
        
        # 在G点云中找最近点
        _, indices, _ = kdtree_g.search_knn_vector_3d(sample_point, 1)
        if len(indices) > 0:
            image[y, x, 1] = int(np.asarray(pcd_g.normals)[indices[0], 1])
        
        # 在B点云中找最近点
        _, indices, _ = kdtree_b.search_knn_vector_3d(sample_point, 1)
        if len(indices) > 0:
            image[y, x, 2] = int(np.asarray(pcd_b.normals)[indices[0], 2])
    
    # 每完成5%保存一次图片（转成 uint8 用于 JPG）
    if (y + 1) % save_interval == 0 or y == height - 1:
        # 根据最大最小值映射到0-255
        img_min = image.min()
        img_max = image.max()
        if img_max > img_min:
            image_normalized = (image - img_min) / (img_max - img_min) * 255
        else:
            image_normalized = np.zeros_like(image)
        image_uint8 = image_normalized.astype(np.uint8)
        cv2.imwrite(f"output_ply2png.jpg", image_uint8)


# 最终保存为 16 位 PNG（保持原始 int16 精度）
cv2.imwrite("output_ply2png.png", image)

