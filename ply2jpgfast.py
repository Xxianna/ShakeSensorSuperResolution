import open3d as o3d
import numpy as np
import cv2
from tqdm import tqdm
import multiprocessing as mp


# 放大系数
resolution_mul = 4



def process_single_channel(file_path, channel_idx, output_file, x_min, x_max, y_min, y_max, width, height, show_progress=False):
    # 读取点云
    pcd = o3d.io.read_point_cloud(file_path)
    
    # 构建KD树
    kdtree = o3d.geometry.KDTreeFlann(pcd)
    
    # 处理图像
    image = np.zeros((height, width), dtype=np.int16)
    colors = np.asarray(pcd.normals)
    
    xy_range_x = x_max - x_min
    xy_range_y = y_max - y_min
    
    iterator = tqdm(range(height), desc=f"Processing Channel {channel_idx}") if show_progress else range(height)
    
    for y in iterator:
        for x in range(width):
            sample_x = x_min + (x / (width - 1)) * xy_range_x
            sample_y = y_min + (y / (height - 1)) * xy_range_y
            sample_point = np.array([sample_x, sample_y, 0.0])
            
            _, indices, _ = kdtree.search_knn_vector_3d(sample_point, 1)
            if len(indices) > 0:
                image[y, x] = int(colors[indices[0], channel_idx])
    
    return image

def get_bounding_box():
    # 读取三个单色点云
    pcd_r = o3d.io.read_point_cloud("result_R.ply")
    pcd_g = o3d.io.read_point_cloud("result_G.ply")
    pcd_b = o3d.io.read_point_cloud("result_B.ply")
    
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
    
    return x_min, x_max, y_min, y_max

def main():
    # 获取包围盒
    x_min, x_max, y_min, y_max = get_bounding_box()
    print(f"取整后XY包围盒范围: x=[{x_min}, {x_max}], y=[{y_min}, {y_max}]")
    
    # 构建二维图像，分辨率为包围盒尺寸的二倍
    width = (x_max - x_min + 1) * resolution_mul
    height = (y_max - y_min + 1) * resolution_mul
    
    # 多进程处理三个通道
    with mp.Pool(3) as pool:
        results = pool.starmap(process_single_channel, [
            ("result_R.ply", 0, "output_ply2png_R.jpg", x_min, x_max, y_min, y_max, width, height, False),
            ("result_G.ply", 1, "output_ply2png_G.jpg", x_min, x_max, y_min, y_max, width, height, True),
            ("result_B.ply", 2, "output_ply2png_B.jpg", x_min, x_max, y_min, y_max, width, height, False),
        ])
    
    # 合成最终图像
    img_r, img_g, img_b = results
    final_image = np.stack([img_r, img_g, img_b], axis=-1)
    final_image_uint16 = np.clip(final_image, 0, 65535).astype(np.uint16)
    cv2.imwrite("output_ply2png_fast.png", final_image_uint16)

if __name__ == '__main__':
    mp.set_start_method('spawn')
    main()
