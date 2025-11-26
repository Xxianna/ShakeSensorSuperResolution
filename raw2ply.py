import os
import rawpy
import cv2
import numpy as np
import open3d as o3d


## 将raw转换为ply，按照每个点的拜尔滤镜直接构建彩色点云
## TODO 特征点获取算法是否需要更新 配准算法是否需要更新 是否应该考虑旋转构建三维平面




# 获取所有DNG文件
files = sorted([f for f in os.listdir('input') if f.lower().endswith('.dng')])

# files = files[0:2]

aligned_images = []
M_list = []  # 存储每张图的变换矩阵

for i, file in enumerate(files):
    # 读取并解拜尔
    with rawpy.imread(os.path.join('input', file)) as raw:
        img = raw.postprocess(use_camera_wb=True)
    
    # 不再进行上采样，直接使用原始尺寸图像
    aligned_images.append(img)
    
    if i == 0:
        base_img = img
        M_list.append(np.eye(3))  # 第一张为基准，无变换
    else:
        # SIFT特征检测
        sift = cv2.SIFT_create()
        kp1, des1 = sift.detectAndCompute(base_img, None)
        kp2, des2 = sift.detectAndCompute(img, None)
        
        # 特征匹配
        matcher = cv2.BFMatcher()
        matches = matcher.knnMatch(des1, des2, k=2)
        
        good = []
        for m, n in matches:
            if m.distance < 0.7 * n.distance:
                good.append(m)
                
        if len(good) > 4:
            src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
            
            # 计算单应性矩阵
            M, _ = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
            M_list.append(M)

            print(M)
        else:
            M_list.append(np.eye(3))  # 匹配失败则不变化

# 生成并合并点云（修改后不使用 aligned_images）
final_pcd = o3d.geometry.PointCloud()

# 基准图像用于获取尺寸
with rawpy.imread(os.path.join('input', files[0])) as raw:
    base_img = raw.postprocess(use_camera_wb=True)
h_base, w_base = base_img.shape[:2]


def raw_to_bayer_rgb(path):
    with rawpy.imread(path) as raw:
        pattern = raw.raw_pattern
        raw_data = raw.raw_image
        h, w = raw_data.shape
        
        # 创建输出RGB图像（保持原始数据位数）
        rgb = np.zeros((h, w, 3), dtype=raw_data.dtype)
        
        # 扩展Bayer模式到全图尺寸
        r_map = np.tile(pattern == 0, (h//2+1, w//2+1))[:h,:w]
        g_map = np.tile((pattern == 1) | (pattern == 3), (h//2+1, w//2+1))[:h,:w]
        b_map = np.tile(pattern == 2, (h//2+1, w//2+1))[:h,:w]
        
        # 将原始数据分配到对应通道
        rgb[..., 0] = raw_data * r_map  # R通道
        rgb[..., 1] = raw_data * g_map  # G通道
        rgb[..., 2] = raw_data * b_map  # B通道

        rgb = np.rot90(rgb, k=2)  # raw数据和输出数据有旋转180度关系
        
        return rgb

for i, file in enumerate(files):
    # 重新读取图像
    img = raw_to_bayer_rgb(os.path.join('input', file))

    h, w = img.shape[:2]

    # 生成该图像的所有像素坐标 (x,y) -> 注意OpenCV习惯是反的
    y_coords, x_coords = np.mgrid[0:h, 0:w]
    coords = np.hstack([x_coords.reshape(-1, 1), y_coords.reshape(-1, 1)]).astype(np.float32)

    # 应用单应性变换到基准坐标系
    if i != 0 and M_list[i] is not None:
        coords = cv2.perspectiveTransform(coords.reshape(1, -1, 2), M_list[i]).reshape(-1, 2)

    # 构造三维点云 Z=0
    points = np.hstack([coords, np.zeros((coords.shape[0], 1))])

    # 提取颜色值并归一化
    colors = img.reshape(-1, 3)[:, ::-1] / 65536.0  # BGR -> RGB and normalize 
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    final_pcd += pcd

o3d.io.write_point_cloud("result.ply", final_pcd)
