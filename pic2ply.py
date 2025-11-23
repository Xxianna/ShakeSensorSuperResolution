import os
import rawpy
import cv2
import numpy as np
import open3d as o3d

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
        else:
            M_list.append(np.eye(3))  # 匹配失败则不变化

# 生成并合并点云
final_pcd = o3d.geometry.PointCloud()

# 基准图像尺寸
h_base, w_base = aligned_images[0].shape[:2]

for i, (img, M) in enumerate(zip(aligned_images, M_list)):
    h, w = img.shape[:2]
    
    # 生成该图像的所有像素坐标 (y,x) -> 注意OpenCV习惯
    y_coords, x_coords = np.mgrid[0:h, 0:w]
    coords = np.hstack([x_coords.reshape(-1, 1), y_coords.reshape(-1, 1)]).astype(np.float32)

    # 应用单应性变换到基准坐标系
    if i != 0 and M is not None:
        coords = cv2.perspectiveTransform(coords.reshape(1, -1, 2), M).reshape(-1, 2)

    # 构造三维点云 Z=0
    points = np.hstack([coords, np.zeros((coords.shape[0], 1))])

    # 提取颜色值并归一化
    colors = img.reshape(-1, 3)[:, ::-1] / 255.0  # BGR -> RGB and normalize

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    final_pcd += pcd

# 可选：去除重复点（因为插值可能造成轻微偏移）
# final_pcd = final_pcd.remove_duplicated_points()

o3d.io.write_point_cloud("result1.ply", final_pcd)
