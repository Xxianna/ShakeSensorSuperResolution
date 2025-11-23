import os
import rawpy
import cv2
import numpy as np
from skimage.transform import resize_local_mean
import imageio

# 创建文件夹
os.makedirs('middle', exist_ok=True)

# 获取所有DNG文件
files = sorted([f for f in os.listdir('input') if f.lower().endswith('.dng')])
aligned_images = []

for i, file in enumerate(files):
    # 读取并解拜尔
    with rawpy.imread(os.path.join('input', file)) as raw:
        img = raw.postprocess(use_camera_wb=True)
    
    # 两倍放大（邻近采样）
    h, w = img.shape[:2]
    img_upscaled = cv2.resize(img, (w*2, h*2), interpolation=cv2.INTER_NEAREST)
    
    # 第一张作为基准
    if i == 0:
        base_img = img_upscaled
        cv2.imwrite(f'middle/aligned_0.png', img_upscaled)
        aligned_images.append(img_upscaled)
    else:
        # SIFT特征检测
        sift = cv2.SIFT_create()
        kp1, des1 = sift.detectAndCompute(base_img, None)
        kp2, des2 = sift.detectAndCompute(img_upscaled, None)
        
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
            
            # 应用变换
            aligned = cv2.warpPerspective(img_upscaled, M, (base_img.shape[1], base_img.shape[0]))
        else:
            aligned = img_upscaled
            
        cv2.imwrite(f'middle/aligned_{i}.png', aligned)
        aligned_images.append(aligned)

# 转换为numpy数组并计算平均
aligned_array = np.array(aligned_images, dtype=np.float32)
average_img = np.mean(aligned_array, axis=0).astype(np.uint8)

# 保存结果
cv2.imwrite('result.png', average_img)
