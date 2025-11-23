import rawpy
import numpy as np
import cv2

import rawpy
import numpy as np


# 不解拜尔阵列的输出每个颜色滤镜的传感器效果


def raw_to_bayer_rgb(path):
    with rawpy.imread(path) as raw:
        pattern = raw.raw_pattern
        raw_data = raw.raw_image
        h, w = raw_data.shape

        print(raw_data)
        
        # 创建输出RGB图像（保持原始数据位数）
        rgb = np.zeros((h, w, 3), dtype=raw_data.dtype)
        
        # 扩展Bayer模式到全图尺寸
        r_map = np.tile(pattern == 0, (h//2+1, w//2+1))[:h,:w]
        g_map = np.tile((pattern == 1) | (pattern == 3), (h//2+1, w//2+1))[:h,:w]
        b_map = np.tile(pattern == 2, (h//2+1, w//2+1))[:h,:w]
        
        # 将原始数据分配到对应通道
        rgb[..., 0] = raw_data * r_map*100  # R通道
        rgb[..., 1] = raw_data * g_map*100  # G通道
        rgb[..., 2] = raw_data * b_map*100  # B通道
        
        return rgb


def save_16bit_png(filename, rgb_image):
    # 直接保存16位数据
    cv2.imwrite(filename, rgb_image)

img = raw_to_bayer_rgb("input/IMG20251119214636.dng")
save_16bit_png('test.png', img)
