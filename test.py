import cv2
import numpy as np

# 读取图像
img = cv2.imread('output_ply2png_fast.png', cv2.IMREAD_UNCHANGED)

# 转换为浮点型并归一化到0-1范围
img_normalized = img.astype(np.float32)
img_normalized = (img_normalized - img_normalized.min()) / (img_normalized.max() - img_normalized.min())

# 缩放到0-255并转换为uint8
img_8bit = (img_normalized * 255).astype(np.uint8)

# 保存结果
cv2.imwrite('output_ply2png_fast_8.jpg', img_8bit)
