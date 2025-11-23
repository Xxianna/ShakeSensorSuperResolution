import os
import cv2
import numpy as np


# 先每两张图合成一次，再将所有之前的合成合成为一张
# 经过验证，一加12主摄没有第二次合成的意义，拍摄两张图（12mp）合成一次即为最高性价比，和原生48mp的高像素模式效果差不多
# 也即一加12主摄的镜头分辨率不超过50mp





def align_and_blend(images):
    """ 输入一组16位图像，返回配准后融合的16位图像 """
    assert all(img.dtype == np.uint16 for img in images), "All images must be 16-bit"

    # 先对所有图像进行2倍上采样（邻近插值）
    upscaled_images = []
    for img in images:
        h, w = img.shape[:2]
        upscaled = cv2.resize(img, (w*2, h*2), interpolation=cv2.INTER_NEAREST)
        upscaled_images.append(upscaled)
    
    base = upscaled_images[0]
    blended = [base]

    # 用于SIFT的8位图像
    base_8bit = cv2.convertScaleAbs(base, alpha=(255.0/65535.0))
    sift = cv2.SIFT_create()

    for img in upscaled_images[1:]:
        img_8bit = cv2.convertScaleAbs(img, alpha=(255.0/65535.0))
        
        kp1, des1 = sift.detectAndCompute(base_8bit, None)
        kp2, des2 = sift.detectAndCompute(img_8bit, None)

        if des1 is None or des2 is None:
            blended.append(img)
            continue

        matcher = cv2.BFMatcher()
        matches = matcher.knnMatch(des1, des2, k=2)

        good = [m for m, n in matches if m.distance < 0.7 * n.distance]

        if len(good) > 4:
            src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
            M, _ = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
            aligned = cv2.warpPerspective(img, M, (base.shape[1], base.shape[0]))
        else:
            aligned = img

        blended.append(aligned)

    # 平均融合（保持16位）
    stack = np.stack(blended, axis=0).astype(np.float32)
    result = np.mean(stack, axis=0).astype(np.uint16)
    return result




def multi_scale_blend(image_groups, output_dir='middle'):
    """
    只进行两级合成：
    第一级：每组两张图像对齐融合，输出中间结果；
    第二级：将所有中间结果再次对齐融合，得最终高分辨率图像。
    """
    # os.makedirs(output_dir, exist_ok=True)

    # ---------- 第一级：两两配准融合 ----------
    first_level_results = []
    for idx, group in enumerate(image_groups):
        print(f"Processing group {idx} at Level 1")
        merged = align_and_blend(group)  # 内部包含2倍上采样
        # filename = os.path.join(output_dir, f'level1_group{idx}.png')
        # cv2.imwrite(filename, merged)
        first_level_results.append(merged)
        print(merged.shape)

    # ---------- 第二级：整体融合 ----------
    print("Processing final merge at Level 2")
    final_image = align_and_blend(first_level_results)  # 包含第二次2倍上采样

    final_path = os.path.join(output_dir, 'final_result.png')
    cv2.imwrite(final_path, final_image)
    print(final_image.shape)

    return final_image



# 使用示例：
if __name__ == "__main__":
    from glob import glob
    import rawpy

    def load_dng_as_16bit(path):
        with rawpy.imread(path) as raw:
            img = raw.postprocess(use_camera_wb=True, output_bps=16)
        # 不再在这里做 resize
        return img  # 直接返回原图

    input_files = sorted(glob("input/*.dng"))
    loaded_images = [load_dng_as_16bit(f) for f in input_files]

    # 每两个分为一组传入函数
    groups = [loaded_images[i:i+2] for i in range(0, len(loaded_images), 2)]
    multi_scale_blend(groups)
