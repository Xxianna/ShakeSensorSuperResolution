import os
import time
import rawpy
import cv2
import numpy as np
import open3d as o3d
from tqdm import tqdm
import multiprocessing as mp


input_dir = "inputd600"
outputname = "d600_96mp.png"
target_resolution_times = 2
rot_180_for_raw = False     # 对手机的dng有时需要旋转，对NEF有时不需要旋转。没配上就转一下




def greeninfo(str):
    if not hasattr(greeninfo, 'start_time'):
        greeninfo.start_time = time.time()
    elapsed = time.time() - greeninfo.start_time
    hours, remainder = divmod(elapsed, 3600)
    minutes, seconds = divmod(remainder, 60)
    timestamp = f"[{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}]"
    print(f"\033[32m【{timestamp}】=========={str}==========\033[0m")

def raw2ply(input_dir, rot_180_for_raw = False):
    # 获取所有DNG文件
    # files = sorted([f for f in os.listdir(input_dir) if f.lower().endswith('.dng')])
    files = sorted([f for f in os.listdir(input_dir)])

    # files = files[0:2]

    aligned_images = []
    M_list = []  # 存储每张图的变换矩阵

    for i, file in enumerate(files):
        # 读取并解拜尔
        with rawpy.imread(os.path.join(input_dir, file)) as raw:
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

                # print(M)
            else:
                M_list.append(np.eye(3))  # 匹配失败则不变化

    greeninfo("图像配准测算完成，开始生成相应点云")

    # 生成并合并点云（修改后不使用 aligned_images）
    final_pcd = o3d.geometry.PointCloud()

    # 基准图像用于获取尺寸
    with rawpy.imread(os.path.join(input_dir, files[0])) as raw:
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

            if rot_180_for_raw:
                rgb = np.rot90(rgb, k=2)  # raw数据和输出数据有旋转180度关系
            
            return rgb

    for i, file in enumerate(files):
        # 重新读取图像
        img = raw_to_bayer_rgb(os.path.join(input_dir, file))

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
        colors = img.reshape(-1, 3)[:, ::-1].astype(np.float32)  # BGR -> RGB and normalize 

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.normals = o3d.utility.Vector3dVector(colors)
        final_pcd += pcd

    # o3d.io.write_point_cloud("result.ply", final_pcd)
    # print(">>> 已生成配准点云！")
    return final_pcd

def plyRGB23(pcd):
    # 读取点云
    # pcd = o3d.io.read_point_cloud("result.ply")
    points = np.asarray(pcd.points)
    normals = np.asarray(pcd.normals)  # 从法线读取颜色信息

    # 创建布尔索引（基于法线中的颜色值）
    idx_r = (normals[:, 0] > 0) & (normals[:, 1] == 0) & (normals[:, 2] == 0)
    idx_g = (normals[:, 1] > 0) & (normals[:, 0] == 0) & (normals[:, 2] == 0)
    idx_b = (normals[:, 2] > 0) & (normals[:, 0] == 0) & (normals[:, 1] == 0)

    # 创建三个点云对象
    pcd_R = o3d.geometry.PointCloud()
    pcd_R.points = o3d.utility.Vector3dVector(points[idx_r])
    pcd_R.normals = o3d.utility.Vector3dVector(normals[idx_r])

    pcd_G = o3d.geometry.PointCloud()
    pcd_G.points = o3d.utility.Vector3dVector(points[idx_g])
    pcd_G.normals = o3d.utility.Vector3dVector(normals[idx_g])

    pcd_B = o3d.geometry.PointCloud()
    pcd_B.points = o3d.utility.Vector3dVector(points[idx_b])
    pcd_B.normals = o3d.utility.Vector3dVector(normals[idx_b])

    return pcd_R, pcd_G, pcd_B

def pcd2np(pcd):
    """将Open3D PointCloud转换为numpy数组"""
    points = np.asarray(pcd.points)
    normals = np.asarray(pcd.normals) if pcd.has_normals() else None
    return points, normals

def np2pcd(points, normals=None):
    """从numpy数组创建Open3D PointCloud"""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    if normals is not None:
        pcd.normals = o3d.utility.Vector3dVector(normals)
    return pcd


def process_single_channel(pcd_pos, pcd_nor, channel_idx, x_min, x_max, y_min, y_max, width, height, show_progress=False):
    # 读取点云
    # pcd = o3d.io.read_point_cloud(file_path)
    pcd = np2pcd(pcd_pos, pcd_nor)

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

def get_bounding_box(pcd_r, pcd_g, pcd_b):
    # 读取三个单色点云
    # pcd_r = o3d.io.read_point_cloud("result_R.ply")
    # pcd_g = o3d.io.read_point_cloud("result_G.ply")
    # pcd_b = o3d.io.read_point_cloud("result_B.ply")
    
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

def ply2jpgfast(pcd_r, pcd_g, pcd_b, resolution_mul):
    mp.set_start_method('spawn')
    # 获取包围盒
    x_min, x_max, y_min, y_max = get_bounding_box(pcd_r, pcd_g, pcd_b)
    print(f"取整后XY包围盒范围: x=[{x_min}, {x_max}], y=[{y_min}, {y_max}]")
    
    # 构建二维图像，分辨率为包围盒尺寸的二倍
    width = (x_max - x_min + 1) * resolution_mul
    height = (y_max - y_min + 1) * resolution_mul

    # 转换点云为numpy数组
    points_r, normals_r = pcd2np(pcd_r)
    points_g, normals_g = pcd2np(pcd_g)
    points_b, normals_b = pcd2np(pcd_b)
    
    # 多进程处理三个通道
    with mp.Pool(3) as pool:
        results = pool.starmap(process_single_channel, [
            (points_r, normals_r, 0, x_min, x_max, y_min, y_max, width, height, False),
            (points_g, normals_g, 1, x_min, x_max, y_min, y_max, width, height, True),
            (points_b, normals_b, 2, x_min, x_max, y_min, y_max, width, height, False),
        ])

    # 合成最终图像
    img_r, img_g, img_b = results
    final_image = np.stack([img_r, img_g, img_b], axis=-1)
    final_image_uint16 = np.clip(final_image, 0, 65535).astype(np.uint16)
    # cv2.imwrite("output_ply2png_fast.png", final_image_uint16)
    return final_image_uint16


def pic8bit(img):
    # 转换为浮点型并归一化到0-1范围
    img_normalized = img.astype(np.float32)
    img_normalized = (img_normalized - img_normalized.min()) / (img_normalized.max() - img_normalized.min())
    # 缩放到0-255并转换为uint8
    img_8bit = (img_normalized * 255).astype(np.uint8)
    return img_8bit






if __name__ == '__main__':
    greeninfo(f"开始摇摇乐，输入文件夹{input_dir}，输出文件{outputname}")
    ply = raw2ply(input_dir, rot_180_for_raw)
    greeninfo(f"生成全局点云完成，开始拆分RGB点云")
    pcd_r, pcd_g, pcd_b = plyRGB23(ply)
    greeninfo(f"拆分RGB点云完成，开始采样，放大倍率{target_resolution_times}")
    image = ply2jpgfast(pcd_r, pcd_g, pcd_b, target_resolution_times)
    greeninfo(f"采样完成，将16bit色深图像存储为{outputname}")
    cv2.imwrite(outputname, image)
    greeninfo(f"存储完成，并存储8bit压缩预览图{outputname}.jpg")
    cv2.imwrite(f"{outputname}.jpg", pic8bit(image))
    greeninfo(f"全部完成！")







