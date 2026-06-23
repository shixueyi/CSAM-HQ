import cv2
import numpy as np
from scipy.ndimage import binary_erosion, generate_binary_structure, distance_transform_edt

# 1. 这里放入你计算 HD 的核心函数（方法一的代码）
def compute_hd_and_hd95(mask_gt, mask_pred, voxel_spacing=None):
    mask_gt = np.atleast_1d(mask_gt.astype(bool))
    mask_pred = np.atleast_1d(mask_pred.astype(bool))
    if not np.any(mask_gt) or not np.any(mask_pred):
        return np.nan, np.nan
    struct = generate_binary_structure(mask_gt.ndim, 1)
    gt_surface = mask_gt ^ binary_erosion(mask_gt, structure=struct)
    pred_surface = mask_pred ^ binary_erosion(mask_pred, structure=struct)
    if not np.any(gt_surface) or not np.any(pred_surface):
        return np.nan, np.nan
    dt_gt = distance_transform_edt(~gt_surface, sampling=voxel_spacing)
    dt_pred = distance_transform_edt(~pred_surface, sampling=voxel_spacing)
    d_pred_to_gt = dt_gt[pred_surface]
    d_gt_to_pred = dt_pred[gt_surface]
    all_distances = np.concatenate([d_pred_to_gt, d_gt_to_pred])
    hd = np.max(all_distances)
    hd95 = np.percentile(all_distances, 95)
    return hd, hd95

# ==================== 2. 读取你的真实图片文件 ====================
# 【请把下面的路径替换为你电脑上真实的图片路径】
gt_path = "/mnt/data2/shixy/SurgicalSAM-HQ/data/endovis_2017/0/binary_annotations/seq4/00032_class3.png" 
pred_path = "/mnt/data2/shixy/SurgicalSAM-HQ/surgicalSAM/output_masks/endovis_2017/seq4/00032.png"

# 使用 cv2.IMREAD_GRAYSCALE 以灰度图模式读取（Mask通常是单通道灰度图）
img_gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
img_pred = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)

# 确保图片读取成功
if img_gt is None or img_pred is None:
    raise ValueError("图片读取失败，请检查文件路径是否正确！")

# 关键步骤：将图片转为布尔类型（True/False）或 0/1 二值图
# 假设你的 Mask 中目标区域是 255（白色），背景是 0（黑色）
mask_gt = img_gt > 0
mask_pred = img_pred > 0

# ==================== 3. 调用函数计算 ====================
hd, hd95 = compute_hd_and_hd95(mask_gt, mask_pred)

print(f"真实图片计算结果：")
print(f"Hausdorff Distance (HD): {hd:.4f} 像素")
print(f"95% Hausdorff Distance (HD95): {hd95:.4f} 像素")