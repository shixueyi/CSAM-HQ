import sys
sys.path.append("..")
from segment_anything import sam_model_registry
import torch 
from torch.utils.data import DataLoader
from dataset_crf import Endovis18Dataset, Endovis17Dataset, CataractsDataset
from model import Prototype_Prompt_Encoder, Learnable_Prototypes
from model_forward import model_forward_function
import argparse
import torch.nn.functional as F
from utils_crf import apply_crf
from utils import read_gt_endovis_masks, create_binary_masks, create_endovis_masks, eval_endovis
import os
import os.path as osp
import cv2
import numpy as np
import time

def count_parameters(model, name="Model"):
     """计算并打印模型的参数量"""
     total_params = sum(p.numel() for p in model.parameters())
     trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
     print(f"--- {name} Parameters ---")
     print(f"Total Params: {total_params:,}")
     print(f"Trainable Params: {trainable_params:,}")
     return total_params

# # # ================= NEW: 辅助函数 =================
def prob_to_logit(prob, eps=1e-6):
# #     """将概率 (0-1) 转换回 Logits (-inf 到 +inf)，防止数值不稳定性"""
    prob = np.clip(prob, eps, 1 - eps)
    return np.log(prob / (1 - prob))
# # # ================================================

print("======> Process Arguments")
parser = argparse.ArgumentParser()
# # # 【修改 2】加入 "Cataracts" 选项
parser.add_argument('--dataset', type=str, default="endovis_2018", choices=["endovis_2018", "endovis_2017", "Cataracts"], help='specify dataset')
parser.add_argument('--fold', type=int, default=0, choices=[0,1,2,3], help='specify fold number for endovis_2017 dataset')
parser.add_argument('--output_dir', type=str, default="./predictions", help='directory to save predicted masks')
args = parser.parse_args()


print("======> Set Parameters for Inference" )
dataset_name = args.dataset
fold = args.fold
thr = 0  # 默认阈值 (针对 Logits)
data_root_dir = f"../data/{dataset_name}"

# # # 【新增】定义默认分辨率 (Endovis 标准)，后面针对 Cataracts 修改
img_h = 1024
img_w = 1280

print("======> Load Dataset-Specific Parameters" )
if "18" in dataset_name:
    num_tokens = 2
    # Endovis 18 分辨率保持默认
    img_h, img_w = 1024, 1280
    dataset = Endovis18Dataset(data_root_dir = data_root_dir, 
                                mode = "val",
                                vit_mode = "h")
    # 注意检查这里的路径是否正确
    surgicalSAM_ckp = f"./work_dirs/{dataset_name}/model_ckp.pth"
    
    gt_endovis_masks = read_gt_endovis_masks(data_root_dir = data_root_dir,
                                            mode = "val")
    save_path_root = osp.join(args.output_dir, dataset_name)

elif "17" in dataset_name:
    num_tokens = 4
    # Endovis 17 分辨率保持默认
    img_h, img_w = 1024, 1280
    dataset = Endovis17Dataset(data_root_dir = data_root_dir, 
                                mode = "val",
                                fold = fold, 
                                vit_mode = "h",
                                version = 0)
    surgicalSAM_ckp = f"./work_dirs/{dataset_name}/{fold}/model_ckp.pth"
    
    gt_endovis_masks = read_gt_endovis_masks(data_root_dir = data_root_dir, 
                                             mode = "val", 
                                             fold = fold)
    save_path_root = osp.join(args.output_dir, dataset_name, f"fold{fold}")

# # # 【新增】Cataracts 配置块
elif "Cataracts" in dataset_name:
    num_tokens = 4
    # 【重要】修改分辨率为 Cataracts 的尺寸
    img_h, img_w = 768, 1024
    
    dataset = CataractsDataset(data_root_dir = data_root_dir, 
                                mode = "val",
                                fold = fold, 
                                vit_mode = "h",
                                version = 0)
    
# #     # 假设你的 checkpoint 保存在 work_dirs/Cataracts/0/model_ckp.pth
    surgicalSAM_ckp = f"./work_dirs/Cataracts/{fold}/model_ckp.pth"
    
    gt_endovis_masks = read_gt_endovis_masks(data_root_dir = data_root_dir, 
                                             mode = "val", 
                                             fold = fold)
    save_path_root = osp.join(args.output_dir, "Cataracts", f"fold{fold}")

# 检查 Checkpoint 是否存在，如果不存在尝试找 latest
if not osp.exists(surgicalSAM_ckp):
    alt_ckp = surgicalSAM_ckp.replace("model_ckp.pth", "latest_model.pth")
    if osp.exists(alt_ckp):
        print(f"Warning: model_ckp.pth not found, using {alt_ckp}")
        surgicalSAM_ckp = alt_ckp

dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)


print("======> Load SAM" )
sam_checkpoint = "../ckp/sam/sam_hq_vit_h.pth"
model_type = "vit_h_no_image_encoder"
sam_prompt_encoder, sam_decoder = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam_prompt_encoder.cuda()
sam_decoder.cuda()


print("======> Load Prototypes and Prototype-based Prompt Encoder" )
learnable_prototypes_model = Learnable_Prototypes(num_classes = 7, feat_dim = 256).cuda()
protoype_prompt_encoder =  Prototype_Prompt_Encoder(feat_dim = 256, 
                                                    hidden_dim_dense = 128, 
                                                    hidden_dim_sparse = 128, 
                                                    size = 64, 
                                                    num_tokens = num_tokens).cuda()

# # # 【新增】定义投影层 (因为训练时加了，推理时加载权重需要结构一致，虽然推理时不使用它)
# # # 注意：如果你的模型是在加入 clip_projector 之前训练的，这里不需要加。
# # # 如果是在加入之后训练的，必须加，否则 load_state_dict 会报错 key mismatch (strict=True时)。
# # # 这里为了安全，我们用 strict=False 加载权重，这样无论有没有投影层都能跑
clip_projector = torch.nn.Linear(512, 256).cuda()
            
if osp.exists(surgicalSAM_ckp):
    print(f"Loading checkpoint from: {surgicalSAM_ckp}")
    checkpoint = torch.load(surgicalSAM_ckp)
    # 使用 strict=False，兼容有无 clip_projector 的情况
    protoype_prompt_encoder.load_state_dict(checkpoint['prototype_prompt_encoder_state_dict'], strict=False)
    sam_decoder.load_state_dict(checkpoint['sam_decoder_state_dict'], strict=False)
    learnable_prototypes_model.load_state_dict(checkpoint['prototypes_state_dict'], strict=False)
else:
    print(f"Error: Checkpoint not found at {surgicalSAM_ckp}")
    sys.exit(1)

for model in [sam_prompt_encoder, sam_decoder, protoype_prompt_encoder, learnable_prototypes_model]:
    for param in model.parameters():
        param.requires_grad = False

# # # ================= 【关键修改】打印参数量 =================
print("\n" + "="*30)
print("STATISTICS: MODEL PARAMETERS")
total_all = 0
total_all += count_parameters(sam_prompt_encoder, "SAM Prompt Encoder")
total_all += count_parameters(sam_decoder, "SAM Decoder")
total_all += count_parameters(learnable_prototypes_model, "Learnable Prototypes")
total_all += count_parameters(protoype_prompt_encoder, "Prototype Prompt Encoder")
print(f"\n[Summary] Total Inference Parameters: {total_all:,}")
print("="*30 + "\n")

for name, param in sam_prompt_encoder.named_parameters():
    param.requires_grad = False
for name, param in sam_decoder.named_parameters():
    param.requires_grad = False
for name, param in protoype_prompt_encoder.named_parameters():
    param.requires_grad = False
for name, param in learnable_prototypes_model.named_parameters():
    param.requires_grad = False


print("======> Start Inference")
binary_masks = dict()
protoype_prompt_encoder.eval()
sam_decoder.eval()
learnable_prototypes_model.eval()

# # # 计数器，用于防止打印过多 log
total_model_time = 0.0
total_crf_time = 0.0
total_images = 0
log_counter = 0

with torch.no_grad():
    prototypes = learnable_prototypes_model()

    for sam_feats, mask_names, cls_ids, _, _, images in dataloader: 
        
        sam_feats = sam_feats.cuda().float()
        cls_ids = cls_ids.cuda()  

# #         # ===================================================
# #         # 计时 1：纯神经网络推理 (CSAM-HQ 本体)
# #         # ===================================================
        torch.cuda.synchronize()
        t0 = time.time()  
                
        preds , preds_quality = model_forward_function(protoype_prompt_encoder, sam_prompt_encoder, sam_decoder, sam_feats, prototypes, cls_ids)    
 
        original_h, original_w = images.shape[1], images.shape[2]
        
        if len(preds.shape) == 3:
             preds = preds.unsqueeze(1)
             
        if preds.shape[-2:] != (original_h, original_w):
            preds = F.interpolate(preds, size=(original_h, original_w), mode='bilinear', align_corners=False)

        torch.cuda.synchronize()
        t1 = time.time()
# #         # ===================================================

# #         # 准备 CRF 所需的 Numpy 数据 (不计入算法核心耗时)
        current_img_np = images[0].numpy().astype(np.uint8)
        
# #         # ===================================================
        # 计时 2：CRF 拓扑优化后处理
# #         # ===================================================
        t2 = time.time()
        
        if current_img_np.max() > 0:
            if log_counter < 5:
                print(f"[DEBUG] Running CRF for {mask_names[0]}")
        
        refined_prob = apply_crf(preds[0], current_img_np)
        refined_logits = prob_to_logit(refined_prob)
        final_pred = torch.from_numpy(refined_logits).cuda().unsqueeze(0).unsqueeze(0)
        
# #         # 如果你的 CRF 是纯 CPU (numpy)，这里不需要 synchronize
# #         # 如果你的 CRF 是 GPU 版，保险起见加上：
# #         # torch.cuda.synchronize() 
        t3 = time.time()
# #         # ===================================================
            
        log_counter += 1

# #         # 后处理与字典写入 (绝对不能计入任何推理耗时)
        binary_masks = create_binary_masks(binary_masks, final_pred, preds_quality, mask_names, thr)

# #         # 累计时间
        total_model_time += (t1 - t0)
        total_crf_time += (t3 - t2)
        total_images += 1

# # # === 打印统计结果 ===
avg_model_time = total_model_time / total_images
avg_crf_time = total_crf_time / total_images
avg_total_time = avg_model_time + avg_crf_time
fps_model = 1.0 / avg_model_time if avg_model_time > 0 else 0

print("\n" + "="*30)
print("STATISTICS: INFERENCE SPEED (Corrected)")
print(f"Total Images Processed: {total_images}")
print(f"Average MODEL Time per Image: {avg_model_time * 1000:.2f} ms (Pure Network FPS: {fps_model:.2f})")
print(f"Average CRF Time per Image: {avg_crf_time * 1000:.2f} ms")
print(f"Average TOTAL Time (Model + CRF): {avg_total_time * 1000:.2f} ms")
print("="*30 + "\n")

def clean_mask_data(item):
# #     """
# #     1. 穿透 List/Dict/Tuple
# #     2. 压缩维度 (Squeeze) 到 2D
# #     3. 确保返回的是 Tensor (CPU版)，以满足 utils.py 的要求
# #     """
    if isinstance(item, dict):
        return {k: clean_mask_data(v) for k, v in item.items()}
    if isinstance(item, (list, tuple)):
        return [clean_mask_data(v) for v in item]
    
    if isinstance(item, np.ndarray):
        item = torch.from_numpy(item)
    if isinstance(item, torch.Tensor):
        item = item.detach().cpu()
        if item.ndim > 2:
            item = item.squeeze()
        while item.ndim > 2:
            item = item[0]
        return item
    return item

print("======> Sanity checking and fixing mask dimensions (Tensor Mode)...")

# # # 执行清洗
binary_masks = clean_mask_data(binary_masks)

print(f"======> Dimensions fixed. Converting to Endovis format using size ({img_h}, {img_w})...")
# # # 【修改 3】使用动态变量 img_h, img_w 而不是写死
endovis_masks = create_endovis_masks(binary_masks, img_h, img_w)
endovis_results = eval_endovis(endovis_masks, gt_endovis_masks)

print("======> Evaluation Results:")
print(endovis_results)

# # # ==========================================
# # # 保存预测图像
# # # ==========================================
print(f"======> Saving Predicted Masks to {save_path_root}")

color_map = {
    1: [0, 255, 0],   
    2: [0, 0, 255],   
    3: [255, 0, 0],   
    4: [0, 255, 255], 
    5: [255, 0, 255], 
    6: [255, 255, 0], 
    7: [128, 128, 128]
 }

# # os.makedirs(save_path_root, exist_ok=True) 

for file_name, mask in endovis_masks.items():
    h, w = mask.shape
    color_img = np.zeros((h, w, 3), dtype=np.uint8)

    for cls_id, color in color_map.items():
        color_img[mask == cls_id] = color

    full_save_path = osp.join(save_path_root, file_name)
    
    if not osp.exists(osp.dirname(full_save_path)):
        os.makedirs(osp.dirname(full_save_path))

    cv2.imwrite(full_save_path, color_img)

print("======> All Predictions Saved.")
# import sys
# sys.path.append("..")
# from segment_anything import sam_model_registry
# import torch 
# from torch.utils.data import DataLoader
# # 【修改 1】导入 CataractsDataset
# from dataset_crf import Endovis18Dataset, Endovis17Dataset, CataractsDataset
# from model import Prototype_Prompt_Encoder, Learnable_Prototypes
# from model_forward import model_forward_function
# import argparse
# import torch.nn.functional as F
# from utils_crf import apply_crf
# from utils import read_gt_endovis_masks, create_binary_masks, create_endovis_masks, eval_endovis
# import os
# import os.path as osp
# import cv2
# import numpy as np
# import time

# def count_parameters(model, name="Model"):
#     """计算并打印模型的参数量"""
#     total_params = sum(p.numel() for p in model.parameters())
#     trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
#     print(f"--- {name} Parameters ---")
#     print(f"Total Params: {total_params:,}")
#     print(f"Trainable Params: {trainable_params:,}")
#     return total_params

# # ================= NEW: 辅助函数 =================
# def prob_to_logit(prob, eps=1e-6):
#     """将概率 (0-1) 转换回 Logits (-inf 到 +inf)，防止数值不稳定性"""
#     prob = np.clip(prob, eps, 1 - eps)
#     return np.log(prob / (1 - prob))
# # ================================================

# print("======> Process Arguments")
# parser = argparse.ArgumentParser()
# # 【修改 2】加入 "Cataracts" 选项
# parser.add_argument('--dataset', type=str, default="endovis_2018", choices=["endovis_2018", "endovis_2017", "Cataracts"], help='specify dataset')
# parser.add_argument('--fold', type=int, default=0, choices=[0,1,2,3], help='specify fold number for endovis_2017 dataset')
# parser.add_argument('--output_dir', type=str, default="./predictions", help='directory to save predicted masks')
# args = parser.parse_args()


# print("======> Set Parameters for Inference" )
# dataset_name = args.dataset
# fold = args.fold
# thr = 0  # 默认阈值 (针对 Logits)
# data_root_dir = f"../data/{dataset_name}"

# # 【新增】定义默认分辨率 (Endovis 标准)，后面针对 Cataracts 修改
# img_h = 1024
# img_w = 1280

# # 【CPU 修改 1】统一定义设备为 CPU
# device = torch.device("cpu")

# print("======> Load Dataset-Specific Parameters" )
# if "18" in dataset_name:
#     num_tokens = 2
#     img_h, img_w = 1024, 1280
#     dataset = Endovis18Dataset(data_root_dir = data_root_dir, 
#                                 mode = "val",
#                                 vit_mode = "h")
#     surgicalSAM_ckp = f"/mnt/data2/shixy/SurgicalSAM-HQ/surgicalSAM/work_dirs/endovis_2018/model_ckp.pth"
    
#     gt_endovis_masks = read_gt_endovis_masks(data_root_dir = data_root_dir,
#                                             mode = "val")
#     save_path_root = osp.join(args.output_dir, dataset_name)

# elif "17" in dataset_name:
#     num_tokens = 4
#     img_h, img_w = 1024, 1280
#     dataset = Endovis17Dataset(data_root_dir = data_root_dir, 
#                                 mode = "val",
#                                 fold = fold, 
#                                 vit_mode = "h",
#                                 version = 0)
#     surgicalSAM_ckp = f"./work_dirs/{dataset_name}/{fold}/model_ckp.pth"
    
#     gt_endovis_masks = read_gt_endovis_masks(data_root_dir = data_root_dir, 
#                                              mode = "val", 
#                                              fold = fold)
#     save_path_root = osp.join(args.output_dir, dataset_name, f"fold{fold}")

# # 【新增】Cataracts 配置块
# elif "Cataracts" in dataset_name:
#     num_tokens = 4
#     # 【重要】修改分辨率为 Cataracts 的尺寸
#     img_h, img_w = 768, 1024
    
#     dataset = CataractsDataset(data_root_dir = data_root_dir, 
#                                 mode = "val",
#                                 fold = fold, 
#                                 vit_mode = "h",
#                                 version = 0)
    
#     surgicalSAM_ckp = f"./work_dirs/Cataracts/{fold}/model_ckp.pth"
    
#     gt_endovis_masks = read_gt_endovis_masks(data_root_dir = data_root_dir, 
#                                              mode = "val", 
#                                              fold = fold)
#     save_path_root = osp.join(args.output_dir, "Cataracts", f"fold{fold}")

# if not osp.exists(surgicalSAM_ckp):
#     alt_ckp = surgicalSAM_ckp.replace("model_ckp.pth", "latest_model.pth")
#     if osp.exists(alt_ckp):
#         print(f"Warning: model_ckp.pth not found, using {alt_ckp}")
#         surgicalSAM_ckp = alt_ckp

# dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)

# print("======> Load SAM" )
# sam_checkpoint = "../ckp/sam/sam_hq_vit_h.pth"
# model_type = "vit_h_no_image_encoder"
# sam_prompt_encoder, sam_decoder = sam_model_registry[model_type](checkpoint=sam_checkpoint)

# # 【CPU 修改 2】将模型加载到 CPU
# sam_prompt_encoder.to(device)
# sam_decoder.to(device)

# print("======> Load Prototypes and Prototype-based Prompt Encoder" )
# learnable_prototypes_model = Learnable_Prototypes(num_classes = 7, feat_dim = 256).to(device)
# protoype_prompt_encoder =  Prototype_Prompt_Encoder(feat_dim = 256, 
#                                                     hidden_dim_dense = 128, 
#                                                     hidden_dim_sparse = 128, 
#                                                     size = 64, 
#                                                     num_tokens = num_tokens).to(device)

# # 【CPU 修改 3】投射层加载到 CPU
# clip_projector = torch.nn.Linear(512, 256).to(device)
            
# if osp.exists(surgicalSAM_ckp):
#     print(f"Loading checkpoint from: {surgicalSAM_ckp}")
#     # 【CPU 修改 4】加上 map_location='cpu' 确保 GPU 保存的模型能加载到 CPU
#     checkpoint = torch.load(surgicalSAM_ckp, map_location=device)
#     protoype_prompt_encoder.load_state_dict(checkpoint['prototype_prompt_encoder_state_dict'], strict=False)
#     sam_decoder.load_state_dict(checkpoint['sam_decoder_state_dict'], strict=False)
#     learnable_prototypes_model.load_state_dict(checkpoint['prototypes_state_dict'], strict=False)
# else:
#     print(f"Error: Checkpoint not found at {surgicalSAM_ckp}")
#     sys.exit(1)

# # 冻结参数
# for model in [sam_prompt_encoder, sam_decoder, protoype_prompt_encoder, learnable_prototypes_model]:
#     for param in model.parameters():
#         param.requires_grad = False

# print("\n" + "="*30)
# print("STATISTICS: MODEL PARAMETERS")
# total_all = 0
# total_all += count_parameters(sam_prompt_encoder, "SAM Prompt Encoder")
# total_all += count_parameters(sam_decoder, "SAM Decoder")
# total_all += count_parameters(learnable_prototypes_model, "Learnable Prototypes")
# total_all += count_parameters(protoype_prompt_encoder, "Prototype Prompt Encoder")
# print(f"\n[Summary] Total Inference Parameters: {total_all:,}")
# print("="*30 + "\n")

# print("======> Start Inference (Running on CPU)")
# binary_masks = dict()
# protoype_prompt_encoder.eval()
# sam_decoder.eval()
# learnable_prototypes_model.eval()

# total_model_time = 0.0
# total_crf_time = 0.0
# total_images = 0
# log_counter = 0

# with torch.no_grad():
#     prototypes = learnable_prototypes_model()

#     for sam_feats, mask_names, cls_ids, _, _, images in dataloader: 
        
#         # 【CPU 修改 5】移除 .cuda()
#         sam_feats = sam_feats.to(device).float()
#         cls_ids = cls_ids.to(device)  

#         # ===================================================
#         # 计时 1：纯神经网络推理 (CSAM-HQ 本体)
#         # ===================================================
#         # 【CPU 修改 6】CPU 推理是自带同步的，删除 torch.cuda.synchronize()
#         t0 = time.time()  
                
#         preds , preds_quality = model_forward_function(protoype_prompt_encoder, sam_prompt_encoder, sam_decoder, sam_feats, prototypes, cls_ids)    
 
#         original_h, original_w = images.shape[1], images.shape[2]
        
#         if len(preds.shape) == 3:
#              preds = preds.unsqueeze(1)
             
#         if preds.shape[-2:] != (original_h, original_w):
#             preds = F.interpolate(preds, size=(original_h, original_w), mode='bilinear', align_corners=False)

#         # 【CPU 修改 6】删除 torch.cuda.synchronize()
#         t1 = time.time()
#         # ===================================================

#         current_img_np = images[0].numpy().astype(np.uint8)
        
#         # ===================================================
#         # 计时 2：CRF 拓扑优化后处理
#         # ===================================================
#         t2 = time.time()
        
#         if current_img_np.max() > 0:
#             if log_counter < 5:
#                 print(f"[DEBUG] Running CRF for {mask_names[0]}")
        
#         refined_prob = apply_crf(preds[0], current_img_np)
#         refined_logits = prob_to_logit(refined_prob)
#         # 【CPU 修改 7】移除 .cuda()，Tensor 留在 CPU 上
#         final_pred = torch.from_numpy(refined_logits).unsqueeze(0).unsqueeze(0)
        
#         t3 = time.time()
#         # ===================================================
            
#         log_counter += 1

#         binary_masks = create_binary_masks(binary_masks, final_pred, preds_quality, mask_names, thr)

#         total_model_time += (t1 - t0)
#         total_crf_time += (t3 - t2)
#         total_images += 1

# # === 打印统计结果 ===
# avg_model_time = total_model_time / total_images
# avg_crf_time = total_crf_time / total_images
# avg_total_time = avg_model_time + avg_crf_time
# fps_model = 1.0 / avg_model_time if avg_model_time > 0 else 0

# print("\n" + "="*30)
# print("STATISTICS: CPU INFERENCE SPEED")
# print(f"Total Images Processed: {total_images}")
# print(f"Average MODEL Time per Image: {avg_model_time * 1000:.2f} ms (Pure Network FPS: {fps_model:.2f})")
# print(f"Average CRF Time per Image: {avg_crf_time * 1000:.2f} ms")
# print(f"Average TOTAL Time (Model + CRF): {avg_total_time * 1000:.2f} ms")
# print("="*30 + "\n")

# def clean_mask_data(item):
#     """
#     1. 穿透 List/Dict/Tuple
#     2. 压缩维度 (Squeeze) 到 2D
#     3. 确保返回的是 Tensor (CPU版)
#     """
#     if isinstance(item, dict):
#         return {k: clean_mask_data(v) for k, v in item.items()}
#     if isinstance(item, (list, tuple)):
#         return [clean_mask_data(v) for v in item]
    
#     if isinstance(item, np.ndarray):
#         item = torch.from_numpy(item)
#     if isinstance(item, torch.Tensor):
#         item = item.detach().cpu()
#         if item.ndim > 2:
#             item = item.squeeze()
#         while item.ndim > 2:
#             item = item[0]
#         return item
#     return item

# print("======> Sanity checking and fixing mask dimensions (Tensor Mode)...")

# binary_masks = clean_mask_data(binary_masks)

# print(f"======> Dimensions fixed. Converting to Endovis format using size ({img_h}, {img_w})...")
# endovis_masks = create_endovis_masks(binary_masks, img_h, img_w)
# endovis_results = eval_endovis(endovis_masks, gt_endovis_masks)

# print("======> Evaluation Results:")
# print(endovis_results)

# # ==========================================
# # 保存预测图像
# # ==========================================
# print(f"======> Saving Predicted Masks to {save_path_root}")

# color_map = {
#     1: [0, 255, 0],   
#     2: [0, 0, 255],   
#     3: [255, 0, 0],   
#     4: [0, 255, 255], 
#     5: [255, 0, 255], 
#     6: [255, 255, 0], 
#     7: [128, 128, 128]
# }

# os.makedirs(save_path_root, exist_ok=True) 

# for file_name, mask in endovis_masks.items():
#     h, w = mask.shape
#     color_img = np.zeros((h, w, 3), dtype=np.uint8)

#     for cls_id, color in color_map.items():
#         color_img[mask == cls_id] = color

#     full_save_path = osp.join(save_path_root, file_name)
    
#     if not osp.exists(osp.dirname(full_save_path)):
#         os.makedirs(osp.dirname(full_save_path))

#     cv2.imwrite(full_save_path, color_img)

# print("======> All Predictions Saved.")