import sys
sys.path.append("..")
from segment_anything import sam_model_registry
import torch 
from torch.utils.data import DataLoader
from dataset import Endovis18Dataset, Endovis17Dataset
from model import Prototype_Prompt_Encoder, Learnable_Prototypes
from model_forward import model_forward_function
import argparse
from utils import read_gt_endovis_masks, create_binary_masks, create_endovis_masks, eval_endovis
import time
# 【新增】引入保存图片所需的标准库
import os
import cv2
import numpy as np


print("======> Process Arguments")
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default="endovis_2018", choices=["endovis_2018", "endovis_2017"], help='specify dataset')
parser.add_argument('--fold', type=int, default=0, choices=[0,1,2,3], help='specify fold number for endovis_2017 dataset')
# 【新增】添加一个参数用于指定预测 Mask 的保存根目录
parser.add_argument('--save_dir', type=str, default="./output_masks", help='directory to save predicted masks')
args = parser.parse_args()


print("======> Set Parameters for Inference" )
dataset_name = args.dataset
fold = args.fold
thr = 0
data_root_dir = f"../data/{dataset_name}"

log_counter = 0 

total_inference_time = 0.0
total_images = 0


print("======> Load Dataset-Specific Parameters" )
if "18" in dataset_name:
    num_tokens = 2
    dataset = Endovis18Dataset(data_root_dir = data_root_dir, 
                                mode = "val",
                                vit_mode = "h")
    surgicalSAM_ckp = f"../ckp/surgical_sam/{dataset_name}/model_ckp.pth"
    
    gt_endovis_masks = read_gt_endovis_masks(data_root_dir = data_root_dir,
                                            mode = "val")

elif "17" in dataset_name:
    num_tokens = 4
    dataset = Endovis17Dataset(data_root_dir = data_root_dir, 
                                mode = "val",
                                fold = fold, 
                                vit_mode = "h",
                                version = 0)
    surgicalSAM_ckp = f"/mnt/data2/shixy/SurgicalSAM-HQ/surgicalSAM/work_dirs/endovis_2017/2/model_ckp82.pth"
    
    gt_endovis_masks = read_gt_endovis_masks(data_root_dir = data_root_dir,
                                            mode = "val",
                                            fold = fold)
    
dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=4)


print("======> Load SAM" )
sam_checkpoint = "/mnt/data2/shixy/SurgicalSAM-HQ/ckp/sam/sam_hq_vit_h.pth"
model_type = "vit_h_no_image_encoder"
sam_prompt_encoder, sam_decoder = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam_prompt_encoder.cuda()
sam_decoder.cuda()


print("======> Load Prototypes and Prototype-based Prompt Encoder" )
# define the models
learnable_prototypes_model = Learnable_Prototypes(num_classes = 7, feat_dim = 256).cuda()
protoype_prompt_encoder = Prototype_Prompt_Encoder(feat_dim = 256, 
                                                    hidden_dim_dense = 128, 
                                                    hidden_dim_sparse = 128, 
                                                    size = 64, 
                                                    num_tokens = num_tokens).cuda()
            
# load the weight for prototype-based prompt encoder, mask decoder, and prototypes
checkpoint = torch.load(surgicalSAM_ckp)
protoype_prompt_encoder.load_state_dict(checkpoint['prototype_prompt_encoder_state_dict'])
sam_decoder.load_state_dict(checkpoint['sam_decoder_state_dict'])
learnable_prototypes_model.load_state_dict(checkpoint['prototypes_state_dict'])

# set requires_grad to False to the whole model 
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

with torch.no_grad():
    prototypes = learnable_prototypes_model()

    for sam_feats, mask_names, cls_ids, _, _ in dataloader: 
        
        sam_feats = sam_feats.cuda()
        cls_ids = cls_ids.cuda()    

        torch.cuda.synchronize()
        start_time = time.time() 
                
        preds , preds_quality = model_forward_function(protoype_prompt_encoder, sam_prompt_encoder, sam_decoder, sam_feats, prototypes, cls_ids)    
 
        log_counter += 1

        binary_masks = create_binary_masks(binary_masks, preds, preds_quality, mask_names, thr)

        torch.cuda.synchronize()
        end_time = time.time()
        
        # 累计时间与帧数
        total_inference_time += (end_time - start_time)
        total_images += 1

avg_time_per_image = total_inference_time / total_images
fps = 1.0 / avg_time_per_image

print("\n" + "="*30)
print("STATISTICS: INFERENCE SPEED")
print(f"Total Images Processed: {total_images}")
print(f"Total Inference Time: {total_inference_time:.2f} seconds")
print(f"Average Time per Image: {avg_time_per_image * 1000:.2f} ms")
print(f"FPS (Frames Per Second): {fps:.2f}")
print("="*30 + "\n")

print("======> Sanity checking and fixing mask dimensions (Tensor Mode)...")

# 此时 endovis_masks 内部的矩阵尺寸已被恢复为 1024x1280
endovis_masks = create_endovis_masks(binary_masks, 1024, 1280)


# ==================== 【新增：批量保存预测 Mask 图片】 ====================
print("======> Saving predicted mask images...")
# 按数据集名称分子文件夹，避免混淆
final_save_dir = os.path.join(args.save_dir, dataset_name)
os.makedirs(final_save_dir, exist_ok=True)

for mask_name, mask_arr in endovis_masks.items():
    # 如果 mask_arr 依然是 PyTorch Tensor，先转成 numpy 数组
    if hasattr(mask_arr, 'cpu'):
        mask_arr = mask_arr.cpu().numpy()
        
    # 构建最终的保存路径
    save_path = os.path.join(final_save_dir, mask_name)
    
    # 医疗数据集的 mask_name 可能包含子目录（如 "seq_1/frame000"），自动创建对应的子文件夹
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # 确保文件名带有 .png 后缀
    if not save_path.lower().endswith(('.png', '.jpg', '.jpeg')):
        save_path += '.png'
        
    # 核心转换：只要值大于 0（有工具的地方）一律转为 255（纯白色），其余为 0（纯黑色）
    # 这样保存出的黑白二值图，可以百分百完美兼容 HD95 的计算
    binary_img = (mask_arr > 0).astype(np.uint8) * 255
    
    # 保存图片
    cv2.imwrite(save_path, binary_img)

print(f"======> Successfully saved {len(endovis_masks)} masks to: {final_save_dir}")
# =========================================================================


endovis_results = eval_endovis(endovis_masks, gt_endovis_masks)
print(endovis_results)