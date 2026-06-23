import sys
sys.path.append("..")
import os
import os.path as osp 
import random 
import argparse
import numpy as np 
import torch 
import torch.nn.functional as F
from torch.utils.data import DataLoader
from dataset import Endovis18Dataset, Endovis17Dataset, CataractsDataset
from segment_anything import sam_model_registry
from model import Learnable_Prototypes, Prototype_Prompt_Encoder
from utils import print_log, create_binary_masks, create_endovis_masks, eval_endovis, read_gt_endovis_masks
from model_forward import model_forward_function
from loss import DiceLoss
from pytorch_metric_learning import losses

print("======> Process Arguments")
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default="endovis_2017", choices=["endovis_2018", "endovis_2017", "Cataracts"], help='specify dataset')
parser.add_argument('--fold', type=int, default=0, choices=[0,1,2,3], help='specify fold number for endovis_2017/Cataracts dataset')
args = parser.parse_args()

print("======> Set Parameters for Training" )
dataset_name = args.dataset
fold = args.fold
thr = 0
seed = 666  
data_root_dir = f"../data/{dataset_name}"
batch_size = 32
vit_mode = "h"

random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(seed)

print("======> Load Dataset-Specific Parameters" )
num_classes = 7 
img_h, img_w = 1024, 1280 
# 【修正点】：根据数据集设置不同的 Embedding 维度
embed_dim = 256 # 默认 SAM 特征维度

if "18" in dataset_name:
    num_tokens = 2
    num_classes = 1
    val_dataset = Endovis18Dataset(data_root_dir=data_root_dir, mode="val", vit_mode="h")
    gt_endovis_masks = read_gt_endovis_masks(data_root_dir=data_root_dir, mode="val")
    num_epochs = 500
    lr = 0.001
    save_dir = "./work_dirs/endovis_2018/"

elif "17" in dataset_name:
    num_tokens = 4
    num_classes = 7
    val_dataset = Endovis17Dataset(data_root_dir=data_root_dir, mode="val", fold=fold, vit_mode="h", version=0)
    gt_endovis_masks = read_gt_endovis_masks(data_root_dir=data_root_dir, mode="val", fold=fold)
    num_epochs = 2000
    lr = 0.0001
    save_dir = f"./work_dirs/endovis_2017/{fold}"

elif "Cataracts" in dataset_name:
    num_tokens = 4
    num_classes = 7
    embed_dim = 512 # 【关键】：假设白内障数据集用的是 512 维 CLIP 特征
    img_h, img_w = 768, 1024 
    val_dataset = CataractsDataset(data_root_dir=data_root_dir, mode="val", fold=fold, vit_mode="h", version=0)
    gt_endovis_masks = read_gt_endovis_masks(data_root_dir=data_root_dir, mode="val", fold=fold)
    num_epochs = 1000
    lr = 0.0001
    save_dir = f"./work_dirs/Cataracts/{fold}"

val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

# 【方案 A】：定义类别权重 Tensor
if "17" in dataset_name:
    # 给第 2 类 (PF) 和第 7 类 (UP) 设置 2.5 权重，其余为 1.0
    class_weight_list = [1.0, 2.5, 1.0, 1.0, 1.0, 1.0, 2.5]
else:
    class_weight_list = [1.0] * num_classes
class_weights_tensor = torch.tensor(class_weight_list).cuda()

print("======> Load SAM" )
sam_checkpoint = "../ckp/sam/sam_hq_vit_h.pth"
model_type = "vit_h_no_image_encoder"
sam_prompt_encoder, sam_decoder = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam_prompt_encoder.cuda()
sam_decoder.cuda()

for name, param in sam_prompt_encoder.named_parameters(): param.requires_grad = False
for name, param in sam_decoder.named_parameters(): param.requires_grad = True

print("======> Load Prototypes and Prototype-based Prompt Encoder" )
learnable_prototypes_model = Learnable_Prototypes(num_classes=num_classes, feat_dim=256).cuda()

# 【修正点】：使用动态 embed_dim，解决 32x256 和 512x256 不匹配报错
clip_projector = torch.nn.Linear(embed_dim, 256).cuda()

protoype_prompt_encoder = Prototype_Prompt_Encoder(feat_dim=256, hidden_dim_dense=128, hidden_dim_sparse=128, size=64, num_tokens=num_tokens).cuda()
 
with open(sam_checkpoint, "rb") as f:
    state_dict = torch.load(f)
    sam_pn_embeddings_weight = {k.split("prompt_encoder.point_embeddings.")[-1]: v for k, v in state_dict.items() if k.startswith("prompt_encoder.point_embeddings") and ("0" in k or "1" in k)}
    sam_pn_embeddings_weight_ckp = {"0.weight": torch.concat([sam_pn_embeddings_weight['0.weight'] for _ in range(num_tokens)], dim=0),
                                    "1.weight": torch.concat([sam_pn_embeddings_weight['1.weight'] for _ in range(num_tokens)], dim=0)}
    protoype_prompt_encoder.pn_cls_embeddings.load_state_dict(sam_pn_embeddings_weight_ckp)

for name, param in learnable_prototypes_model.named_parameters(): param.requires_grad = True
for name, param in protoype_prompt_encoder.named_parameters():
    if "pn_cls_embeddings" in name: param.requires_grad = False
    else: param.requires_grad = True
              
print("======> Define Optmiser and Loss")
seg_loss_model = DiceLoss().cuda()
contrastive_loss_model = losses.NTXentLoss(temperature=0.07).cuda()
# 【核心新增】：MSE 用于校准高质量分数
iou_loss_model = torch.nn.MSELoss().cuda() 

optimiser = torch.optim.Adam([
            {'params': learnable_prototypes_model.parameters()},
            {'params': protoype_prompt_encoder.parameters()},
            {'params': sam_decoder.parameters()},
            {'params': clip_projector.parameters()}
        ], lr=lr, weight_decay=0.0001)

os.makedirs(save_dir, exist_ok=True) 
log_file = osp.join(save_dir, "log.txt")
print_log(str(args), log_file)

print("======> Checking for checkpoints to resume" )
start_epoch = 0 
best_challenge_iou_val = -100.0
resume_ckp_path = osp.join(save_dir, 'model_ckp.pth')

if osp.exists(resume_ckp_path):
    print_log(f"恢复训练自: {resume_ckp_path}", log_file)
    checkpoint = torch.load(resume_ckp_path, map_location='cuda')
    protoype_prompt_encoder.load_state_dict(checkpoint['prototype_prompt_encoder_state_dict'])
    sam_decoder.load_state_dict(checkpoint['sam_decoder_state_dict'])
    learnable_prototypes_model.load_state_dict(checkpoint['prototypes_state_dict'])
    if 'optimizer_state_dict' in checkpoint: optimiser.load_state_dict(checkpoint['optimizer_state_dict'])
    if 'best_score' in checkpoint: best_challenge_iou_val = checkpoint['best_score']
    if 'epoch' in checkpoint: start_epoch = checkpoint['epoch'] + 1

print("======> Start Training and Validation" )

for epoch in range(start_epoch, num_epochs):    
    if epoch % 2 == 0 : version = 0 
    else: version = int((epoch % 80 + 1)/2)
    
    if "18" in dataset_name:
        train_dataset = Endovis18Dataset(data_root_dir=data_root_dir, mode="train", vit_mode="h", version=version)
    elif "17" in dataset_name:
        train_dataset = Endovis17Dataset(data_root_dir=data_root_dir, mode="train", fold=fold, vit_mode="h", version=version)
    elif "Cataracts" in dataset_name:
        train_dataset = CataractsDataset(data_root_dir=data_root_dir, mode="train", fold=fold, vit_mode="h", version=0)
        
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    
    protoype_prompt_encoder.train()
    sam_decoder.train()
    learnable_prototypes_model.train()

    for sam_feats, _, cls_ids, masks, class_embeddings in train_dataloader: 
        
        norms = torch.norm(class_embeddings, p=2, dim=1)
        valid_mask = (norms > 0.1) & (~torch.isnan(norms)) # 【提高过滤阈值】
        
        if not valid_mask.any(): continue

        sam_feats = sam_feats[valid_mask].cuda().float()
        cls_ids = cls_ids[valid_mask].cuda()
        masks = masks[valid_mask].cuda()
        class_embeddings = class_embeddings[valid_mask].cuda().float()
        
        # 【方案 A】：为当前 Batch 映射类别权重
        batch_weights = class_weights_tensor[cls_ids - 1]

        sam_feats = sam_feats.permute(0, 2, 3, 1)
        class_embeddings = clip_projector(class_embeddings)
        class_embeddings = class_embeddings / class_embeddings.norm(dim=-1, keepdim=True)
            
        prototypes = learnable_prototypes_model()
        
        # 获取三个输出
        preds_final, preds_base, preds_quality = model_forward_function(protoype_prompt_encoder, sam_prompt_encoder, sam_decoder, sam_feats, prototypes, cls_ids)    
        
        # 尺寸对齐
        if len(preds_final.shape) == 3: preds_final = preds_final.unsqueeze(1)
        if len(preds_base.shape) == 3: preds_base = preds_base.unsqueeze(1)
        if len(masks.shape) == 3: masks = masks.unsqueeze(1)
        gt_mask = masks / 255.0

        if preds_final.shape[-2:] != masks.shape[-2:]:
            preds_final = F.interpolate(preds_final, size=masks.shape[-2:], mode='bilinear', align_corners=False)
            preds_base = F.interpolate(preds_base, size=masks.shape[-2:], mode='bilinear', align_corners=False)

        # 【方案 A】：计算带权重的 Dice 损失
        loss_seg_final_raw = seg_loss_model(preds_final, gt_mask, reduction_override='none')
        loss_seg_final = (loss_seg_final_raw * batch_weights).mean()
        
        loss_seg_base_raw = seg_loss_model(preds_base, gt_mask, reduction_override='none')
        loss_seg_base = (loss_seg_base_raw * batch_weights).mean()

        # 计算对比损失 (调低权重为 0.1)
        contrastive_loss = contrastive_loss_model(prototypes, torch.tensor([i for i in range(1, prototypes.size()[0] + 1)]).cuda(), ref_emb=class_embeddings, ref_labels=cls_ids)
        
        # 【校准核心】：计算真实 IoU 并训练 IoU Head
        with torch.no_grad():
            pred_mask_bin = (preds_final > 0).float()
            intersection = (pred_mask_bin * gt_mask).sum(dim=(2, 3))
            union = pred_mask_bin.sum(dim=(2, 3)) + gt_mask.sum(dim=(2, 3)) - intersection
            real_iou = (intersection / (union + 1e-6)) # [B, 1]

        loss_iou = iou_loss_model(preds_quality.reshape(-1, 1), real_iou.reshape(-1, 1))

        # 总损失公式
        loss = loss_seg_final + 0.5 * loss_seg_base + 0.1 * contrastive_loss + 1.0 * loss_iou
   
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
        
    # --- Validation 阶段 ---
    binary_masks = dict()
    protoype_prompt_encoder.eval()
    sam_decoder.eval()
    learnable_prototypes_model.eval()

    with torch.no_grad():
        prototypes = learnable_prototypes_model()
        for sam_feats, mask_names, cls_ids, _, _ in val_dataloader: 
            sam_feats = sam_feats.cuda().float().permute(0, 2, 3, 1)
            cls_ids = cls_ids.cuda()    
            preds, _, preds_quality = model_forward_function(protoype_prompt_encoder, sam_prompt_encoder, sam_decoder, sam_feats, prototypes, cls_ids)    
            binary_masks = create_binary_masks(binary_masks, preds, preds_quality, mask_names, thr)

    endovis_masks = create_endovis_masks(binary_masks, img_h, img_w)
    endovis_results = eval_endovis(endovis_masks, gt_endovis_masks)
            
    print_log(f"Validation - Epoch: {epoch}; IoU_Results: {endovis_results} ", log_file)
    current_iou = endovis_results["challengIoU"]
    
    if current_iou > best_challenge_iou_val:
        best_challenge_iou_val = current_iou
        torch.save({
            'epoch': epoch,
            'best_score': best_challenge_iou_val,
            'prototype_prompt_encoder_state_dict': protoype_prompt_encoder.state_dict(),
            'sam_decoder_state_dict': sam_decoder.state_dict(),
            'prototypes_state_dict': learnable_prototypes_model.state_dict(),
            'optimizer_state_dict': optimiser.state_dict(),
        }, osp.join(save_dir, 'model_ckp.pth'))
        print_log(f"Best IoU: {best_challenge_iou_val:.4f} at Epoch {epoch}", log_file)        
    else:
        if epoch % 10 == 0:
            torch.save({
                'epoch': epoch,
                'best_score': best_challenge_iou_val,
                'prototype_prompt_encoder_state_dict': protoype_prompt_encoder.state_dict(),
                'sam_decoder_state_dict': sam_decoder.state_dict(),
                'prototypes_state_dict': learnable_prototypes_model.state_dict(),
                'optimizer_state_dict': optimiser.state_dict(),
            }, osp.join(save_dir, 'latest_model.pth'))