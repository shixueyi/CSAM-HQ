import sys
sys.path.append("..")
import os
import os.path as osp 
import random 
import argparse
import numpy as np 
import torch 
from torch.utils.data import DataLoader
# 【修改点 1】: 导入 CataractsDataset
from dataset import Endovis18Dataset, Endovis17Dataset, CataractsDataset
from segment_anything import sam_model_registry
from model import Learnable_Prototypes, Prototype_Prompt_Encoder
from utils import print_log, create_binary_masks, create_endovis_masks, eval_endovis, read_gt_endovis_masks
from model_forward import model_forward_function
from loss import DiceLoss
from pytorch_metric_learning import losses

print("======> Process Arguments")
parser = argparse.ArgumentParser()
# 【修改点 2】: 在 choices 中加入 "Cataracts"
parser.add_argument('--dataset', type=str, default="Cataracts", choices=["endovis_2018", "endovis_2017", "Cataracts"], help='specify dataset')
parser.add_argument('--fold', type=int, default=0, choices=[0,1,2,3], help='specify fold number for endovis_2017/Cataracts dataset')
args = parser.parse_args()

print("======> Set Parameters for Training" )
dataset_name = args.dataset
fold = args.fold
thr = 0
seed = 666  
data_root_dir = f"../data/{dataset_name}" # 请确保你的数据放在这里，或者手动指定绝对路径
batch_size = 32 # 如果显存不够，改小这个，比如 8 或 16
vit_mode = "h"

# set seed for reproducibility 
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(seed)

print("======> Load Dataset-Specific Parameters" )

# 初始化默认值，防止未定义报错
num_classes = 7 
img_h, img_w = 1024, 1280 # 默认尺寸

if "18" in dataset_name:
    num_tokens = 2
    num_classes = 1 # Endovis18 这里的实现通常是二分类或者 instrument 作为一个类
    img_h, img_w = 1024, 1280
    val_dataset = Endovis18Dataset(data_root_dir = data_root_dir, 
                                   mode="val",
                                   vit_mode = "h")
    
    gt_endovis_masks = read_gt_endovis_masks(data_root_dir = data_root_dir, mode = "val")
    num_epochs = 500
    lr = 0.001
    save_dir = "./work_dirs/endovis_2018/"

elif "17" in dataset_name:
    num_tokens = 4
    num_classes = 7
    img_h, img_w = 1024, 1280
    val_dataset = Endovis17Dataset(data_root_dir = data_root_dir,
                                   mode = "val",
                                   fold = fold, 
                                   vit_mode = "h",
                                   version = 0)
    
    gt_endovis_masks = read_gt_endovis_masks(data_root_dir = data_root_dir, 
                                             mode = "val", 
                                             fold = fold)
    num_epochs = 2000
    lr = 0.0001
    save_dir = f"./work_dirs/endovis_2017/{fold}"

# 【修改点 3】: 新增 Cataracts 配置块
elif "Cataracts" in dataset_name:
    num_tokens = 4  # 假设也用 4 个 token，类似 Endovis17
    num_classes = 7 # 你的 class_map 里有 7 类
    
    # 【注意】: 这里需要设置原始图像的分辨率，用于评估时把 Mask 还原回去
    # 白内障手术视频通常是 1920x1080，请根据你的原始图片尺寸修改这里！
    img_h, img_w = 1080, 1920 
    
    # 这里的 data_root_dir 需要对应你 dataset.py 里的默认路径，或者你在命令行里传入绝对路径
    # 如果 data_root_dir 变量不对，请在这里手动写死，例如:
    # data_root_dir = "/mnt/data6T/shixy/SurgicalSAM-HQ/data/CATARACTS/"
    
    val_dataset = CataractsDataset(data_root_dir = data_root_dir,
                                   mode = "val",
                                   fold = fold,
                                   vit_mode = "h",
                                   version = 0)
    
    # 这里的 read_gt_endovis_masks 需要你的 utils.py 支持读取该路径结构
    # 如果你的目录结构和 Endovis17 一模一样（seqX/binary_annotations/...），这个函数应该能通用
    gt_endovis_masks = read_gt_endovis_masks(data_root_dir = data_root_dir, 
                                             mode = "val", 
                                             fold = fold)
    num_epochs = 1000 # 可以根据需要调整
    lr = 0.0001
    save_dir = f"./work_dirs/Cataracts/{fold}"

    
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

print("======> Load SAM" )
if vit_mode == "h":
    sam_checkpoint = "../ckp/sam/sam_hq_vit_h.pth"
model_type = "vit_h_no_image_encoder"
sam_prompt_encoder, sam_decoder = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam_prompt_encoder.cuda()
sam_decoder.cuda()

for name, param in sam_prompt_encoder.named_parameters():
    param.requires_grad = False
for name, param in sam_decoder.named_parameters():
    param.requires_grad = True

print("======> Load Prototypes and Prototype-based Prompt Encoder" )
# 【修改点】: 使用动态变量 num_classes，而不是写死 7
learnable_prototypes_model = Learnable_Prototypes(num_classes = num_classes, feat_dim = 256).cuda()

protoype_prompt_encoder =  Prototype_Prompt_Encoder(feat_dim = 256, 
                                                    hidden_dim_dense = 128, 
                                                    hidden_dim_sparse = 128, 
                                                    size = 64, 
                                                    num_tokens = num_tokens).cuda()
 
with open(sam_checkpoint, "rb") as f:
    state_dict = torch.load(f)
    sam_pn_embeddings_weight = {k.split("prompt_encoder.point_embeddings.")[-1]: v for k, v in state_dict.items() if k.startswith("prompt_encoder.point_embeddings") and ("0" in k or "1" in k)}
    sam_pn_embeddings_weight_ckp = {"0.weight": torch.concat([sam_pn_embeddings_weight['0.weight'] for _ in range(num_tokens)], dim=0),
                                    "1.weight": torch.concat([sam_pn_embeddings_weight['1.weight'] for _ in range(num_tokens)], dim=0)}

    protoype_prompt_encoder.pn_cls_embeddings.load_state_dict(sam_pn_embeddings_weight_ckp)

for name, param in learnable_prototypes_model.named_parameters():
    param.requires_grad = True
    
for name, param in protoype_prompt_encoder.named_parameters():
    if "pn_cls_embeddings" in name:
        param.requires_grad = False
    else:
        param.requires_grad = True
              
print("======> Define Optmiser and Loss")
seg_loss_model = DiceLoss().cuda()
contrastive_loss_model = losses.NTXentLoss(temperature=0.07).cuda()
optimiser = torch.optim.Adam([
            {'params': learnable_prototypes_model.parameters()},
            {'params': protoype_prompt_encoder.parameters()},
            {'params': sam_decoder.parameters()}
        ], lr = lr, weight_decay = 0.0001)


print("======> Set Saving Directories and Logs")
os.makedirs(save_dir, exist_ok = True) 
log_file = osp.join(save_dir, "log.txt")
print_log(str(args), log_file)

print("======> Checking for checkpoints to resume" )

start_epoch = 0 
best_challenge_iou_val = -100.0

resume_ckp_path = osp.join(save_dir, 'model_ckp.pth')
if osp.exists(resume_ckp_path):
    print_log(f"发现已存在的模型文件 {resume_ckp_path}，正在恢复训练...", log_file)
    checkpoint = torch.load(resume_ckp_path, map_location='cuda')
    
    protoype_prompt_encoder.load_state_dict(checkpoint['prototype_prompt_encoder_state_dict'])
    sam_decoder.load_state_dict(checkpoint['sam_decoder_state_dict'])
    learnable_prototypes_model.load_state_dict(checkpoint['prototypes_state_dict'])
    
    if 'optimizer_state_dict' in checkpoint:
        optimiser.load_state_dict(checkpoint['optimizer_state_dict'])
    
    if 'best_score' in checkpoint:
        best_challenge_iou_val = checkpoint['best_score']
    if 'epoch' in checkpoint:
        start_epoch = checkpoint['epoch'] + 1
        
    print_log(f"成功恢复！将从第 {start_epoch} 轮开始训练，当前最高 IoU 为: {best_challenge_iou_val:.4f}", log_file)
else:
    print_log("未发现检查点，将从头开始训练。", log_file)

print("======> Start Training and Validation" )

for epoch in range(start_epoch, num_epochs):    
    
    # choose the augmentation version to use for the current epoch 
    # 【注意】如果你没有为 Cataracts 生成 version 1-40 的增广数据（只生成了 version 0）
    # 那么这里可能会报错。如果没有增广数据，建议把下面的 loop 逻辑改一下，或者强制 version=0
    if epoch % 2 == 0 :
        version = 0 
    else:
        version = int((epoch % 80 + 1)/2)
    
    # 临时修复：如果 Cataracts 只有 version 0 数据，取消注释下面这行
    # if "Cataracts" in dataset_name: version = 0

    if "18" in dataset_name:
        train_dataset = Endovis18Dataset(data_root_dir = data_root_dir,
                                         mode="train",
                                         vit_mode = vit_mode,
                                         version = version)
        
    elif "17" in dataset_name:
        train_dataset = Endovis17Dataset(data_root_dir = data_root_dir,
                                         mode="train",
                                         fold = fold,
                                         vit_mode = vit_mode,
                                         version = version)

    # 【修改点 4】: 新增 Cataracts 训练集加载逻辑
    elif "Cataracts" in dataset_name:
        train_dataset = CataractsDataset(data_root_dir = data_root_dir,
                                         mode="train",
                                         fold = fold,
                                         vit_mode = vit_mode,
                                         version = version) # 如果没有增广文件夹，这里会报错，需改为 version=0
        
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    
    
    # training 
    protoype_prompt_encoder.train()
    sam_decoder.train()
    learnable_prototypes_model.train()

    for sam_feats, _, cls_ids, masks, class_embeddings in train_dataloader: 
        
        norms = torch.norm(class_embeddings, p=2, dim=1)
        valid_mask = (norms > 0.05) 
        
        if not valid_mask.any():
            continue

        sam_feats = sam_feats[valid_mask].cuda()
        cls_ids = cls_ids[valid_mask].cuda()
        masks = masks[valid_mask].cuda()
        class_embeddings = class_embeddings[valid_mask].cuda()
            
        prototypes = learnable_prototypes_model()
        
        preds, _ = model_forward_function(protoype_prompt_encoder, sam_prompt_encoder, sam_decoder, sam_feats, prototypes, cls_ids)    
        
        # compute loss 
        contrastive_loss = contrastive_loss_model(prototypes, torch.tensor([i for i in range(1, prototypes.size()[0] + 1)]).cuda(), ref_emb = class_embeddings, ref_labels = cls_ids)
        seg_loss = seg_loss_model(preds, masks/255)
    
        loss = seg_loss + 0.1 * contrastive_loss
   
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
        
    # validation 
    binary_masks = dict()
    protoype_prompt_encoder.eval()
    sam_decoder.eval()
    learnable_prototypes_model.eval()

    with torch.no_grad():
        prototypes = learnable_prototypes_model()
        
        for sam_feats, mask_names, cls_ids, _, _ in val_dataloader: 
            
            sam_feats = sam_feats.cuda()
            cls_ids = cls_ids.cuda()    
            
            preds , preds_quality = model_forward_function(protoype_prompt_encoder, sam_prompt_encoder, sam_decoder, sam_feats, prototypes, cls_ids)    
 
            binary_masks = create_binary_masks(binary_masks, preds, preds_quality, mask_names, thr)

    # 【注意】: 这里使用了前面定义的 img_h, img_w (1080, 1920)
    endovis_masks = create_endovis_masks(binary_masks, img_h, img_w)
    endovis_results = eval_endovis(endovis_masks, gt_endovis_masks)
            
    print_log(f"Validation - Epoch: {epoch}/{num_epochs-1}; IoU_Results: {endovis_results} ", log_file)
    
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

        print_log(f"Best Challenge IoU: {best_challenge_iou_val:.4f} at Epoch {epoch}", log_file)        
    else:
        print_log(f"Epoch {epoch} performance did not improve. Continuing...", log_file)
        if epoch % 10 == 0:
            torch.save({
                'epoch': epoch,
                'best_score': best_challenge_iou_val,
                'prototype_prompt_encoder_state_dict': protoype_prompt_encoder.state_dict(),
                'sam_decoder_state_dict': sam_decoder.state_dict(),
                'prototypes_state_dict': learnable_prototypes_model.state_dict(),
                'optimizer_state_dict': optimiser.state_dict(),
            }, osp.join(save_dir, 'latest_model.pth'))