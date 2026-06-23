import os
import json
import cv2
import numpy as np
import torch
import glob
from tqdm import tqdm
from segment_anything import sam_model_registry, SamPredictor
import clip

# ================= 1. 配置区域 =================

# 数据集根目录：指向包含 seq1, seq2... 的父级文件夹
# 例如： F:\ch\bigdatset\json\train
dataset_root = "/mnt/data6T/shixy/SurgicalSAM-HQ/data/CATARACTS/0/"

# 输出根目录：生成的数据存放在哪里
# 通常跟 dataset_root 一样，这样就会生成 F:\ch\bigdatset\json\train\annotations\seq1...
output_root = dataset_root 

# 图片文件夹逻辑：
# 如果图片也在 seq 结构里，比如 F:\ch\bigdatset\images\train\seq1...
# 请设置 img_root_base = r"F:\ch\bigdatset\images\train"
# 如果图片就在 json 文件夹里，设为 None
img_root_base = None 

# 输出文件夹名称
out_dir_names = {
    "anno": "annotations",
    "binary": "binary_annotations",
    "embed": "class_embeddings_h",
    "feats": "sam_features_h"
}

# 手术器械白名单 (全局统一 ID)
class_map = {
    'Spatula': 1,
    'Phacoemulsification Tip': 2,
    'Lens Injector': 3,
    'Capsulorhexis Forceps': 4,
    'Incision Knife': 5,
    'Slit Knife': 6,
    'Katena Forceps': 7
}

# 模型配置
sam_checkpoint = "/mnt/data6T/shixy/SurgicalSAM-HQ/ckp/sam/sam_vit_h_4b8939.pth"
model_type = "vit_h"
device = "cuda" if torch.cuda.is_available() else "cpu"

# =================================================

def get_sam_model():
    print(f"正在加载 SAM 模型 ({device})...")
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    return sam

def get_clip_model():
    print(f"正在加载 CLIP 模型 ({device})...")
    model, preprocess = clip.load("ViT-B/32", device=device)
    return model

def process_all_sequences():
    # 1. 加载模型 (只加载一次)
    sam = get_sam_model()
    predictor = SamPredictor(sam)
    clip_model = get_clip_model()

    # 2. 获取所有 seq 文件夹
    # 假设 dataset_root 下面全是 seq1, seq2 这种文件夹
    seq_dirs = [d for d in os.listdir(dataset_root) if os.path.isdir(os.path.join(dataset_root, d))]
    
    # 过滤掉我们自己生成的输出文件夹 (防止重复运行报错)
    exclude_dirs = list(out_dir_names.values())
    seq_dirs = [d for d in seq_dirs if d not in exclude_dirs]

    print(f"检测到 {len(seq_dirs)} 个序列文件夹: {seq_dirs}")

    # 3. 开始遍历每一个 seq
    for seq_name in tqdm(seq_dirs, desc="Total Progress"):
        
        # 当前处理的 seq 路径 (输入)
        # e.g., .../train/seq1
        current_seq_in_path = os.path.join(dataset_root, seq_name)
        
        # 定义当前 seq 的输出路径 (核心修改点)
        # 结构变成: output_root / annotations / seq1 / ...
        current_out_paths = {}
        for key, dir_name in out_dir_names.items():
            # e.g., .../train/annotations/seq1
            path = os.path.join(output_root, dir_name, seq_name)
            os.makedirs(path, exist_ok=True) # 创建 seq1 文件夹
            current_out_paths[key] = path

        # 查找当前 seq 下的所有 json
        json_files = glob.glob(os.path.join(current_seq_in_path, "*.json"))
        
        if not json_files:
            continue

        # 处理当前 seq 下的每一张图
        for json_file in tqdm(json_files, desc=f"Processing {seq_name}", leave=False):
            
            # --- A. 解析文件名 ---
            filename = os.path.basename(json_file)
            image_name = filename.replace(".json", "") 
            stem = os.path.splitext(image_name)[0] # 不带后缀的文件名

            # --- B. 读取图片 ---
            # 自动适配路径
            if img_root_base:
                # 假设图片路径是 base/seq1/xxx.png
                img_path = os.path.join(img_root_base, seq_name, image_name)
            else:
                # 默认图片跟 json 在一起
                img_path = os.path.join(current_seq_in_path, image_name)

            image = cv2.imread(img_path)
            # 尝试修复后缀
            if image is None:
                if not image_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                     image = cv2.imread(img_path + ".png")
            
            if image is None:
                # print(f"Warning: Missing image {img_path}")
                continue

            original_h, original_w = image.shape[:2]

            # --- C. 解析 JSON ---
            with open(json_file, 'r') as f:
                data = json.load(f)

            full_mask = np.zeros((original_h, original_w), dtype=np.uint8)
            class_polygons = {} 
            present_classes = {} 

            objects = data.get('objects', [])
            
            # 过滤与解析
            for obj in objects:
                class_title = obj.get('classTitle')
                class_id = class_map.get(class_title) # 白名单过滤
                
                if class_id is None: continue
                
                present_classes[class_id] = class_title
                points = obj.get('points', {}).get('exterior', [])
                if not points: continue
                
                pts = np.array(points, np.int32).reshape((-1, 1, 2))
                
                if class_id not in class_polygons:
                    class_polygons[class_id] = []
                class_polygons[class_id].append(pts)

            # --- D. 保存结果 ---
            
            # 1. 保存 Masks (注意路径是 current_out_paths)
            for cid, polys in class_polygons.items():
                cv2.fillPoly(full_mask, polys, color=int(cid))
                
                binary_mask = np.zeros((original_h, original_w), dtype=np.uint8)
                cv2.fillPoly(binary_mask, polys, color=255)
                
                bin_filename = f"{stem}_class{cid}.png"
                # Save to: .../binary_annotations/seq1/xxx_class1.png
                cv2.imwrite(os.path.join(current_out_paths["binary"], bin_filename), binary_mask)

            # Save to: .../annotations/seq1/xxx.png
            cv2.imwrite(os.path.join(current_out_paths["anno"], f"{stem}.png"), full_mask)

            # 2. 保存 CLIP Embeddings
            if present_classes:
                with torch.no_grad():
                    for cid, cname in present_classes.items():
                        prompt = f"A photo of a {cname}."
                        text_input = clip.tokenize([prompt]).to(device)
                        text_features = clip_model.encode_text(text_input)
                        text_features /= text_features.norm(dim=-1, keepdim=True)
                        embed_npy = text_features.cpu().numpy()[0]
                        
                        # Save to: .../class_embeddings_h/seq1/xxx_class1.npy
                        np.save(os.path.join(current_out_paths["embed"], f"{stem}_class{cid}.npy"), embed_npy)

            # 3. 保存 SAM Features
            # Save to: .../sam_features_h/seq1/xxx.npy
            # 检查文件是否已存在（可选优化，跳过已生成的）
            sam_out_path = os.path.join(current_out_paths["feats"], f"{stem}.npy")
            if not os.path.exists(sam_out_path):
                with torch.no_grad():
                    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    predictor.set_image(image_rgb)
                    image_embedding = predictor.get_image_embedding().cpu().numpy()
                    np.save(sam_out_path, image_embedding)

if __name__ == "__main__":
    process_all_sequences()