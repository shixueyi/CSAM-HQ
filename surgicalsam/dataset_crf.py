from torch.utils.data import Dataset
import os 
import os.path as osp
import re 
import numpy as np 
import cv2 

class Endovis18Dataset(Dataset):
    def __init__(self, data_root_dir = "../data/endovis_2018", 
                 mode = "val", 
                 vit_mode = "h",
                 version = 0):
        
        """Define the Endovis18 dataset

        Args:
            data_root_dir (str, optional): root dir containing all data for Endovis18. Defaults to "../data/endovis_2018".
            mode (str, optional): either in "train" or "val" mode. Defaults to "val".
            vit_mode (str, optional): "h", "l", "b" for huge, large, and base versions of SAM. Defaults to "h".
            version (int, optional): augmentation version to use. Defaults to 0.
        """
        
        self.vit_mode = vit_mode
       
        # directory containing all binary annotations
        if mode == "train":
            self.mask_dir = osp.join(data_root_dir, mode, str(version), "binary_annotations")
        elif mode == "val":
            self.mask_dir = osp.join(data_root_dir, mode, "binary_annotations")

        # put all binary masks into a list
        self.mask_list = []
        for subdir, _, files in os.walk(self.mask_dir):
            if len(files) == 0:
                continue 
            self.mask_list += [osp.join(osp.basename(subdir),i) for i in files]

    def __len__(self):
        return len(self.mask_list)

    def __getitem__(self, index):
        mask_name = self.mask_list[index]
        cls_id = int(re.search(r"class(\d+)", mask_name).group(1))
        
        # 1. 加载 SAM 特征
        feat_dir = osp.join(self.mask_dir.replace("binary_annotations", f"sam_features_{self.vit_mode}"), mask_name.split("_")[0] + ".npy")
        sam_feat = np.load(feat_dir)
        
        # 2. 加载 GT Mask
        mask_path = osp.join(self.mask_dir, mask_name)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        # 3. 加载 Class Embedding
        class_embedding_path = osp.join(self.mask_dir.replace("binary_annotations", f"class_embeddings_{self.vit_mode}"), mask_name.replace("png","npy"))
        class_embedding = np.load(class_embedding_path)

        # ================= NEW: 安全加载图像逻辑 =================
        # 尝试猜测原图路径 (假设在 images 文件夹下)
        img_path = mask_path.replace("binary_annotations", "images") 
        
        # 容错：如果找不到，尝试去掉 _class 后缀查找
        if not osp.exists(img_path):
             base_name = mask_name.split("_class")[0] + ".png" 
             img_path = osp.join(osp.dirname(img_path), base_name)
        
        if osp.exists(img_path):
            # 找到了原图 (Endovis 2017 可能会走这里)
            image = cv2.imread(img_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            # 没找到原图 (Endovis 2018 会走这里)
            # 返回全黑图片，形状与 mask 一致
            image = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
        # =======================================================

        return sam_feat, mask_name, cls_id, mask, class_embedding, image
 

class Endovis17Dataset(Dataset):
    def __init__(self, data_root_dir = "../data/endovis_2017", 
                 mode = "val",
                 fold = 0,  
                 vit_mode = "h",
                 version = 0):
                        
        self.vit_mode = vit_mode
        self.mode = mode
        
        all_folds = list(range(1, 9))
        fold_seq = {0: [1, 3],
                    1: [2, 5],
                    2: [4, 8],
                    3: [6, 7]}
        
        if mode == "train":
            seqs = [x for x in all_folds if x not in fold_seq[fold]]     
        elif mode == "val":
            seqs = fold_seq[fold]

        # 这里的目录是 binary_annotations
        self.mask_dir = osp.join(data_root_dir, str(version), "binary_annotations")
        
        self.mask_list = []
        for seq in seqs:
            seq_path = osp.join(self.mask_dir, f"seq{seq}")
            # mask_list 里的格式是: "seq1/00000_class1.png"
            if os.path.exists(seq_path):
                self.mask_list += [f"seq{seq}/{mask}" for mask in os.listdir(seq_path)]
            else:
                print(f"Warning: Sequence path not found {seq_path}")
            
    def __len__(self):
        return len(self.mask_list)

    def __getitem__(self, index):
        mask_name = self.mask_list[index]
        # mask_name 示例: "seq1/00190_class2.png"
        
        # 1. 获取 Class ID
        cls_id = int(re.search(r"class(\d+)", mask_name).group(1))
        
        # 2. 获取 SAM 特征
        # 注意：这里假设 sam_features 文件夹结构和 mask 是一样的
        base_name_only = mask_name.split("_")[0] # "seq1/00190"
        feat_dir = osp.join(self.mask_dir.replace("binary_annotations", f"sam_features_{self.vit_mode}"), base_name_only + ".npy")
        sam_feat = np.load(feat_dir)
        
        # 3. 获取 GT Mask (用于评估)
        mask_path = osp.join(self.mask_dir, mask_name)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        # 4. 获取 Embedding
        class_embedding_path = osp.join(self.mask_dir.replace("binary_annotations", f"class_embeddings_{self.vit_mode}"), mask_name.replace("png","npy"))
        class_embedding = np.load(class_embedding_path)
        
        # =======================================================
        # NEW: 针对你截图路径的修复逻辑
        # =======================================================
        
        # 步骤 A: 确定 images 根目录
        # 你的 mask_dir 是 .../endovis_2017/0/binary_annotations
        # 我们要改成 .../endovis_2017/0/images
        img_root_dir = self.mask_dir.replace("binary_annotations", "images")
        
        # 步骤 B: 解析 mask 名字
        # mask_name = "seq1/00190_class2.png"
        seq_dir = osp.dirname(mask_name)       # "seq1"
        filename = osp.basename(mask_name)     # "00190_class2.png"
        
        # 步骤 C: 提取纯数字编号 "00190"
        # 逻辑：以 "_class" 分割，取第一部分
        file_id = filename.split("_class")[0] 
        
        # 步骤 D: 拼接成 .jpg 路径
        # 结果应为: .../images/seq1/00190.jpg
        img_real_path = osp.join(img_root_dir, seq_dir, f"{file_id}.jpg")
        
        # 步骤 E: 读取
        if osp.exists(img_real_path):
            image = cv2.imread(img_real_path)
            # OpenCV 读入是 BGR，CRF 需要 RGB 或 BGR 均可，但最好转 RGB 保持一致
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            # 如果还是找不到（比如 2018 数据集），返回全黑
            # print(f"[Warning] Image not found: {img_real_path}") # 调试时可开启
            image = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
            
        # =======================================================
        
        return sam_feat, mask_name, cls_id, mask, class_embedding, image
    
class CataractsDataset(Dataset):
    def __init__(self, data_root_dir="/mnt/data6T/shixy/SurgicalSAM-HQ/data/CATARACTS/", 
                 mode="val",
                 fold=0,  
                 vit_mode="h",
                 version=0):
        """
        Args:
            fold (int): 默认为 4 折交叉验证 (0, 1, 2, 3)
        """
        
        self.vit_mode = vit_mode
        
        
        all_folds = list(range(1, 9)) # [1, 2, ..., 25]
        
        fold_seq = {
            0: [1, 2],
            1: [3, 4],
            2: [5, 6],
            3: [7, 8]
        }
        
        if mode == "train":
            seqs = [x for x in all_folds if x not in fold_seq[fold]]     
        elif mode == "val":
            seqs = fold_seq[fold]
        
        self.mask_dir = osp.join(data_root_dir, str(version), "binary_annotations")
        
        self.mask_list = []
        for seq in seqs:
            seq_path = osp.join(self.mask_dir, f"seq{seq}")
            if os.path.isdir(seq_path):
                # 结果类似于: ['seq1/case01_class1.png', 'seq1/case01_class2.png', ...]
                self.mask_list += [f"seq{seq}/{mask}" for mask in os.listdir(seq_path) if mask.endswith(".png")]
            else:
                print(f"Warning: Sequence folder not found: {seq_path}")
            
    def __len__(self):
        return len(self.mask_list)

    def __getitem__(self, index):
        mask_name = self.mask_list[index]
        
        # 1. 获取 Class ID
        cls_id = int(re.search(r"class(\d+)", mask_name).group(1))
        
        # 2. 获取 SAM Feature
        file_stem = mask_name.split("_class")[0] 
        feat_path = osp.join(self.mask_dir.replace("binary_annotations", f"sam_features_{self.vit_mode}"), file_stem + ".npy")
        # 记得之前修过的维度问题
        sam_feat = np.load(feat_path)[0]
        
        # 3. 获取 GT Mask
        mask_path = osp.join(self.mask_dir, mask_name)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        # 4. 获取 Class Embedding
        class_embedding_path = osp.join(self.mask_dir.replace("binary_annotations", f"class_embeddings_{self.vit_mode}"), mask_name.replace(".png", ".npy"))
        class_embedding = np.load(class_embedding_path)

        # ==================== 【新增】读取原图 ====================
        # 逻辑：从 binary_annotations 路径推导出 images 路径
        # 例如: .../0/binary_annotations/seq1/case01_class1.png 
        #   -> .../0/images/seq1/case01.png
        
        # 构造原图路径
        image_dir = self.mask_dir.replace("binary_annotations", "images")
        
        # 还原原图文件名：去掉 _classX 后缀
        # 例如 mask_name 是 seq1/case5180_01_class1.png
        # file_stem 是 seq1/case5180_01
        image_name = file_stem + ".png" 
        
        image_path = osp.join(image_dir, image_name)
        
        # 读取图片 (OpenCV 读取的是 BGR，通常模型处理或是 CRF 可能需要 RGB，视具体实现而定)
        # 这里为了稳健，读取后不转 Tensor，直接返回 numpy 数组，交给 dataloader 处理
        image = cv2.imread(image_path)
        if image is not None:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # 转为 RGB
        else:
            # 如果找不到图（防止报错），生成一个全黑图
            # print(f"Warning: Image not found {image_path}")
            image = np.zeros((1024, 1280, 3), dtype=np.uint8) # 注意尺寸要改对

        # ========================================================

        # 返回 6 个值
        return sam_feat, mask_name, cls_id, mask, class_embedding, image
    
