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
        
        # get class id from mask_name 
        cls_id = int(re.search(r"class(\d+)", mask_name).group(1))
        
        # get pre-computed sam feature 
        feat_dir = osp.join(self.mask_dir.replace("binary_annotations", f"sam_features_{self.vit_mode}"), mask_name.split("_")[0] + ".npy")
        sam_feat = np.load(feat_dir)
        
        # get ground-truth mask
        mask_path = osp.join(self.mask_dir, mask_name)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        # get class embedding
        class_embedding_path = osp.join(self.mask_dir.replace("binary_annotations", f"class_embeddings_{self.vit_mode}"), mask_name.replace("png","npy"))
        class_embedding = np.load(class_embedding_path)

        return sam_feat, mask_name, cls_id, mask, class_embedding
 

class Endovis17Dataset(Dataset):
    def __init__(self, data_root_dir = "../data/endovis_2017", 
                 mode = "val",
                 fold = 0,  
                 vit_mode = "h",
                 version = 0):
                        
        self.vit_mode = vit_mode
        
        all_folds = list(range(1, 9))
        fold_seq = {0: [1, 3],
                    1: [2, 5],
                    2: [4, 8],
                    3: [6, 7]}
        
        if mode == "train":
            seqs = [x for x in all_folds if x not in fold_seq[fold]]     
        elif mode == "val":
            seqs = fold_seq[fold]

        self.mask_dir = osp.join(data_root_dir, str(version), "binary_annotations")
        
        self.mask_list = []
        for seq in seqs:
            seq_path = osp.join(self.mask_dir, f"seq{seq}")
            self.mask_list += [f"seq{seq}/{mask}" for mask in os.listdir(seq_path)]
            
    def __len__(self):
        return len(self.mask_list)

    def __getitem__(self, index):
        mask_name = self.mask_list[index]
        
        # get class id from mask_name 
        cls_id = int(re.search(r"class(\d+)", mask_name).group(1))
        
        # get pre-computed sam feature 
        feat_dir = osp.join(self.mask_dir.replace("binary_annotations", f"sam_features_{self.vit_mode}"), mask_name.split("_")[0] + ".npy")
        sam_feat = np.load(feat_dir)
        
        # get ground-truth mask
        mask_path = osp.join(self.mask_dir, mask_name)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        # get class embedding
        class_embedding_path = osp.join(self.mask_dir.replace("binary_annotations", f"class_embeddings_{self.vit_mode}"), mask_name.replace("png","npy"))
        class_embedding = np.load(class_embedding_path)
        
        return sam_feat, mask_name, cls_id, mask, class_embedding
    
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
        # mask_name 示例: "seq1/case5180_05_class1.png"
        
        # 1. 获取 Class ID
        # 使用正则提取 "class" 后面的数字
        cls_id = int(re.search(r"class(\d+)", mask_name).group(1))
        
        # 2. 获取 SAM Feature 路径
        # 【关键修改】 Endovis17 使用 split("_")[0]，这会导致你的文件名被截断出错。
        # 你的文件名中有下划线 (case5180_05)，所以我们要用 "_class" 来分割。
        # 原始: "seq1/case5180_05_class1.png" -> 截取 "seq1/case5180_05" -> 加上 ".npy"
        file_stem = mask_name.split("_class")[0] 
        feat_path = osp.join(self.mask_dir.replace("binary_annotations", f"sam_features_{self.vit_mode}"), file_stem + ".npy")
        
        sam_feat = np.load(feat_path)[0]
        
        # 3. 获取 Ground-Truth Mask
        mask_path = osp.join(self.mask_dir, mask_name)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        # 4. 获取 Class Embedding
        # 替换 binary_annotations -> class_embeddings_h
        # 替换 .png -> .npy
        class_embedding_path = osp.join(self.mask_dir.replace("binary_annotations", f"class_embeddings_{self.vit_mode}"), mask_name.replace(".png", ".npy"))
        class_embedding = np.load(class_embedding_path)
        
        return sam_feat, mask_name, cls_id, mask, class_embedding