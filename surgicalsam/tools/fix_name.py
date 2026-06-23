import os

# 1. 设置数据集根目录 
data_root = "/mnt/data6T/shixy/SurgicalSAM-HQ/data/endovis_2017/"

# 2. 定义需要处理的版本范围 (1-40)
versions = [str(i) for i in range(1, 41)]

# 3. 定义可能出现命名错误的两个特征文件夹
target_folders = ["sam_features_h"]

print("开始批量修正命名错误")

count = 0
for v in versions:
    v_path = os.path.join(data_root, v)
    
    # 检查版本文件夹是否存在 (防止你只生成了一部分)
    if not os.path.exists(v_path):
        continue
        
    for folder_name in target_folders:
        folder_path = os.path.join(v_path, folder_name)
        
        if not os.path.exists(folder_path):
            continue

        # 使用 os.walk 递归遍历所有子文件�?(seq1, seq2...seq8)
        for root, _, files in os.walk(folder_path):
            for filename in files:
                old_path = os.path.join(root, filename)
                new_filename = None
                
                # 修复 sam_features_h 里的 xxxxnpy.npy
                if filename.endswith("npy.npy"):
                    new_filename = filename.replace("npy.npy", ".npy")
                
                # 修复可能存在�?xxxx.npy.npy
                elif filename.endswith(".npy.npy"):
                    new_filename = filename.replace(".npy.npy", ".npy")
                
                if new_filename:
                    new_path = os.path.join(root, new_filename)
                    # 执行重命�?                    os.rename(old_path, new_path)
                    count += 1

print(f"修正完成！共处理了 {count} 个文件")
print("现在你可以放心运行 train.py 了")