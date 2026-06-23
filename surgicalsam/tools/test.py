import numpy as np
import os

# 1. 设置数据集根目录
data_root = "/mnt/data6T/shixy/SurgicalSAM/data/endovis_2017"

# 2. 检查版本 1 到 40
versions = [str(i) for i in range(1, 41)]

print("开始内容检查...")

bad_files_count = 0
zero_files_count = 0
nan_files_count = 0

for v in versions:
    # 我们重点检查类别嵌入，因为它是直接输入 Loss 计算的
    path = os.path.join(data_root, v, "class_embeddings_h")
    
    if not os.path.exists(path):
        continue

    for root, _, files in os.walk(path):
        for filename in files:
            if filename.endswith(".npy"):
                file_path = os.path.join(root, filename)
                
                try:
                    data = np.load(file_path)
                    
                    # 检查是否全为 0
                    if np.all(data == 0):
                        print(f"[全0警告]: {file_path}")
                        zero_files_count += 1
                        bad_files_count += 1
                    
                    # 检查是否包含 NaN
                    elif np.isnan(data).any():
                        print(f"[NaN警告]: {file_path}")
                        nan_files_count += 1
                        bad_files_count += 1
                        
                except Exception as e:
                    print(f"[读取失败]: {file_path}, 错误: {e}")

print("\n" + "="*30)
print(f"检查完成！")
print(f"全为 0 的文件数: {zero_files_count}")
print(f"包含 NaN 的文件数: {nan_files_count}")
print(f"异常文件总数: {bad_files_count}")
print("="*30)

if bad_files_count > 0:
    print("建议：删除包含异常文件的版本文件夹，并重新生成这几个版本的数据。")
else:
    print("所有文件内容在数值上均正常。")