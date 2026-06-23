import numpy as np 
import cv2 
import torch 
import os 
import os.path as osp 
import re 



def create_binary_masks(binary_masks, preds, preds_quality, mask_names, thr):

    """Gather the predicted binary masks of different frames and classes into a dictionary, mask quality is also recorded

    Returns:
        dict: a dictionary containing all predicted binary masks organised based on sequence, frame, and mask name
    """
    preds = preds.cpu()
    preds_quality = preds_quality.cpu()
    
    pred_masks = (preds > thr).int()
    

    for pred_mask, mask_name, pred_quality in zip(pred_masks, mask_names, preds_quality):        
      
        seq_name = mask_name.split("/")[0]
        #frame_name = osp.basename(mask_name).split("_")[0]
        frame_name = osp.basename(mask_name).split("_class")[0]
        
        if seq_name not in binary_masks.keys():
            binary_masks[seq_name] = dict()
        
        if frame_name not in binary_masks[seq_name].keys():
            binary_masks[seq_name][frame_name] = list()
            
        binary_masks[seq_name][frame_name].append({
            "mask_name": mask_name,
            "mask": pred_mask,
            "mask_quality": pred_quality.item()
        })
        
    return binary_masks
        

def create_endovis_masks(binary_masks, H, W):
    """given the dictionary containing all predicted binary masks, compute final prediction of each frame and organise the prediction masks into a dictionary
       H - height of image 
       W - width of image
    
    Returns: a dictionary containing one prediction mask for each frame with the frame name as key and its predicted mask as value; 
             For each frame, the binary masks of different classes are conbined into a single prediction mask;
             The prediction mask for each frame is a 1024 x 1280 map with each value representing the class id for the pixel;
             
    """
    
    endovis_masks = dict()
    
    for seq in binary_masks.keys():
        
        for frame in binary_masks[seq].keys():
            
            endovis_mask = np.zeros((H, W))
    
            binary_masks_list = binary_masks[seq][frame]

            binary_masks_list = sorted(binary_masks_list, key=lambda x: x["mask_quality"])
           
            for binary_mask in binary_masks_list:
                mask_name  = binary_mask["mask_name"]
                predicted_label = int(re.search(r"class(\d+)", mask_name).group(1))
                mask = binary_mask["mask"].numpy()
                endovis_mask[mask==1] = predicted_label

            endovis_mask = endovis_mask.astype(int)

            endovis_masks[f"{seq}/{frame}.png"] = endovis_mask
    
    return endovis_masks


def eval_endovis(endovis_masks, gt_endovis_masks):
    """Given the predicted masks and groundtruth annotations, predict the challenge IoU, IoU, mean class IoU, and the IoU for each class
        
      ** The evaluation code is taken from the official evaluation code of paper: ISINet: An Instance-Based Approach for Surgical Instrument Segmentation
      ** at https://github.com/BCV-Uniandes/ISINet
      
    Args:
        endovis_masks (dict): the dictionary containing the predicted mask for each frame 
        gt_endovis_masks (dict): the dictionary containing the groundtruth mask for each frame 

    Returns:
        dict: a dictionary containing the evaluation results for different metrics 
    """

    endovis_results = dict()
    num_classes = 7
    
    all_im_iou_acc = []
    all_im_iou_acc_challenge = []
    cum_I, cum_U = 0, 0
    class_ious = {c: [] for c in range(1, num_classes+1)}
    
    for file_name, prediction in endovis_masks.items():
       
        full_mask = gt_endovis_masks[file_name]
        
        im_iou = []
        im_iou_challenge = []
        target = full_mask.numpy()
        gt_classes = np.unique(target)
        gt_classes.sort()
        gt_classes = gt_classes[gt_classes > 0] 
        if np.sum(prediction) == 0:
            if target.sum() > 0: 
                all_im_iou_acc.append(0)
                all_im_iou_acc_challenge.append(0)
                for class_id in gt_classes:
                    class_ious[class_id].append(0)
            continue

        gt_classes = torch.unique(full_mask)
        # loop through all classes from 1 to num_classes 
        for class_id in range(1, num_classes + 1): 

            current_pred = (prediction == class_id).astype(np.float64)
            current_target = (full_mask.numpy() == class_id).astype(np.float64)

            if current_pred.astype(np.float64).sum() != 0 or current_target.astype(np.float64).sum() != 0:
                i, u = compute_mask_IU_endovis(current_pred, current_target)     
                im_iou.append(i/u)
                cum_I += i
                cum_U += u
                class_ious[class_id].append(i/u)
                if class_id in gt_classes:
                    im_iou_challenge.append(i/u)
        
        if len(im_iou) > 0:
            all_im_iou_acc.append(np.mean(im_iou))
        if len(im_iou_challenge) > 0:
            all_im_iou_acc_challenge.append(np.mean(im_iou_challenge))

    # calculate final metrics
    final_im_iou = cum_I / (cum_U + 1e-15)
    mean_im_iou = np.mean(all_im_iou_acc)
    mean_im_iou_challenge = np.mean(all_im_iou_acc_challenge)

    final_class_im_iou = torch.zeros(9)
    cIoU_per_class = []
    for c in range(1, num_classes + 1):
        final_class_im_iou[c-1] = torch.tensor(class_ious[c]).float().mean()
        cIoU_per_class.append(round((final_class_im_iou[c-1]*100).item(), 3))
        
    mean_class_iou = torch.tensor([torch.tensor(values).float().mean() for c, values in class_ious.items() if len(values) > 0]).mean().item()
    
    endovis_results["challengIoU"] = round(mean_im_iou_challenge*100,3)
    endovis_results["IoU"] = round(mean_im_iou*100,3)
    endovis_results["mcIoU"] = round(mean_class_iou*100,3)
    endovis_results["mIoU"] = round(final_im_iou*100,3)
    
    endovis_results["cIoU_per_class"] = cIoU_per_class
    
    return endovis_results



def compute_mask_IU_endovis(masks, target):
    """compute iou used for evaluation
    """
    assert target.shape[-2:] == masks.shape[-2:]
    temp = masks * target
    intersection = temp.sum()
    union = ((masks + target) - temp).sum()
    return intersection, union


def read_gt_endovis_masks(data_root_dir = "../data/endovis_2018",
                          mode = "val", 
                          fold = None):
    
    """Read the annotation masks into a dictionary to be used as ground truth in evaluation.

    Returns:
        dict: mask names as key and annotation masks as value 
    """
    # gt_endovis_masks = dict()
    
    # if "2018" in data_root_dir:
    #     gt_endovis_masks_path = osp.join(data_root_dir, mode, "annotations")
    #     for seq in os.listdir(gt_endovis_masks_path):
    #         for mask_name in os.listdir(osp.join(gt_endovis_masks_path, seq)):
    #             full_mask_name = f"{seq}/{mask_name}"
    #             mask = torch.from_numpy(cv2.imread(osp.join(gt_endovis_masks_path, full_mask_name),cv2.IMREAD_GRAYSCALE))
    #             gt_endovis_masks[full_mask_name] = mask
                

    gt_endovis_masks = dict()
    
    if "2018" in data_root_dir:
        # 统一从 'binary_annotations' 读取，与 dataset.py 保持一致
        gt_endovis_masks_path = osp.join(data_root_dir, mode, "binary_annotations")

        print(f"Reading and merging binary annotations for mode '{mode}' from: {gt_endovis_masks_path}")
        
        # 检查路径是否存在，如果不存在则直接返回空字典，避免崩溃
        if not osp.exists(gt_endovis_masks_path):
            print(f"Warning: Annotation path not found: {gt_endovis_masks_path}")
            return gt_endovis_masks

        for seq in os.listdir(gt_endovis_masks_path):
            seq_path = osp.join(gt_endovis_masks_path, seq)
            if not osp.isdir(seq_path): continue

            # 1. 按帧名对所有二值化mask文件进行分组
            frame_files = {}
            for mask_name in os.listdir(seq_path):
                if not mask_name.endswith('.png'): continue # 避免处理非png文件
                frame_id = mask_name.split('_')[0]
                if frame_id not in frame_files:
                    frame_files[frame_id] = []
                frame_files[frame_id].append(mask_name)

            # 2. 遍历每个帧，合并其所有的类别mask
            for frame_id, files in frame_files.items():
                combined_mask = None
                
                for mask_name in files:
                    # 读取二值化mask
                    mask_path = osp.join(seq_path, mask_name)
                    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                    if mask is None: continue

                    # 第一次循环时，初始化一个空的组合mask
                    if combined_mask is None:
                        H, W = mask.shape
                        combined_mask = np.zeros((H, W), dtype=np.uint8)

                    # 从文件名中提取类别ID
                    match = re.search(r'_class(\d+)', mask_name)
                    if not match: continue
                    class_id = int(match.group(1))
                    
                    # 将这个类别的信息合并到最终的mask中
                    combined_mask[mask > 0] = class_id
                
                if combined_mask is not None:
                    # 使用 'seq/frame_id.png' 格式作为key，以便评估时匹配
                    full_mask_name = f"{seq}/{frame_id}.png"
                    gt_endovis_masks[full_mask_name] = torch.from_numpy(combined_mask)
    elif "2017" in data_root_dir:
        if fold == "all":
            seqs = [1,2,3,4,5,6,7,8]
            
        elif fold in [0,1,2,3]:
            fold_seq = {0: [1, 3],
                        1: [2, 5],
                        2: [4, 8],
                        3: [6, 7]}
            
            seqs = fold_seq[fold]
        
        gt_endovis_masks_path = osp.join(data_root_dir, "0", "annotations")
        
        for seq in seqs:
            for mask_name in os.listdir(osp.join(gt_endovis_masks_path, f"seq{seq}")):
                full_mask_name = f"seq{seq}/{mask_name}"
                mask = torch.from_numpy(cv2.imread(osp.join(gt_endovis_masks_path, full_mask_name),cv2.IMREAD_GRAYSCALE))
                gt_endovis_masks[full_mask_name] = mask
    

    elif "cataracts" in data_root_dir.lower():
        
        # 这里的 fold_seq 必须和你 dataset.py 里的一模一样！
        # 假设你 dataset.py 里定义的是:
        fold_seq = {
            0: [1, 2],
            1: [3, 4],
            2: [5, 6],
            3: [7, 8]
        }
        
        # 根据 fold 获取序列列表
        if fold in fold_seq:
            seqs = fold_seq[fold]
        else:
            # 如果没传 fold 或者 fold 不对，默认返回空或者全集
            print(f"Warning: Invalid fold {fold} for Cataracts. Returning empty GT.")
            return {}

        # 路径结构: data_root/0/annotations/seqX/xxx.png
        # 注意: 这里的 "0" 对应 version=0
        gt_masks_path = osp.join(data_root_dir, "0", "annotations")
        
        if not osp.exists(gt_masks_path):
             print(f"Error: GT path not found: {gt_masks_path}")
             return {}

        print(f"Loading Cataracts GT masks from: {gt_masks_path} for seqs: {seqs}")

        for seq in seqs:
            seq_dir = osp.join(gt_masks_path, f"seq{seq}")
            
            if not osp.exists(seq_dir):
                print(f"Warning: Sequence directory not found: {seq_dir}")
                continue
            
            for mask_name in os.listdir(seq_dir):
                if not mask_name.endswith(".png"): continue
                
                # 构造 key: "seq1/case5180.png"
                full_mask_name = f"seq{seq}/{mask_name}"
                
                # 读取图片
                mask_path = osp.join(seq_dir, mask_name)
                mask_np = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                
                if mask_np is None:
                    print(f"Warning: Failed to read image: {mask_path}")
                    continue
                    
                mask = torch.from_numpy(mask_np)
                gt_endovis_masks[full_mask_name] = mask
            
            
    return gt_endovis_masks


def print_log(str_to_print, log_file):
    """Print a string and meanwhile write it to a log file
    """
    print(str_to_print)
    with open(log_file, "a") as file:
        file.write(str_to_print+"\n")