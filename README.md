# CSAM-HQ: A Multi-stage Refinement Framework for Surgical Instrument Segmentation based on SAM and Probabilistic Graphical Models
## 1. Method Overview
### Flowchart
![Method Architecture](./assert/总体框架.pdf)
> **Description:**
> The proposed architecture is a high-quality surgical instrument segmentation framework based on the Segment Anything Model (SAM). As shown in the figure, the pipeline consists of four key components:
>
> 1.  **Image Encoder Path (Frozen ❄️):**
>     Utilizes a pre-trained Transformer-based encoder (e.g., ViT) to extract image embeddings from input surgical frames. The weights are frozen during training to leverage robust feature representations from foundation models.
>
> 2.  **Prototype-Based Prompt Encoder (Trainable 🔥):**
>     Unlike standard geometric prompts (points/boxes), this module integrates a **Class Prototype Memory**. It retrieves specific instrument prototypes (e.g., Grasping forceps, Scissors, Vessel Sealer) from the embedding space to generate semantic-aware prompts. This path is fully trainable to adapt to the surgical domain.
>
> 3.  **HQ-Mask Decoder with HQ-Token:**
>     The core decoder fuses image embeddings with sparse prompt embeddings. It introduces a specialized **HQ-Token** into the input sequence of the Two-Way Transformer. This token, combined with **HQ-MLP**, captures high-frequency details to correct mask errors, fusing with the Base Mask to produce a refined High-Quality Mask.
>
> 4.  **CRF Refinement Module:**
>     A post-processing "Decoding and Optimization" stage that employs a Conditional Random Field (CRF). It calculates pairwise and unary terms to minimize an energy function, further refining the mask boundaries for precise segmentation output.

## 2. Project Structure
The directory structure of this project is as follows:


[CSAM-HQ]/
├── data/                  # 存放数据集的文件夹
│   ├── Endovis2017/       # 原始数据
│   └── Endovis2018/       # 原始数据
│   └── CATARACTS/         # 原始数据
├── models/                # 模型定义
│   ├── mask_decoder.py    # 网络结构代码
│   └── model_forword.py   # 自定义层
├── utils/                 # 工具函数
│   ├── dataset.py      # 数据加载器
│   └── utils.py         # 评价指标
├── configs/               # 配置文件 (yaml/json)
├── train.py               # 训练主脚本
├── inference.py                # 测试/推理脚本
├── requirements.txt       # 依赖包列表
├── README.md              # 说明文档
└── assets/                # 存放README用到的图片

## 3. Dataset 
This project uses the Endovis2017、Endovis2018、CATARACTS dataset.
Download:https://github.com/wenxi-yue/SurgicalSAM/blob/main/README.md、
Source: 点击这里下载数据集 (填写链接)
Format:sam_feature、image

Data Splitting 
We follow the splitting principle mentioned in the paper:
Training Set: [80% (6 seq)]
Validation Set: [20% (2 seq)]
Note:The CATARACTS datasets were divided according to the patients, with the Endovis2017 and Endovis2018 datasets strictly following the publicly disclosed division rules of the competition.

## 4. Training
Environment Setup
First, install the required dependencies: pip install -r requirements.txt

Implementation Details & Hyperparameters
We implemented the model using **PyTorch**. The input images were resized to **1024×1024**. Following the setup in Surgical-SAM, we utilized the **frozen SAM ViT-H encoder** as the backbone to extract high-level features. Only the **prompt encoder** and **mask decoder** were fine-tuned during training.

#### Key Hyperparameters

| Parameter | Value | Description |
| :--- | :--- | :--- |
| **Backbone** | SAM ViT-H | **Frozen** weights (not trained) |
| **Input Resolution** | 1024 $\times$ 1024 | - |
| **Batch Size** | 32 | - |
| **Optimizer** | Adam | - |
| **Loss Function** | Contrastive Loss | Temperature $\tau = 0.07$ |
| **Device** | 1x NVIDIA RTX 4090D | Single GPU training |

#### Dataset-Specific Settings

Different settings were applied depending on the dataset used (EndoVis2017, EndoVis2018, or CATARACTS):

| Dataset | Learning Rate (LR) | Prototype Count ($n$) |
| :--- | :---: | :---: |
| **EndoVis 2017** | 0.001 | 2 |
| **CATARACTS** | - | 2 |
| **EndoVis 2018** | 0.0001 | 4 |

> **Note:** As mentioned in the paper, we set the prototype count $n=2$ for EndoVis2017 and CATARACTS, and $n=4$ for EndoVis2018 due to the complexity differences.

Training Steps 
To train the model from scratch, run the following command: python train.py  --dataset endovis_2017\CATARACTS  --fold 0
python train.py  --dataset endovis_2018

Checkpoints
The trained model weights will be saved in the ./checkpoints/ directory.

## 5. Results

