# CSAM-HQ: A Multi-stage Refinement Framework for Surgical Instrument Segmentation based on SAM and Probabilistic Graphical Models
## 1. Method Overview
### Flowchart
![Method Architecture](assert/总体框架.pdf)
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

│   ├── dataset.py         # 数据加载器

│   └── utils.py           # 评价指标

├── configs/               # 配置文件

├── train.py               # 训练主脚本

├── inference.py           # 测试/推理脚本

├── requirements.txt       # 依赖包列表

├── README.md              # 说明文档

└── assets/                # 存放README用到的图片

## 3. Dataset 
This project uses the Endovis2017、Endovis2018、CATARACTS dataset.

## Download ##

Source: (https://github.com/wenxi-yue/SurgicalSAM/blob/main/README.md

https://ieee-dataport.org/open-access/cataracts?check_logged_in=1)

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
| **CATARACTS** | 0.001 | 2 |
| **EndoVis 2018** | 0.0001 | 4 |

> **Note:** As mentioned in the paper, we set the prototype count $n=2$ for EndoVis2017 and CATARACTS, and $n=4$ for EndoVis2018 due to the complexity differences.

Training Steps 
To train the model from scratch, run the following command: python train.py  --dataset endovis_2017\CATARACTS  --fold 0
python train.py  --dataset endovis_2018

Inference Steps
To inference the model from scratch, run the following command: python inference.py  --dataset endovis_2017\CATARACTS  --fold 0
python inference.py  --dataset endovis_2018

Checkpoints
The trained model weights will be saved in the ./checkpoints/ directory.

## 5. Results
We compare our **CSAM-HQ** with state-of-the-art methods on three public datasets: EndoVis2017, EndoVis2018, and CATARACTS. The best results are highlighted in **bold**.

### 🏆 EndoVis2017 & EndoVis2018
Performance comparison on robotic surgical instrument segmentation.

| Dataset | Models | Ch. IoU $\uparrow$ | IoU $\uparrow$ | McIoU $\uparrow$ | BF | PF | LND | VS | GR | SI | MCS | UP | Params $\downarrow$ |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **EndoVis**<br>**2017** | ISINet | 55.62 | 52.20 | 28.96 | 38.70 | 38.50 | 50.09 | 27.43 | 2.10 | - | 28.72 | 12.56 | 162.52M |
| | TernausNet | 35.27 | 12.67 | 10.17 | 13.45 | 12.39 | 20.51 | 5.97 | 1.08 | - | 1.00 | 16.76 | 32.20M |
| | MF-TAPNet | 37.25 | 13.49 | 10.77 | 16.39 | 14.11 | 19.01 | 8.11 | 0.31 | - | 4.09 | 13.40 | 37.73M |
| | Surgical-SAM | 69.94 | 69.94 | **67.03** | 68.30 | **51.77** | 75.52 | 68.24 | 57.63 | - | **86.95** | **60.80** | **4.65M** |
| | **CSAM-HQ (Ours)** | **71.186** | **71.186** | 66.554 | **72.011** | 39.603 | **76.140** | **69.162** | **63.645** | - | 67.784 | 49.213 | 4.99M |
| | | | | | | | | | | | | | |
| **EndoVis**<br>**2018** | ISINet | 73.03 | 70.94 | 40.21 | 73.83 | 48.61 | 30.98 | 37.68 | - | 0.00 | - | - | 162.52M |
| | TernausNet | 46.22 | 39.87 | 14.19 | 44.20 | 4.67 | 0.00 | 0.00 | - | 0.00 | - | - | 32.20M |
| | MF-TAPNet | 67.87 | 39.14 | 24.68 | 69.23 | 6.10 | 11.68 | 14.00 | - | 0.91 | - | - | 37.73M |
| | Surgical-SAM | 71.233 | 71.233 | 62.473 | **79.139** | 51.353 | **89.271** | 84.427 | - | 31.693 | - | - | **4.65M** |
| | **CSAM-HQ (Ours)** | **74.536** | **74.536** | **68.405** | 75.003 | **56.531** | 88.631 | **93.334** | - | **46.096** | - | - | 4.99M |

> **Abbreviations:**
> *   **BF**: Bipolar Forceps, **PF**: Prograsp Forceps, **LND**: Large Needle Driver, **VS**: Vessel Sealer
> *   **GR**: Grasping Retractor (2017 only), **SI**: Suction Instrument (2018 only)
> *   **MCS**: Monopolar Curved Scissors (2017 only), **UP**: Ultrasound Probe (2017 only)

### 👁️ CATARACTS
Generalization performance comparison on cataract surgery dataset.

| Models | Ch. IoU $\uparrow$ | IoU $\uparrow$ | McIoU $\uparrow$ | Spatula | PT | LI | CF | IK | SK | KF | Params |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Surgical-SAM | 55.59 | 55.59 | 57.296 | 57.042 | 61.91 | 57.29 | 50.53 | 64.876 | **56.59** | 1.01 | **4.65M** |
| **CSAM-HQ (Ours)** | **57.57** | **57.57** | **61.67** | **57.32** | **66.19** | **58.91** | **54.60** | **72.17** | 53.43 | **17.35** | 4.99M |

> **Abbreviations:**
> **PT**: Phacoemulsification Tip, **LI**: Lens Injector, **CF**: Capsulorhexis Forceps, **IK**: Incision Knife, **SK**: Slit Knife, **KF**: Katena Forceps.

### 🎨 Visual Results (Qualitative Analysis)

To demonstrate the superior performance of our method, we visualize the segmentation results on challenging surgical scenes.

#### 1. Comparison on EndoVis2017
The following figure illustrates the qualitative segmentation results. Each row represents a distinct challenging surgical scene. Compared to other state-of-the-art methods (ISINet, TernausNet, Surgical-SAM), **CSAM-HQ** (Ours) produces masks with sharper boundaries and fewer artifacts.

<p align="可视化">
  <img src="assets/vis_endovis.png" width="95%" alt="EndoVis Visualization">
</p>

#### 2. Generalization on CATARACTS (Unseen Dataset)
We further evaluate the robustness of our model on the **unseen** CATARACTS dataset to test cross-domain generalization.

<p align="白内障">
  <img src="assets/vis_cataracts.png" width="95%" alt="CATARACTS Visualization">
</p>

> **Observation:** As shown above, compared to the Surgical-SAM baseline, **CSAM-HQ** demonstrates superior robustness in cross-domain scenarios. It accurately captures **thin instruments** and **fine tips** that are often missed or fragmented by the baseline model.
