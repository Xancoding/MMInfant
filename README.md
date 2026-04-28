# Infant Multimodal Cry Detection System

本项目是一个面向婴儿的**多模态哭泣检测系统**，融合音频、身体运动和面部视觉信息，通过机器学习模型对婴儿是否处于哭闹状态进行分类。系统支持单模态独立评估以及多模态早期融合（Early Fusion）与晚期融合（Late Fusion）策略，并采用分层分组交叉验证确保评估的严谨性。

---

## 整体架构

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         原始数据采集                                      │
│                    婴儿视频 (.mp4) + 音频 (.wav)                          │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
            ┌───────────────────────┴───────────────────────┐
            ▼                                               ▼
┌───────────────────────┐                     ┌───────────────────────────┐
│     InfantVision/     │                     │         code/             │
│   (视觉特征提取子模块)  │                     │    (主分析管道子模块)      │
│                       │                     │                           │
│  • 身体部位分割        │                     │  • 音频特征提取            │
│  • 光流运动估计        │                     │  • 运动/面部高阶统计特征   │
│  • 面部68点关键点检测   │                     │  • 多模态分类与融合评估    │
└───────────────────────┘                     └───────────────────────────┘
            │                                               ▲
            │    Body/*_motion_features.json                │
            │    Face/*_face_landmarks.json                 │
            └───────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                        评估结果输出                                      │
│         Accuracy / Precision / Recall / Specificity / F1-Score          │
│              单模态 · Early Fusion · Late Fusion (Stacking)              │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 模块说明

| 模块 | 路径 | 核心功能 | 详细文档 |
|---|---|---|---|
| **视觉特征提取** | [`InfantVision/`](./InfantVision/) | 基于深度学习模型从婴儿视频中提取身体部位分割掩码、光流运动向量及面部68点关键点坐标 | [InfantVision/README.md](./InfantVision/README.md) |
| **主分析管道** | [`code/`](./code/) | 从原始音频与视觉JSON特征中提取高阶统计特征，基于 SVM / RF / LightGBM 进行单模态与多模态分类评估 | [code/README.md](./code/README.md) |

---

## 目录结构

```
.
├── InfantVision/                  # 视觉特征提取子模块
│   ├── body_seg_5.py              # 身体部位分割与运动分析
│   ├── facial_landmark_68.py      # 面部68点关键点检测
│   ├── ckpt/                      # 预训练模型权重
│   ├── models/                    # YOLOv5-face 模型定义
│   ├── lib/                       # HRNet 关键点模型库
│   └── README.md                  # 视觉模块详细文档
│
├── code/                          # 主分析管道子模块
│   ├── main.py                    # 主控入口
│   ├── config.py                  # 全局配置
│   ├── data_loader.py             # 数据加载与特征缓存
│   ├── evaluator.py               # 模型评估器
│   ├── utils.py                   # 通用工具
│   ├── features/                  # 三模态特征提取
│   │   ├── Feature_Extraction_Audio.py
│   │   ├── Feature_Extraction_Body.py
│   │   └── Feature_Extraction_Face.py
│   └── README.md                  # 主分析管道详细文档
│
├── dataset/                       # 真实数据集（Features 已开源）
│   ├── NEWBORN200/
│   └── NICU50/
│
├── sample/                        # 样例数据（真实视频+音频，用于快速验证流程）
│   ├── data/                      # 原始音视频
│   ├── Body/                      # InfantVision 身体运动输出
│   └── Face/                      # InfantVision 面部关键点输出
│
└── README.md                      # 本文件（项目总览）
```

---

## 快速开始

### 1. 环境准备

```bash
# 克隆仓库后，安装视觉模块依赖
cd InfantVision
pip install torch torchvision opencv-python numpy pillow tqdm matplotlib scipy modelscope

# 安装主分析管道依赖
cd ../code
pip install numpy scipy matplotlib tqdm scikit-learn lightgbm imbalanced-learn librosa torch
```

### 2. 准备模型权重（视觉模块）

参考 [InfantVision/README.md](./InfantVision/README.md) 的「模型权重准备」章节，将 `face.pt` 和 `hrnet-r90jt.pth` 放置于 `InfantVision/ckpt/` 目录。

### 3. 运行视觉特征提取

```bash
cd InfantVision
python body_seg_5.py         # 生成 Body/*_motion_features.json
python facial_landmark_68.py # 生成 Face/*_face_landmarks.json
```

### 4. 配置并运行主分析管道

```bash
cd ../code
# 1. 编辑 config.py，设置 dataDir 指向你的数据集
# 2. 编辑 main.py，在 tasks 列表中启用需要的评估任务
python main.py
```

更多配置与使用细节，请参阅 [code/README.md](./code/README.md)。

---

## 数据集

本项目支持以下婴儿多模态数据集：

- **NICU50**：50 例 NICU 环境婴儿视频-音频记录
- **NEWBORN200**：200 例新生儿视频-音频记录

数据集目录结构要求：

```
dataset/
└── NICU50/
    ├── data/          # .wav 音频文件
    ├── Label/         # 哭声标签 (.txt)
    ├── Scene/         # 场景标签 (.txt，可选)
    ├── Body/          # 身体运动 JSON（由 InfantVision 生成）
    └── Face/          # 面部关键点 JSON（由 InfantVision 生成）
```

---

## 典型工作流

```
Step 1: 原始视频 ──→ InfantVision/body_seg_5.py
                         └──→ Body/subject_motion_features.json

Step 2: 原始视频 ──→ InfantVision/facial_landmark_68.py
                         └──→ Face/subject_face_landmarks.json

Step 3: 原始音频 + JSON 特征 ──→ code/main.py
                                      ├──→ 音频特征提取
                                      ├──→ 运动特征提取
                                      ├──→ 面部特征提取
                                      └──→ 分类评估
                                             ├── 单模态 (Audio / Motion / Face)
                                             ├── Early Fusion (特征拼接)
                                             └── Late Fusion (Stacking)
```

---

## 引用与致谢

本项目使用了以下开源库与模型：

- **pyAudioAnalysis**：音频特征提取基础库
- **scikit-learn / LightGBM / imbalanced-learn**：机器学习与数据平衡
- **librosa**：音频加载与处理
- **ModelScope / `iic/cv_resnet101_image-multiple-human-parsing`**：人体部位分割
- **[Infant Facial Landmark Detection and Tracking](https://github.com/ostadabbas/Infant-Facial-Landmark-Detection-and-Tracking)**：婴儿人脸检测与68点关键点回归
