# Infant Multimodal Cry Detection System

本项目是一个面向婴儿的**多模态哭泣检测系统**，融合音频、身体运动和面部视觉信息，通过机器学习模型对婴儿是否处于哭闹状态进行分类。系统支持单模态独立评估以及多模态早期融合（Early Fusion）与晚期融合（Late Fusion）策略，并采用分层分组交叉验证确保评估的严谨性。

---

## 整体架构

```
Step 1: 原始视频 ──→ InfantVision/body_seg_5.py
                         └──→ Body/*_motion_features.json

Step 2: 原始视频 ──→ InfantVision/facial_landmark_68.py
                         └──→ Face/*_face_landmarks.json

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

---

## 目录结构

```
├── InfantVision/                  # 视觉特征提取（body_seg_5.py / facial_landmark_68.py）
├── code/                          # 主分析管道（main.py / config.py / features/）
├── dataset/                       # 数据集（原始数据不开源，Features 已开源）
├── sample/                        # 样例数据（视频+音频，需运行 InfantVision 生成 JSON）
└── README.md
```

---

## 快速开始

### 1. 环境准备

```bash
# conda 创建环境
conda create -n MMInfant python=3.8 -y
conda activate MMInfant

# 安装依赖
pip install -r requirements.txt
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

本项目支持以下婴儿多模态数据集（**原始数据不开源，Features 已开源**）：

- **NICU50**：50 例 NICU 环境婴儿视频-音频记录
- **NEWBORN200**：200 例新生儿视频-音频记录

原始数据（视频、音频、标签）不开源，文件结构组织如下：

```
dataset/
└── NICU50/
    ├── data/          # .wav 音频文件
    ├── Label/         # 哭声标签 (.txt)
    ├── Scene/         # 场景标签 (.txt，可选）
    ├── Body/          # InfantVision 输出的身体运动 JSON
    └── Face/          # InfantVision 输出的面部关键点 JSON
```

> `sample/` 目录下有样例视频和音频，Body/Face JSON 需自行运行 InfantVision 生成后用于快速验证流程。

---

## 引用与致谢

本项目使用了以下开源库与模型：

- **scikit-learn / LightGBM / imbalanced-learn**：机器学习与数据平衡
- **librosa**：音频加载与处理
- **ModelScope / `iic/cv_resnet101_image-multiple-human-parsing`**：人体部位分割
- **[Infant Facial Landmark Detection and Tracking](https://github.com/ostadabbas/Infant-Facial-Landmark-Detection-and-Tracking)**：婴儿人脸检测与68点关键点回归
