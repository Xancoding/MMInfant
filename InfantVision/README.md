# InfantVision

**InfantVision** 是婴儿多模态哭泣检测系统的视觉特征提取子模块，负责从婴儿视频中提取身体运动特征与面部关键点特征，为下游的多模态机器学习分析提供视觉模态输入。

---

## 功能概述

本模块包含两条独立的视频处理流水线：

### 1. 人体部位分割与运动分析 (`body_seg_5.py`)

- **人体部位分割**：基于 [ModelScope `iic/cv_resnet101_image-multiple-human-parsing`](https://modelscope.cn/models/iic/cv_resnet101_image-multiple-human-parsing) 模型，对视频帧进行像素级人体部位解析，提取以下部位掩码：
  - 左臂 (Left-arm) / 右臂 (Right-arm)
  - 左腿 (Left-leg) / 右腿 (Right-leg)
  - 面部 (Face)
  - 躯干皮肤 (Torso-skin)
- **光流运动估计**：使用 Farneback 光流算法计算各部位及全局画面的运动向量。
- **输出**：
  - 带分割掩码叠加的演示视频 (`*_masked.avi`)
  - 首帧分割可视化图 (`*_seg_result.png`)
  - 逐帧运动特征 JSON (`*_motion_features.json`)

### 2. 面部 68 点关键点检测 (`facial_landmark_68.py`)

- **人脸检测与关键点回归**：基于 [Infant Facial Landmark Detection and Tracking](https://github.com/ostadabbas/Infant-Facial-Landmark-Detection-and-Tracking) 项目中的 YOLOv5-face 检测器与 HRNet 关键点模型，对婴儿面部进行边界框定位与 68 点关键点回归。
- **输出**：
  - 带关键点与边界框的演示视频 (`*_face_landmarks.avi`)
  - 首帧检测结果图 (`*_face_landmarks.png`)
  - 逐帧关键点与置信度 JSON (`*_face_landmarks.json`)

---

## 模型权重准备

运行脚本前，请将以下权重文件放置于 `ckpt/` 目录。点击下方链接前往 Google Drive 下载：

📁 **[MMInfant 模型权重](https://drive.google.com/drive/folders/1iM7lHMQksjks5_eYu8Xw6c_uAqejfbYL?usp=sharing)**

| 文件 | 大小 | 说明 |
|---|---|---|
| `face.pt` | 357 MB | YOLOv5-face 人脸检测器权重 |
| `hrnet-r90jt.pth` | 38 MB | HRNet 68点关键点模型权重 |

下载后将两个文件放入 `InfantVision/ckpt/` 目录即可。

---

## 使用方法

### 身体部位分割与运动分析

```bash
cd InfantVision
python body_seg_5.py
```

**默认配置**（可在脚本内修改）：
- 输入视频：`../sample/data/50cm3kg1.mp4`
- 输出目录：`../sample/Body/`
- 分割置信度阈值：`0.7`
- 最小掩码面积：`3000` 像素

### 面部关键点检测

```bash
cd InfantVision
python facial_landmark_68.py
```

**默认配置**（可在脚本内修改）：
- 输入视频：`../sample/data/50cm3kg1.mp4`
- 输出目录：`../sample/Face/`
- HRNet 配置：`./config/hrnet-r90jt.yaml`
- 检测器权重：`./ckpt/face.pt`
- 关键点权重：`./ckpt/hrnet-r90jt.pth`

## 引用

- 人体部位分割：[ModelScope `iic/cv_resnet101_image-multiple-human-parsing`](https://modelscope.cn/models/iic/cv_resnet101_image-multiple-human-parsing)
- 婴儿人脸关键点：[Ostadabbas Lab - Infant Facial Landmark Detection](https://github.com/ostadabbas/Infant-Facial-Landmark-Detection-and-Tracking)
